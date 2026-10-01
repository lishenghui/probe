"""Opt-in CUDA-graph greedy decoding for a fixed Transformers model/adapter.

Requires a model supporting StaticCache and a 4-D additive attention mask.
Prefill stays eager; decode, argmax, position and mask updates share one graph.
Create a new runner after changing weights, adapters, batch size or capacity.
"""

import torch


class StaticDecodeRunner:
    """Reusable fixed-batch greedy generation, always emitting ``new_tokens``.

    This deliberately does not implement EOS stopping, sampling, beams, padding,
    or adapter switching. It is suitable for fixed-length throughput workloads.
    Returned token tensors are owned by the caller; graph buffers stay internal.
    """

    @torch.inference_mode()
    def __init__(self, model, example_ids, new_tokens):
        from transformers import StaticCache

        if (example_ids.ndim != 2 or not example_ids.is_cuda
                or example_ids.dtype != torch.long or new_tokens < 1):
            raise ValueError('Expected CUDA [batch, prefix] IDs and new_tokens >= 1')
        if model.training:
            raise ValueError('Call model.eval() before capturing inference')
        self.model = model
        self.batch, self.prefix = example_ids.shape
        if self.prefix == 0:
            raise ValueError('A nonempty unpadded prefix is required')
        self.new_tokens = new_tokens
        self.capacity = self.prefix + new_tokens + 4
        self.cache = StaticCache(config=model.config, max_cache_len=self.capacity)
        self.token = torch.empty((self.batch, 1), device=example_ids.device, dtype=torch.long)
        self.position = torch.full_like(self.token, self.prefix)
        self.columns = torch.arange(self.capacity, device=example_ids.device)
        self.mask = torch.empty((self.batch, 1, 1, self.capacity),
                                device=example_ids.device, dtype=model.dtype)
        self.prefix_mask = torch.ones_like(example_ids)
        self._prefill(example_ids)
        # Plan selection, Triton compilation and cache allocations must finish
        # before stream capture. These warmups operate on disposable KV state.
        for _ in range(2):
            self._step()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()
        self.graph_logits = self.step_logits
        # CUDA graphs retain addresses, not Python references to external
        # weights. The packed prefill path may rebind base_layer.weight on the
        # next request. Keep captured storage alive for the runner's lifetime.
        self._captured_weights = tuple(model.parameters()) + tuple(model.buffers())

    def _mask(self):
        self.mask.copy_(torch.where(
            self.columns[None, :] <= self.position,
            0.0, torch.finfo(self.mask.dtype).min,
        )[:, None, None, :])

    def _prefill(self, input_ids):
        self.cache.reset()
        out = self.model(input_ids=input_ids, attention_mask=self.prefix_mask,
                         past_key_values=self.cache, use_cache=True, logits_to_keep=1)
        self.token.copy_(out.logits[:, -1].argmax(-1, keepdim=True))
        self.position.fill_(self.prefix)
        self._mask()

    def _step(self):
        out = self.model(input_ids=self.token, position_ids=self.position,
                         attention_mask=self.mask, past_key_values=self.cache,
                         use_cache=True, logits_to_keep=1)
        self.step_logits = out.logits[:, -1]
        self.token.copy_(out.logits[:, -1].argmax(-1, keepdim=True))
        self.position.add_(1)
        self._mask()

    @torch.inference_mode()
    def validate(self, input_ids, tolerance=0.01):
        """Compare graph/eager logits under the same teacher-forced trajectory.

        Argmax near a tie may change with bf16 GEMM rounding. Testing matched
        inputs at every position separates numerical error from that divergence.
        """
        self._prefill(input_ids)
        tokens, logits = [], []
        for _ in range(1, self.new_tokens):
            tokens.append(self.token.clone())
            self._step()
            logits.append(self.step_logits.clone())
        self._prefill(input_ids)
        max_error = 0.0
        for token, reference in zip(tokens, logits):
            self.token.copy_(token)
            self.graph.replay()
            error = float(torch.linalg.vector_norm(self.graph_logits.float() - reference.float())
                          / torch.linalg.vector_norm(reference.float()).clamp_min(1e-8))
            max_error = max(max_error, error)
        if max_error > tolerance:
            raise AssertionError(f'Captured decode logit relative L2 error {max_error:.4g} > {tolerance}')
        return max_error

    @torch.inference_mode()
    def __call__(self, input_ids, *, replay=True):
        if (input_ids.shape != (self.batch, self.prefix)
                or input_ids.device != self.token.device or input_ids.dtype != torch.long):
            raise ValueError('Input must match the captured batch, prefix, device and int64 dtype')
        self._prefill(input_ids)
        result = torch.empty((self.batch, self.prefix + self.new_tokens),
                             device=input_ids.device, dtype=torch.long)
        result[:, :self.prefix].copy_(input_ids)
        result[:, self.prefix:self.prefix+1].copy_(self.token)
        for step in range(1, self.new_tokens):
            if replay:
                self.graph.replay()
            else:
                self._step()
            result[:, self.prefix+step:self.prefix+step+1].copy_(self.token)
        return result

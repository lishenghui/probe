"""Record (observation, action) pairs from an OpenVLA rollout, or replay them.

The closed-loop result says compression costs 11.7% of task success, but that
number cannot separate two very different causes: a large per-step policy error,
or a small one that the environment feedback loop amplifies.  Telling them apart
needs the compressed policy evaluated on the *original* policy's observations,
so no divergence has accumulated yet.

`env_obs` is the right thing to record: the preprocessing that turns it into
model inputs is deterministic and adapter-independent, so replaying it gives all
variants byte-identical inputs without reaching into the model internals.

Appended to the vendored model module at job time; enabled by RQ3_DUMP_DIR.
"""

import os
import pathlib

import torch

_DUMP = os.environ.get("RQ3_DUMP_DIR")
_LIMIT = int(os.environ.get("RQ3_DUMP_LIMIT", "64"))

if _DUMP:
    _dir = pathlib.Path(_DUMP)
    _dir.mkdir(parents=True, exist_ok=True)
    _orig_predict = OpenVLAOFTForRLActionPrediction.predict_action_batch
    _count = {"n": 0}

    def _snapshot(env_obs):
        out = {}
        for key, value in env_obs.items():
            if torch.is_tensor(value):
                out[key] = value.detach().to("cpu").clone()
            else:
                out[key] = value
        return out

    def _probing_predict(self, *args, **kwargs):
        env_obs = kwargs.get("env_obs")
        # Snapshot before the call: predict_action_batch unsqueezes the image
        # tensors in place, so afterwards the shapes no longer match what the
        # environment produced.
        before = _snapshot(env_obs) if env_obs is not None else None
        result = _orig_predict(self, *args, **kwargs)
        if before is not None and _count["n"] < _LIMIT:
            actions = result[0] if isinstance(result, tuple) else result
            if not torch.is_tensor(actions):
                actions = torch.as_tensor(actions)
            torch.save({"env_obs": before, "actions": actions.detach().to("cpu").clone()},
                       _dir / f"step_{_count['n']:05d}.pt")
            _count["n"] += 1
            if _count["n"] >= _LIMIT:
                print(f"[rq3-probe] captured {_LIMIT} observation batches -> {_dir}",
                      flush=True)
        return result

    OpenVLAOFTForRLActionPrediction.predict_action_batch = _probing_predict
    print(f"[rq3-probe] observation capture armed, limit={_LIMIT}, dir={_dir}", flush=True)

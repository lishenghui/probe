"""Batched, resumable EditReward-Bench evaluation for the Qwen3-VL vLLM backend."""

import argparse
import hashlib
import json
import math
import os
import time

from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from editscore import EditScore


def cache_key(key):
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def load_cache(path):
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                cache[item["key"]] = item["result"]
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", default="EditScore/EditReward-Bench")
    parser.add_argument("--dataset-disk")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora", help="optional LoRA adapter; omit for the base-model control")
    parser.add_argument("--merged-cache", help="temporary directory for the merged LoRA model")
    parser.add_argument("--batch-size", type=int, default=8, help="candidate pairs per Python batch")
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    scorer = EditScore(
        backbone="qwen3vl_vllm", model_name_or_path=args.model,
        lora_path=args.lora, cache_dir=args.merged_cache, score_range=25,
        temperature=0.7, max_model_len=4096,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens, num_pass=1,
    )
    dataset = load_from_disk(args.dataset_disk) if args.dataset_disk and os.path.exists(args.dataset_disk) else load_dataset(args.benchmark_dir, split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    cache_dir = os.path.join(args.result_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "qwen3vl_vllm.jsonl")
    cache = load_cache(cache_path)
    pending = []
    started = time.time()

    def flush():
        nonlocal pending
        if not pending:
            return
        outputs = scorer.batch_evaluate(
            [[item[2], item[3]] for item in pending],
            [item[1] for item in pending],
        )
        with open(cache_path, "a", encoding="utf-8") as handle:
            for item, result in zip(pending, outputs):
                key = cache_key(item[0])
                cache[key] = result
                handle.write(json.dumps({"key": key, "result": result}, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        done = len(cache)
        elapsed = time.time() - started
        print(f"progress pairs={done} rate={done / max(elapsed, 1):.3f}/s", flush=True)
        pending = []

    for row in tqdm(dataset, desc="queue/score rows"):
        input_image = row["input_image"].convert("RGB")
        for key, output in zip(row["key"], row["output_images"]):
            if cache_key(key) in cache:
                continue
            output = output.convert("RGB").resize(input_image.size)
            pending.append((key, row["instruction"], input_image, output))
            if len(pending) >= args.batch_size:
                flush()
    flush()

    out_root = os.path.join(args.result_dir, "qwen3vl_vllm")
    initialized_outputs = set()
    for row_idx, row in enumerate(dataset):
        scores = [cache[cache_key(key)] for key in row["key"]]
        dimension = row["dimension"]
        path = os.path.join(out_root, row["task_type"], f"{dimension}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "key": row["key"], "idx": row_idx,
            "score": [score[dimension] for score in scores],
            "SC_reasoning": [score["SC_score_reasoning"] for score in scores],
            "PQ_reasoning": [score["PQ_score_reasoning"] for score in scores],
            "input_image": None, "output_images": [None, None],
        }
        # Replace stale/smoke output on the first row for each file, then append
        # the remaining rows belonging to the same task/dimension.
        mode = "a" if path in initialized_outputs else "w"
        initialized_outputs.add(path)
        with open(path, mode, encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"completed rows={len(dataset)} pairs={len(cache)} elapsed={time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()

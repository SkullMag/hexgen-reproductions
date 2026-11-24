"""Freeze a scaled OpenR1-Math request sample for paired replay."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


DATASET = "open-r1/OpenR1-Math-220k"
DATASET_REVISION = "e4e141ec9dea9f8326f4d347be56105859b2bd68"
SHARD = "data/train-00000-of-00010.parquet"
SHARD_SHA256 = "ccc3a95e4875dc4cff9bec2f0fb1c93e4c9cbb24fb24853fd0b96c416e1a263c"
MODEL_REVISION = "ec052fda178e241c7c443468d2fa1db6618996be"
SAMPLE_SEED = "figure6-openr1-v1"
PILOT_SIZE = 32
MEASURED_SIZE = 480
MAX_PROMPT_TOKENS = 1024
MIN_OUTPUT_TOKENS = 64
MAX_OUTPUT_TOKENS = 256


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze(shard, tokenizer_path, output):
    if digest(shard) != SHARD_SHA256:
        raise ValueError("OpenR1 shard SHA-256 differs from its pinned revision")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    table = pq.read_table(shard, columns=["uuid", "problem", "solution"])
    ranked = sorted(
        (row for row in table.to_pylist() if row["uuid"] and row["problem"] and row["solution"]),
        key=lambda row: hashlib.sha256(f'{SAMPLE_SEED}:{row["uuid"]}'.encode()).digest(),
    )
    rows, seen = [], set()
    for row in ranked:
        problem = row["problem"].strip()
        if not problem or problem in seen:
            continue
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}], tokenize=True, add_generation_prompt=True,
        )
        if len(input_ids) > MAX_PROMPT_TOKENS:
            continue
        solution_tokens = len(tokenizer.encode(row["solution"], add_special_tokens=False))
        new_tokens = min(MAX_OUTPUT_TOKENS, max(MIN_OUTPUT_TOKENS, solution_tokens))
        rows.append({"id": row["uuid"], "input_ids": input_ids, "max_new_tokens": new_tokens})
        seen.add(problem)
        if len(rows) == PILOT_SIZE + MEASURED_SIZE:
            break
    if len(rows) != PILOT_SIZE + MEASURED_SIZE:
        raise ValueError("Shard did not supply the requested number of unique prompts")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "dataset": DATASET, "dataset_revision": DATASET_REVISION, "config": "default",
        "split": "train", "shard": SHARD, "shard_sha256": SHARD_SHA256,
        "tokenizer_model_revision": MODEL_REVISION,
        "tokenizer_files": {p.name: digest(p) for p in sorted(tokenizer_path.iterdir()) if p.is_file()},
        "sample_seed": SAMPLE_SEED, "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "min_output_tokens": MIN_OUTPUT_TOKENS, "max_output_tokens": MAX_OUTPUT_TOKENS,
        "output_length_source": "Qwen token count of the human solution, bounded to 64–256",
        "traces": {},
    }
    for name, requests in (("pilot", rows[:PILOT_SIZE]), ("measured", rows[PILOT_SIZE:])):
        path = output / f"{name}.jsonl"
        with path.open("w") as run:
            for request in requests:
                run.write(json.dumps(request, separators=(",", ":")) + "\n")
        manifest["traces"][name] = {"sha256": digest(path), "requests": len(requests)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.shard, args.tokenizer, args.output), indent=2))

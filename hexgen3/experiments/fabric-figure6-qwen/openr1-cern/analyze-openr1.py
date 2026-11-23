"""Verify paired inputs and summarize captured FABRIC replays."""
import argparse
import csv
import json
import math
import tarfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("plan", type=Path)
parser.add_argument("captures", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
methods = ("hexgen2", "hexgen3")
args.output.mkdir(parents=True, exist_ok=True)


def read_run(pair, method):
    with tarfile.open(args.captures / f"{pair['run']}-{method}.tgz") as archive:
        with archive.extractfile(f"{method}/summary.json") as stream:
            summary = json.load(stream)
        with archive.extractfile(f"{method}/requests.jsonl") as stream:
            requests = sorted(map(json.loads, stream), key=lambda row: row["index"])
    if not summary["capture_complete"] or len(requests) != pair["offered"]:
        raise ValueError(f"{pair['run']} has an incomplete {method} request log")
    return summary, requests


def nearest_rank(values, fraction):
    return sorted(values)[math.ceil(len(values) * fraction) - 1] if values else None


def identity(request):
    return tuple(request[key] for key in (
        "id", "prompt_id", "scheduled_offset_s", "input_tokens", "requested_output_tokens"
    ))


plan = json.loads(args.plan.read_text())
rows = []
for pair in plan:
    if not all((args.captures / f"{pair['run']}-{method}.tgz").is_file() for method in methods):
        continue
    baseline, candidate = (read_run(pair, method) for method in methods)
    for key in ("trace_sha256", "arrivals_sha256", "clock_domain", "duration_seconds"):
        if baseline[0][key] != candidate[0][key]:
            raise ValueError(f"{pair['run']} has different {key} between arms")
    if list(map(identity, baseline[1])) != list(map(identity, candidate[1])):
        raise ValueError(f"{pair['run']} has different offered requests between arms")
    for method, (summary, requests) in zip(methods, (baseline, candidate)):
        within = sum(row["success"] and row["within_measurement_window"] for row in requests)
        lag = [row["submission_lag_seconds"] for row in requests if "submission_lag_seconds" in row]
        rows.append({
            "run": pair["run"], "seed": pair["seed"], "load_multiplier": pair["multiplier"],
            "offered_rate": pair["rate"], "method": method, "arm": pair["order"].index(method) + 1,
            "window_seconds": pair["duration"], "offered": pair["offered"], "in_window": within,
            "succeeded": summary["successful"], "failures": summary["failures"],
            "requests_per_second": summary["throughput_requests_per_second"],
            "tokens_per_second": summary["throughput_tokens_per_second"],
            "p99_seconds": summary["successful_requests_p99_submit_to_completion_seconds"],
            "p99_submission_lag_ms": 1000 * nearest_rank(lag, 0.99) if lag else None,
        })
if not rows:
    raise SystemExit("No complete replay pairs")
with (args.output / "paired-results.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

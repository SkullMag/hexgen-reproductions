"""Freeze the OpenR1 replay order and arrival schedules from the baseline pilot."""
import argparse
import json
from pathlib import Path
from statistics import median

from hexgen2.comparison_client import poisson_arrivals

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("pilot", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--duration", type=int, default=240)
parser.add_argument("--primary-duration", type=int, default=480)
parser.add_argument("--multipliers", type=float, nargs="+", default=(0.85, 0.50, 1.00, 1.10, 1.40))
parser.add_argument("--seeds", type=int, nargs="+", choices=(61, 305, 997), default=(61, 305, 997))
parser.add_argument("--preceding-pairs", type=int, default=0)
args = parser.parse_args()
pilot = json.loads(args.pilot.read_text())
if not pilot["capture_complete"]:
    raise SystemExit("The baseline pilot capture is incomplete")
requests = [json.loads(line) for line in args.pilot.with_name("requests.jsonl").read_text().splitlines()]
failures = [row for row in requests if not row["success"]]
if len(failures) != pilot["failures"] or any(
    row.get("error_type") != "TimeoutError" or row["within_measurement_window"] for row in failures
):
    raise SystemExit("A baseline pilot failure affected the measurement window")
first_failure = min((row["completed_offset_s"] for row in failures), default=float("inf"))
completions = sorted(
    row["completed_offset_s"] for row in requests
    if row["success"] and row["completed_offset_s"] < first_failure
)
if len(completions) < 2:
    raise SystemExit("The baseline pilot needs two completions under backlog")
interval = median(second - first for first, second in zip(completions, completions[1:]))
capacity = 1 / interval

args.output.mkdir(parents=True, exist_ok=False)
(args.output / "calibration.json").write_text(json.dumps({
    "pilot_run": pilot["run_id"],
    "completion_offsets_before_first_timeout_seconds": completions,
    "median_inter_completion_seconds": interval,
    "estimated_hexgen2_requests_per_second": capacity,
    "fixed_window_requests_per_second": pilot["throughput_requests_per_second"],
}, indent=2) + "\n")
plan = []
for seed in args.seeds:
    for multiplier in args.multipliers:
        rate = round(capacity * multiplier, 8)
        run = f"openr1-r{round(multiplier * 100):03}-s{seed:03}"
        duration = args.primary_duration if multiplier == 0.85 else args.duration
        schedule = poisson_arrivals(rate, duration, seed)
        if len(schedule["offsets_s"]) > 160:
            raise SystemExit(f"{run} needs more than 160 request rows for its seed")
        (args.output / f"{run}.json").write_text(json.dumps(schedule, indent=2) + "\n")
        order = ["hexgen2", "hexgen3"] if (args.preceding_pairs + len(plan)) % 2 == 0 else ["hexgen3", "hexgen2"]
        plan.append({
            "run": run,
            "rate": rate,
            "multiplier": multiplier,
            "seed": seed,
            "duration": duration,
            "offered": len(schedule["offsets_s"]),
            "order": order,
        })
(args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

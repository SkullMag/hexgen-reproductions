"""Freeze the OpenR1 replay order and arrival schedules from the baseline pilot."""
import argparse
import json
from pathlib import Path

from hexgen2.comparison_client import poisson_arrivals

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("pilot", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--duration", type=int, default=240)
parser.add_argument("--primary-duration", type=int, default=480)
args = parser.parse_args()
pilot = json.loads(args.pilot.read_text())
capacity = pilot["throughput_requests_per_second"]
if pilot["failures"] or not pilot["capture_complete"] or not capacity:
    raise SystemExit("The baseline pilot did not establish a rate")

args.output.mkdir(parents=True, exist_ok=False)
plan = []
for seed in (61, 305, 997):
    for multiplier in (0.85, 0.50, 1.00, 1.10, 1.40):
        rate = round(capacity * multiplier, 8)
        run = f"openr1-r{round(multiplier * 100):03}-s{seed:03}"
        duration = args.primary_duration if multiplier == 0.85 else args.duration
        schedule = poisson_arrivals(rate, duration, seed)
        if len(schedule["offsets_s"]) > 480:
            raise SystemExit(f"{run} needs more than 480 request rows")
        (args.output / f"{run}.json").write_text(json.dumps(schedule, indent=2) + "\n")
        order = ["hexgen2", "hexgen3"] if len(plan) % 2 == 0 else ["hexgen3", "hexgen2"]
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

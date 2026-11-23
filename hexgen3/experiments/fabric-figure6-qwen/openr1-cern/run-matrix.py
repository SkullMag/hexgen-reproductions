"""Run paired OpenR1 replays sequentially, alternating the first arm."""
import argparse
import datetime
import json
import shlex
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("plan", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--switch", required=True)
parser.add_argument("--gpu", required=True)
parser.add_argument("--cpu", required=True)
parser.add_argument("--pilot-runtime")
parser.add_argument("--limit", type=int)
parser.add_argument("--from-index", type=int, default=0)
args = parser.parse_args()
plan = json.loads(args.plan.read_text())
args.output.mkdir(parents=True, exist_ok=True)


def log(message):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    line = f"{now} {message}"
    print(line, flush=True)
    with (args.output / "commands.log").open("a") as stream:
        stream.write(line + "\n")


def invoke(command, *, check=True, stdout=None):
    log(shlex.join(command))
    return subprocess.run(command, check=check, stdout=stdout)


start_index = args.from_index
end_index = start_index + args.limit if args.limit is not None else len(plan)
active = ("hexgen2", args.pilot_runtime) if args.pilot_runtime else None
for index, pair in enumerate(plan[start_index:end_index], start=start_index):
    run = pair["run"]
    for method in pair["order"]:
        control = ["bash", args.switch, args.gpu, args.cpu]
        if not active or active[0] != method:
            if active:
                invoke([*control, "stop", *active])
            runtime = f"matrix-{run}-{method}"
            invoke([*control, "start", method, runtime, str(29670 + index)])
            active = (method, runtime)
            warm = (
                'cd "$HOME/openr1-controller"; '
                f"bash replay-openr1.sh {method} warm-{runtime} finite --request-timeout 100"
            )
            invoke(["bash", args.cpu, warm])
        replay = (
            'cd "$HOME/openr1-controller"; '
            f"bash replay-openr1.sh {method} {run} online --duration {pair['duration']} "
            f"--arrivals schedules/{run}.json --request-timeout 450 --drain-timeout 460"
        )
        result = invoke(["bash", args.cpu, replay], check=False)
        with (args.output / f"{run}-{method}.tgz").open("wb") as archive:
            collect = f'tar -czf - -C "$HOME/openr1-controller/runs/{run}" {method}'
            invoke(["bash", args.cpu, collect], stdout=archive)
        if result.returncode:
            log(f"{run} {method} recorded failed requests; resetting its runtime")
            invoke([*control, "stop", *active])
            active = None
if active:
    invoke(["bash", args.switch, args.gpu, args.cpu, "stop", *active])

#!/usr/bin/env bash
set -euo pipefail
method="$1"
run="$2"
mode="$3"
shift 3
cd "$HOME/openr1-controller"
case "$method" in
    hexgen2) endpoint=http://10.86.6.11:8210; format=hexgen2 ;;
    hexgen3) endpoint=http://10.86.6.11:8300; format=sglang ;;
    *) exit 2 ;;
esac
case "$mode" in
    finite) trace=qualification.jsonl; inflight=3 ;;
    offline) trace=workload/pilot.jsonl; inflight=4 ;;
    online) trace=workload/measured.jsonl; inflight=256 ;;
    *) exit 2 ;;
esac
export PYTHONPATH=hexgen2/source
exec .venv/bin/python -m hexgen2.comparison_client --method "$method" --request-format "$format" \
    --run-id "$run" --mode "$mode" --endpoint "$endpoint" --trace "$trace" \
    --output "runs/$run/$method" --inflight "$inflight" --request-timeout 240 --drain-timeout 250 "$@"

#!/usr/bin/env bash
set -euo pipefail
gpu="$1"; cpu="$2"; operation="$3"; method="$4"; run="$5"
[[ "$run" =~ ^[a-zA-Z0-9_-]+$ ]]
case "$method" in
    hexgen2) root=hexgen2-qwen ;;
    hexgen3) root=hexgen3-openr1 ;;
    *) exit 2 ;;
esac

start() {
    local node="$1" role="$2" invocation="$3"
    bash "$gpu" "$node" "cd \"\$HOME/$root\"; mkdir -p '$run'; if test -f '$run/$role.pid'; then exit 1; fi;
        nohup setsid $invocation > '$run/$role.log' 2>&1 </dev/null & echo \$! > '$run/$role.pid'"
}

stop() {
    local node="$1"
    shift
    bash "$gpu" "$node" "cd \"\$HOME/$root\";
        for role in $*; do
            pid=\$(cat \"$run/\$role.pid\" 2>/dev/null) || continue;
            if [ \"\$(ps -o pgid= -p \"\$pid\" | tr -d ' ')\" = \"\$pid\" ]; then
                kill -TERM -- \"-\$pid\" 2>/dev/null || true;
                for attempt in 1 2 3 4 5; do
                    kill -0 -- \"-\$pid\" 2>/dev/null || break;
                    sleep 1;
                done;
                kill -KILL -- \"-\$pid\" 2>/dev/null || true;
            fi;
        done"
}

await_http() {
    local url="$1"
    local filter="$2"
    bash "$cpu" "set -o pipefail; for attempt in \$(seq 1 65); do
        if curl -fsS --max-time 2 '$url' 2>/dev/null | { cat; echo; } | grep -Eq '$filter'; then exit 0; fi;
        sleep 2;
    done; exit 1"
}

case "$operation:$method" in
    start:hexgen2)
        port="${6:-29666}"
        [[ "$port" =~ ^[0-9]+$ ]]
        start prefill rank-0 "bash launch-baseline.sh 0 '$run' '$port'" & r0=$!
        start decode-1 rank-1 "bash launch-baseline.sh 1 '$run' '$port'" & r1=$!
        start decode-2 rank-2 "bash launch-baseline.sh 2 '$run' '$port'" & r2=$!
        wait "$r0"; wait "$r1"; wait "$r2"
        await_http http://10.86.6.11:8210/health '"ready": *true'
        ;;
    stop:hexgen2)
        bash "$cpu" "curl -fsS --max-time 3 -H 'Content-Type: application/json' -d '{}' http://10.86.6.11:8210/shutdown" || true
        stop prefill rank-0 & r0=$!
        stop decode-1 rank-1 & r1=$!
        stop decode-2 rank-2 & r2=$!
        wait "$r0"; wait "$r1"; wait "$r2"
        ;;
    start:hexgen3)
        start prefill prefill 'bash launch-hexgen3.sh prefill info' & pre=$!
        start prefill router 'bash launch-hexgen3.sh router info' & router=$!
        start decode-2 stepmesh-scheduler 'bash launch-hexgen3.sh stepmesh-scheduler info'
        bash "$gpu" decode-2 'for attempt in $(seq 1 25); do
            if sudo -n rdma resource show cm_id | grep -q "state LISTEN.*10.86.6.13:8123"; then exit 0; fi;
            sleep 1;
        done; exit 1'
        start decode-1 ffn-0 'bash launch-hexgen3.sh ffn-0 info' & r0=$!
        start decode-2 ffn-1 'bash launch-hexgen3.sh ffn-1 info' & r1=$!
        wait "$r0"; wait "$r1"
        await_http http://10.86.6.12:8303/health '^'
        start decode-2 attn 'bash launch-hexgen3.sh attn info'
        wait "$pre"; wait "$router"
        await_http http://10.86.6.11:8301/health '^'
        await_http http://10.86.6.13:8302/health '^'
        await_http http://10.86.6.11:8300/health '^'
        ;;
    stop:hexgen3)
        stop prefill prefill router & pre=$!
        stop decode-1 ffn-0 & r0=$!
        stop decode-2 attn ffn-1 stepmesh-scheduler & r1=$!
        wait "$pre"; wait "$r0"; wait "$r1"
        ;;
    *) exit 2 ;;
esac

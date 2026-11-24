#!/usr/bin/env bash
set -euo pipefail
role="$1"
log_level="${2:-info}"
cd "$HOME/hexgen3-openr1"
export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export GLOO_SOCKET_IFNAME="$(cat "$HOME/hexgen2-qwen/rdma-iface.txt")"
export SGLANG_LOCAL_IP_NIC="$GLOO_SOCKET_IFNAME"
export NCCL_SOCKET_IFNAME="=$GLOO_SOCKET_IFNAME"
export NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_IB_HCA="=mlx5_0:1" NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=SYS NCCL_NET_GDR_READ=1 NCCL_IB_PCI_RELAXED_ORDERING=0
export NCCL_CUMEM_ENABLE=0 NCCL_CUMEM_HOST_ENABLE=0
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=700
model="$HOME/hexgen2-qwen/checkpoint"
common=(--model-path "$model" --dtype bfloat16 --context-length 1792
    --chunked-prefill-size 2048 --max-prefill-tokens 2048
    --disable-overlap-schedule --disable-cuda-graph --disable-radix-cache
    --disable-custom-all-reduce --skip-server-warmup --skip-tokenizer-init
    --attention-backend triton --sampling-backend pytorch --log-level "$log_level")
pd=(--disaggregation-transfer-backend mooncake --disaggregation-ib-device mlx5_0
    --disaggregation-bootstrap-port 9001)
case "$role" in
    prefill)
        export SGLANG_HOST_IP=10.86.6.11
        exec .venv/bin/python -u -m sglang.launch_server "${common[@]}" "${pd[@]}" \
            --host "$SGLANG_HOST_IP" --port 8301 --disaggregation-mode prefill \
            --mem-fraction-static 0.86 --max-total-tokens 4096
        ;;
    router)
        exec .venv/bin/python -u -m sglang.srt.disaggregation.mini_lb \
            --prefill http://10.86.6.11:8301 --prefill-bootstrap-ports 9001 \
            --decode http://10.86.6.13:8302 --host 10.86.6.11 --port 8300
        ;;
    stepmesh-scheduler|attn|ffn-0|ffn-1)
        export AFD_SCHED_HOST=10.86.6.12
        export DMLC_PS_ROOT_URI=10.86.6.13
        export MLC_INTERFACE="$GLOO_SOCKET_IFNAME" DMLC_INTERFACE="$GLOO_SOCKET_IFNAME"
        export DMLC_ENABLE_RDMA=ibverbs
        export PS_VERBOSE="${PS_VERBOSE:-1}" STEPMESH_SCHEDULER_STARTED=1
        export DMLC_PS_ROOT_PORT=8123 DMLC_NUM_SERVER=1 DMLC_NUM_WORKER=1
        export DMLC_GROUP_SIZE=1 DMLC_NODE_RANK=0 STEPMESH_GPU=0
        if [[ "$role" == stepmesh-scheduler ]]; then
            export DMLC_NODE_HOST=10.86.6.13
            exec .venv/bin/python -u -c 'from sglang.srt.layers.afd import stepmesh_scheduler; stepmesh_scheduler()'
        fi
        if [[ "$role" == attn ]]; then
            export SGLANG_HOST_IP=10.86.6.13 DMLC_NODE_HOST=10.86.6.13
            max_running_requests=16
            if [[ "${OPENR1_BATCH_ONE:-0}" == 1 ]]; then max_running_requests=1; fi
            exec .venv/bin/python -u -m sglang.launch_server "${common[@]}" "${pd[@]}" \
                --host "$SGLANG_HOST_IP" --port 8302 --disaggregation-mode decode \
                --afd-perspective attn --afd-mirco-batch 2 --max-total-tokens 6144 \
                --mem-fraction-static 0.90 --max-running-requests "$max_running_requests"
        fi
        rank="${role##*-}"
        export SGLANG_HOST_IP="10.86.6.$((12 + rank))"
        export DMLC_NODE_HOST="$SGLANG_HOST_IP"
        exec .venv/bin/python -u -m sglang.launch_server "${common[@]}" \
            --host "$SGLANG_HOST_IP" --port 8303 --disaggregation-mode null \
            --afd-perspective ffn --afd-mirco-batch 2 --max-total-tokens 1792 \
            --mem-fraction-static 0.83 --tp 2 --nnodes 2 --node-rank "$rank" \
            --dist-init-addr 10.86.6.12:29722
        ;;
    *)
        printf 'Unknown HexGen-3 role %s\n' "$role" >&2
        exit 2
        ;;
esac

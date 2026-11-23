#!/usr/bin/env bash
set -euo pipefail
rank="$1"; run="$2"; port="$3"
cd "$HOME/hexgen2-qwen"
export PYTHONPATH="$PWD/hexgen2/source"
export RANK="$rank" WORLD_SIZE=3 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=10.86.6.11 MASTER_PORT="$port"
export GLOO_SOCKET_IFNAME="$(cat rdma-iface.txt)"
export NCCL_SOCKET_IFNAME="=$GLOO_SOCKET_IFNAME"
export NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_IB_HCA="=mlx5_0:1" NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=SYS NCCL_NET_GDR_READ=1
export NCCL_IB_PCI_RELAXED_ORDERING=0 NCCL_CUMEM_ENABLE=0 NCCL_CUMEM_HOST_ENABLE=0
export NCCL_LAUNCH_ORDER_IMPLICIT=1 NCCL_LAUNCH_RACE_FATAL=1 CUDA_MODULE_LOADING=LAZY
export LD_PRELOAD="$PWD/nccl-runtime/nvidia/nccl/lib/libnccl.so.2"
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
mkdir -p "$run"
export NCCL_DEBUG_FILE="$PWD/$run/nccl-$rank.log"
phase=decode
if [[ "$rank" == 0 ]]; then phase=prefill; fi
flags=()
if [[ "${4:-}" == audit ]]; then flags+=(--audit); fi
exec .venv/bin/python -u -m hexgen2.comparison_worker --method hexgen2 --three-rank-qwen \
  --phase "$phase" --model-dir "$PWD/checkpoint" --baseline-root "$PWD" --max-batch 2 \
  --batch-wait-ms 10 --bind "10.86.6.$((11 + rank))" --port 8210 --output "$run" "${flags[@]}"

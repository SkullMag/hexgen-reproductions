#!/usr/bin/env bash
set -euo pipefail

rank="$1"
run="$2"
port="$3"
cd "$HOME/hexgen2-qwen"
mkdir -p "$run"
export PYTHONPATH="$PWD/hexgen2/source"
export RANK="$rank" WORLD_SIZE=3 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=10.86.6.11 MASTER_PORT="$port"
export NCCL_SOCKET_IFNAME="=$(cat rdma-iface.txt)"
export NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_IB_HCA="=mlx5_0:1" NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=SYS NCCL_NET_GDR_READ=1
export NCCL_IB_PCI_RELAXED_ORDERING=0 NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0 NCCL_LAUNCH_ORDER_IMPLICIT=1 NCCL_LAUNCH_RACE_FATAL=1
export LD_PRELOAD="$PWD/nccl-runtime/nvidia/nccl/lib/libnccl.so.2"
[[ -r "$LD_PRELOAD" ]]
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,ENV
export NCCL_DEBUG_FILE="$PWD/$run/nccl-rank-$rank.log"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

if [[ "${4:-}" == "link-check" ]]; then
    exec .venv/bin/python -u link-check.py
fi

exec .venv/bin/python -u -m hexgen2.qualify_qwen --device cuda \
  --model-dir "$PWD/checkpoint" \
  --trace "$PWD/hexgen3/experiments/fabric-figure6-qwen/qualification.jsonl" \
  --output "$PWD/$run/qualification.json"

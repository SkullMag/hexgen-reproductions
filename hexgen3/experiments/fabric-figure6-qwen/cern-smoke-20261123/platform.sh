#!/usr/bin/env bash
set -euo pipefail
sudo -n env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=180 update
kernel="$(uname -r)"
modules="linux-modules-nvidia-570-server-open-$kernel"
if apt-cache show "$modules" 2>/dev/null | grep -q '^Package:'; then
    kernel_package="$modules"
else
    kernel_package="linux-headers-$kernel"
fi
sudo -n env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=180 install -y --no-install-recommends \
  "$kernel_package" nvidia-driver-570-server-open ibverbs-utils rdma-core perftest python3-venv
sudo -n modprobe nvidia
sudo -n modprobe nvidia_uvm
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
sudo -n modprobe nvidia-peermem || echo "nvidia-peermem modprobe failed"
lsmod | grep -E 'nvidia|mlx5_ib'
rdma link

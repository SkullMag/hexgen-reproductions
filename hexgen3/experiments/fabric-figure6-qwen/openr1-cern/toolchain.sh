#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/hexgen3-openr1"
curl -fsS --retry 3 -o cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo -n dpkg -i cuda-keyring.deb
sudo -n apt-get -o DPkg::Lock::Timeout=120 update
sudo -n env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=120 install -y --no-install-recommends \
  cuda-nvcc-12-6 cuda-cudart-dev-12-6 cuda-driver-dev-12-6 build-essential \
  librdmacm-dev libibverbs-dev libnuma-dev libzmq3-dev pkg-config cmake ninja-build python3-dev
/usr/local/cuda-12.6/bin/nvcc --version
date -u +%FT%TZ > toolchain.ready

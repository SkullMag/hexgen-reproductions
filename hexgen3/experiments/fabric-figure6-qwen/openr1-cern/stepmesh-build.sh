#!/usr/bin/env bash
set -euo pipefail
root="$HOME/hexgen3-openr1"
cd "$root"
until [[ -s runtime.ready && -s toolchain.ready ]]; do
  for step in runtime3 toolchain; do
    if ! kill -0 "$(cat "$step.pid")" 2>/dev/null; then tail -25 "$step.log"; exit 1; fi
  done
  sleep 8
done
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
export MAX_JOBS=3 PIP_NO_CACHE_DIR=1
cuda_wheel_headers="$(python - <<'PY'
from pathlib import Path
import site
paths = sorted(path for directory in site.getsitepackages()
               for path in (Path(directory) / "nvidia").glob("*/include") if path.is_dir())
assert any((path / "cusparse.h").is_file() for path in paths)
print(":".join(map(str, paths)))
PY
)"
export CPATH="$cuda_wheel_headers${CPATH:+:$CPATH}"
sudo -n env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=120 install -y --no-install-recommends \
  cuda-nvrtc-dev-12-6 cuda-nvtx-12-6
python -m pip install setuptools wheel
cd comm
mkdir -p deps/lib deps/include
# Link Ubuntu's ZeroMQ as the system dependency of the extension.
ln -sfn /usr/lib/x86_64-linux-gnu/libzmq.so deps/lib/libzmq.a
# Compile the released static library without linking the Torch CMake targets.
# BuildExtension below links Torch and CUDA when it builds the Python extension.
if [[ ! -f CMakeLists.release.txt ]]; then cp CMakeLists.txt CMakeLists.release.txt; fi
cat > CMakeLists.txt <<'CMAKE'
cmake_minimum_required(VERSION 3.22)
project(af LANGUAGES CXX)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
execute_process(COMMAND ${Python_EXECUTABLE} -c
    "import torch, pathlib; print(pathlib.Path(torch.__file__).parent); print(int(torch.compiled_with_cxx11_abi()))"
    OUTPUT_VARIABLE torch_config OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" torch_config "${torch_config}")
list(GET torch_config 0 torch_root)
list(GET torch_config 1 torch_abi)
file(GLOB sources "src/*.cc" "src/backend/*.cc")
add_library(af STATIC ${sources})
set_target_properties(af PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(af PRIVATE _GLIBCXX_USE_CXX11_ABI=${torch_abi} DMLC_USE_ZMQ
    DMLC_USE_CUDA STEPMESH_USE_GDR DMLC_USE_RDMA STEPMESH_ENABLE_TRACE)
target_include_directories(af PRIVATE include "${torch_root}/include"
    "${torch_root}/include/torch/csrc/api/include" "$ENV{CUDA_HOME}/include" ${Python_INCLUDE_DIRS})
CMAKE
cmake -S . -B cmake_build -DPython_EXECUTABLE="$root/.venv/bin/python" -DCMAKE_BUILD_TYPE=Release
cmake --build cmake_build --target af --parallel 3
# The extension's CUDA wait kernels need native sm_86 on the A40.
python - <<'PY'
from pathlib import Path
path = Path("setup.py")
text = path.read_text()
original = "'arch=compute_80,code=sm_80',"
if "'arch=compute_86,code=sm_86'" not in text:
    assert text.count(original) == 1
    path.write_text(text.replace(original, original + " '-gencode', 'arch=compute_86,code=sm_86',"))
PY
python -m pip install -v --no-build-isolation --no-deps .
python - <<'PY'
import torch
import fserver_lib
from mooncake.engine import TransferEngine
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(), fserver_lib.__file__)
PY
date -u +%FT%TZ > "$root/stepmesh.ready"

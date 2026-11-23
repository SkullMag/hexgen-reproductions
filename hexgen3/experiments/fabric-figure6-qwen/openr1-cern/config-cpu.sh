#!/usr/bin/env bash
set -euo pipefail
ports=(/sys/bus/pci/devices/"$1"/net/*)
[[ ${#ports[@]} == 1 && -e "${ports[0]}" ]]
nic="${ports[0]##*/}"
sudo -n ip link set "$nic" up
sudo -n ip address replace 10.86.6.14/24 dev "$nic"
cd "$HOME/openr1-controller"
python3 -m venv .venv || sudo -n apt-get -y install python3-venv
[[ -x .venv/bin/pip ]] || python3 -m venv .venv
.venv/bin/pip install 'aiohttp==3.11.18' 'psutil==6.1.0'
ping -c 1 -W 3 10.86.6.11
date -u +%FT%TZ > ready

#!/usr/bin/env bash
set -euo pipefail
bdf="$1"
addr="$2"
ports=(/sys/bus/pci/devices/"$bdf"/net/*)
[[ ${#ports[@]} == 1 && -e "${ports[0]}" ]]
nic="${ports[0]##*/}"
sudo -n ip link set dev "$nic" up
sudo -n ip addr replace "$addr/24" dev "$nic"
ip -br addr show dev "$nic"
rdma link
ls -l /sys/class/infiniband
printf '%s\n' "$nic" > "$HOME/hexgen2-qwen/rdma-iface.txt"

# CERN HexGen-2 Qwen qualification

On November 23, 2026, the three-rank CUDA run on FABRIC slice `hexgen2-qwen-cern-20261123` matched all **25 generated tokens across the three frozen requests** against the colocated BF16 reference. The A40 received the token sequences from the second decode rank and compared them exactly. All three ranks exited with status zero. The controller recorded 07:51:49–07:52:10 UTC, including model loading and process shutdown. [Qualification result](qualification-result.json) · [Run evidence](run-evidence.json)

| Rank | Worker | GPU | Role | L2 address | NIC |
| --- | --- | --- | --- | --- | --- |
| 0 | CERN `cern-w1` | A40 | Full prefill and colocated reference | `10.86.6.11` | `enp8s0`, PCI `0000:08:00.0` |
| 1 | CERN `cern-w3` | A30 | Decode layers 0–11 | `10.86.6.12` | `enp8s0`, PCI `0000:08:00.0` |
| 2 | CERN `cern-w6` | A30 | Decode layers 12–23 | `10.86.6.13` | `enp7s0`, PCI `0000:07:00.0` |

The hosts used Ubuntu 22.04.5, kernel `5.15.0-185-generic`, driver `580.178.04`, PyTorch `2.5.1+cu124`, Transformers `4.51.3`, and preloaded NCCL `2.26.5+cuda12.9`. The `NIC_Basic` interfaces use ConnectX-6 VFs on the reserved L2 bridge. The RoCE v2 IPv4 GID was index 3 on `mlx5_0:1` for all three guests. NCCL logged `NET/IB/GDRDMA` sends and receives for all six directed links in both the Qwen and link-check runs.

| Directed rank link | GPU tensor bandwidth, Gbit/s |
| --- | ---: |
| 0 → 1 | 56.1 |
| 0 → 2 | 63.9 |
| 1 → 0 | 44.7 |
| 1 → 2 | 28.9 |
| 2 → 0 | 74.1 |
| 2 → 1 | 74.2 |

The link check timed eight 64 MiB BF16 tensors per direction at the receiver after two warmup transfers. It checked the received tensor contents and exited with status zero on all ranks. [Directed link measurements](run-evidence.json) · [Link check command](link-check.py)

## Inputs and commands

- The checkpoint is `Qwen/Qwen1.5-MoE-A2.7B-Chat` at `ec052fda178e241c7c443468d2fa1db6618996be`. Each guest verified all eight weight shards against the pinned Hugging Face hashes and reported the same full model identity `5d322584d22c982c73a014afebfbc41072351fa35c837ecae667ae95310931f1`. The run used BF16 with greedy decoding and fixed output lengths of 1, 8, and 16 tokens.
- The frozen request file had SHA-256 `3c64200215bc5794443b828e72017bff59f3860e214e81c8de29d1ba34e5a3d9`. [Frozen requests](../qualification.jsonl)
- The source is `acac14736a473b203fa1ee879cce2f32a3e57116`. Its NCCL shim reuses the group communicator using the earlier CERN pilot's batched P2P approach. The initial CUDA attempt matched all tokens, then both decode ranks hit a PyTorch 2.5 invalid-device-ordinal error during destruction of individual peer communicators. The rerun uses the shared communicator for both the KV packets and pipeline tensors. The Qwen stage and transport CPU tests passed, three tests in total. [Earlier CERN compat code](../../../../hexgen2/experiments/cern-gpudirect-20261109/original13b/runtime/transport_compat.py)

Stage the repository's `hexgen2/source/hexgen2` directory and frozen request file under `$HOME/hexgen2-qwen`, retaining their repository paths. Place these scripts in that directory. On each guest run `bash platform.sh`, `bash runtime.sh`, and `bash nccl-setup.sh`. Bring up the bridge using `bash network.sh <NIC-PCI-address> <L2-address>` with the table above. [Reservation plan](../cern-smoke-plan.json)

```sh
cd "$HOME/hexgen2-qwen"
PYTHONPATH=hexgen2/source HF_HUB_DISABLE_XET=1 .venv/bin/python -m hexgen2.fetch_model \
  --repo Qwen/Qwen1.5-MoE-A2.7B-Chat --revision ec052fda178e241c7c443468d2fa1db6618996be \
  --output "$PWD/checkpoint"
```

The recorded run resumed partial shard downloads as verified byte ranges and distributed the completed shards over the private bridge. Each receiver verified size and SHA-256. The command above downloads the same pinned inputs directly.

Run the following once per worker in parallel, substituting the assigned rank. The launcher pins CUDA device 0, the rendezvous address, the RoCE GID, and NCCL runtime.

```sh
bash qualify-rank.sh "$RANK" qualification-02 29662
bash qualify-rank.sh "$RANK" link-check-03 29664 link-check
```

This run qualifies the three-GPU BF16 inference path. At 07:45:52 UTC FABRIC reported both CERN A40s allocated. The four-GPU comparison, separate CPU load generator, and HexGen-3 qualification remain pending.

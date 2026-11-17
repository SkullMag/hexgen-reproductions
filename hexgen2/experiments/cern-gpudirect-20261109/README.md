# CERN GPUDirect pilot

HexGen and HexGen‑2 ran OpenLLaMA‑13B FP16 on the same 1 A40 + 2 A30 FABRIC fleet with GPUDirect RDMA. Each workload used one matched 24-request queue. The four throughput pairs and comparison with Figure 6 are in the [report](report.html).

- The four result JSON files contain throughput, output audits, and the same-queue host-staged comparison. The protocol files hash the frozen synthetic inputs and record the released optimizer placements.
- [original13b/](original13b/) contains the 155 source files from the measured deployment. The deployed manifest has SHA-256 `44498806b4ca7b428d14d8fac7e8b34ff270af14a73f5c762eb1d84361984662`. The model checkpoint is `openlm-research/open_llama_13b` at `b6d7fde8392250730d24cc2fcfa3b7e5f9a03ce8`.
- [controller/](controller/) preserves the executed pilot scripts and GPUDirect launcher. They still depend on FABRIC provisioning, staging, and benchmark helpers from the original campaign workspace. A fresh GPU rerun from this repository requires porting those helpers and providing a new FABRIC reservation and prepared model weights.

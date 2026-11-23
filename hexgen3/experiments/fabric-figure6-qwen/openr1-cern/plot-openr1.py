"""Render the matched OpenR1 throughput and latency curves."""
import argparse
import csv

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("summary")
parser.add_argument("output")
args = parser.parse_args()
with open(args.summary, newline="") as stream:
    summary = list(csv.DictReader(stream))

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), layout="constrained")
for method, label, color, marker in (
    ("hexgen2", "HexGen-2", "#315d88", "o"),
    ("hexgen3", "HexGen-3", "#bc5223", "s"),
):
    rows = [row for row in summary if row["method"] == method]
    load = [float(row["load_multiplier"]) for row in rows]
    axes[0].plot(load, [60 * float(row["requests_per_second"]) for row in rows],
                 label=label, color=color, marker=marker)
    latency = [(x, float(row["p99_seconds"])) for x, row in zip(load, rows) if row["p99_seconds"]]
    if latency:
        xs, values = zip(*latency)
        axes[1].plot(xs, values, label=label, color=color, marker=marker)
axes[0].set_ylabel("Completed requests per minute")
axes[1].set_ylabel("P99 end-to-end latency (s)")
for ax in axes:
    ax.set_xlabel("Offered load / HexGen-2 pilot rate")
    ax.set_xticks(sorted({float(row["load_multiplier"]) for row in summary}))
    ax.grid(axis="y", color="#d8dde4", linewidth=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside upper right", ncols=2, frameon=False)
fig.text(0.013, 0.967, "OpenR1-Math · CERN FABRIC, 1 A40 + 2 A30s",
         ha="left", va="top", color="#17243a", fontsize=11, fontweight="bold")
fig.savefig(args.output, dpi=180, bbox_inches="tight")

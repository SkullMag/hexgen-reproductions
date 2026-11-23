"""Render the matched OpenR1 throughput and latency curves."""
import argparse
import csv

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("summary")
parser.add_argument("output")
parser.add_argument("--normalized-primary", action="store_true")
args = parser.parse_args()
with open(args.summary, newline="") as stream:
    summary = list(csv.DictReader(stream))


if args.normalized_primary:
    primary = {row["method"]: row for row in summary if float(row["load_multiplier"]) == 0.85}
    methods = ("hexgen2", "hexgen3")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), layout="constrained")
    for ax, metric, label in zip(
        axes, ("requests_per_second", "p99_seconds"),
        ("Completed requests per cost", "P99 end-to-end latency"),
    ):
        raw = [float(primary[method][metric]) for method in methods]
        values = [value / raw[0] for value in raw]
        bars = ax.bar(("HexGen-2", "HexGen-3"), values,
                      color=("#315d88", "#bc5223"), width=0.52)
        ax.bar_label(bars, labels=[f"{value:.2f}×" for value in values], padding=4)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Relative to HexGen-2")
        ax.set_ylim(0, max(values) * 1.25)
        ax.grid(axis="y", color="#d8dde4", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    count = primary["hexgen2"]["succeeded"]
    fig.suptitle(f"OpenR1-Math · 0.85× offered load · {count} completed per system",
                 fontweight="bold", fontsize=12)
    fig.supxlabel("Shared CERN pool of one A40 and two A30s", fontsize=9)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    raise SystemExit(0)

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

"""Assign each replay seed a disjoint ordered segment of the frozen workload."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("trace", type=Path)
args = parser.parse_args()
source = args.trace.read_bytes().splitlines(keepends=True)
if len(source) != 480:
    raise SystemExit("Expected 480 frozen requests")
manifest = []
for index, seed in enumerate((61, 305, 997)):
    start, end = index * 160, (index + 1) * 160
    # The first seed keeps the byte-for-byte input already used by its first arm.
    path = args.trace if index == 0 else args.trace.with_name(f"measured-s{seed}.jsonl")
    contents = b"".join(source if index == 0 else source[start:end])
    if index:
        path.write_bytes(contents)
    manifest.append({"seed": seed, "eligible_range": [start, end], "trace": path.name,
                     "sha256": hashlib.sha256(contents).hexdigest()})
(args.trace.parent / "seed-traces.json").write_text(json.dumps(manifest, indent=2) + "\n")

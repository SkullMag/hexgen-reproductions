"""Run a predeclared same-queue pilot after GPUDirect passes qualification."""
from argparse import ArgumentParser
from itertools import groupby
import json
from statistics import median
import sys

# Capture the selected workload before the shared FABRIC controller binds sys.argv.
PILOT_PORTS = {'hphd': 31853, 'hpld': 31856, 'lphd': 31858, 'lpld': 31860}
parser = ArgumentParser()
parser.add_argument('--workload', choices=tuple(PILOT_PORTS), default='hphd')
PARTITION = parser.parse_args().workload

from qualify import OUT, HERE, GDR, REMOTE, cleanup, parallel, read, write
from common import CD, REFERENCE_NODE, checked, digest, put, run, wait

QUICK = HERE/'cern-fixed-quick'
CONFIRM = HERE/'cern-hphd-confirm'
LANES = ('hexgen-opt', 'hexgen2-opt')
WARM = 'hphd-warmup'
TRANSPORT = 'roce-hphd-cern-gdr-compat'
THRESHOLD = 0.17
INPUTS = ('cases', 'reference', 'protocol', WARM, PARTITION)


def main():
    assert read(OUT/'network-qualification.json')['qualified']
    protocol = read(CONFIRM/'confirmation-protocol.json')
    quick = read(QUICK/'fixed-protocol.json')
    prior = read(QUICK/'results.json')['trials'][PARTITION]
    assert prior['audit']['eligible'] and len(read(QUICK/'hphd.json')) == 24
    for name in INPUTS:
        target = OUT/f'{name}.json'
        if not target.exists():
            target.symlink_to(QUICK/target.name)
    for node in read(OUT/'fleet.json')['hosts']:
        target = OUT/'profiles'/node
        if not target.exists():
            target.symlink_to((QUICK/'profiles'/node).resolve())
    hashes = {name: digest(OUT/f'{name}.json') for name in INPUTS}
    assert all(hashes[PARTITION] == prior['rates'][lane]['workload_sha256'] for lane in LANES)
    assert all(protocol['placement'][lane] == quick['placement'][lane] for lane in LANES)
    receipt = OUT/('pilot-protocol.json' if PARTITION == 'hphd' else f'{PARTITION}-protocol.json')
    write(receipt, {'inputs': hashes, 'placement': {lane: protocol['placement'][lane] for lane in LANES},
          'batch': 4, 'gdr_env': GDR, 'source_manifest': protocol['source_manifest'], 'launch_order': LANES,
          **({'minimum_gain': THRESHOLD} if PARTITION == 'hphd' else {})})
    sys.argv = [sys.argv[0], str(OUT), '--cern', '--calibration-only']
    import run_fleet as fleet
    import compare
    fleet.TRANSPORT = TRANSPORT
    compare.wait = lambda node, label, seconds: wait(node, label, min(seconds, 230))
    source = (CONFIRM/'launch-roce-hphd-compat.py').read_text()
    assert source.count("NCCL_NET_GDR_LEVEL='LOC',") == 1
    assert source.count('trace13b/local-kv-20261108/cern-hphd-confirm/sample_cpu.py') == 1
    source = source.replace("NCCL_NET_GDR_LEVEL='LOC',", "NCCL_NET_GDR_LEVEL='PHB',NCCL_NET_GDR_READ='1',")
    source = source.replace('trace13b/local-kv-20261108/cern-hphd-confirm/sample_cpu.py', REMOTE+'/sample_cpu.py')
    launcher = OUT/f'launch-{TRANSPORT}.py'
    if launcher.exists():
        assert launcher.read_text() == source
    else:
        launcher.write_text(source)

    def stage(node):
        checked(run(node, "sudo -n modprobe nvidia_peermem", 20), node)
        for path in [launcher, fleet.CAMPAIGN/'sample_cpu.py',
                     *[OUT/f'{name}.json' for name in ('cases', WARM, PARTITION)]]:
            put(node, path, REMOTE+'/'+path.name)
        if node == REFERENCE_NODE:
            expected = digest(CONFIRM/'oracle-cpu-ulp.py')
            checked(run(node, CD+f"printf '%s  teacherforce_reference.py\\n' {expected} | sha256sum -c -", 20), node)

    parallel(stage)
    variants = {}
    for index, lane in enumerate(LANES):
        variants.update(fleet.run_arm(lane, protocol['placement'][lane], 'qual', 31851+index))
    cases = [{**compare.CASES[0], 'native_variants': variants}] if variants else []
    errors = compare.material_errors(compare.reference_replay('cern-gdr-pilot-qualification-reference', cases))
    assert not errors, errors
    qualification = {'passed': True, 'nonexact_variants_audited': len(variants)}
    saved = OUT/'model-qualification.json'
    if saved.exists():
        assert read(saved) == qualification
    else:
        write(saved, qualification)
    results, gaps, steps = {}, {}, {}
    base_port = PILOT_PORTS[PARTITION]
    for index, lane in enumerate(LANES):
        row = fleet.run_arm(lane, protocol['placement'][lane], PARTITION, base_port+index)
        assert fleet.cpu_gate(row), (lane, 'CPU pressure')
        target = OUT/'runs'/row['label']
        assert read(target/'plan.json')['expected_source_manifest_sha256'] == protocol['source_manifest']
        [path] = target.rglob('native-token-events-rank-2.jsonl')
        events = [json.loads(line) for line in path.open()]
        batches = [[event['utc_ns'] for event in batch] for _, batch in
                   groupby((e for e in events if e['phase'] == 'decode'), key=lambda e: e['ids'])]
        assert len(batches) == 6
        gaps[lane] = median((b[0]-a[-1])/1e9 for a,b in zip(batches, batches[1:]))
        steps[lane] = median((b-a)/1e6 for batch in batches for a,b in zip(batch,batch[1:]))
        results[lane] = row
    audit = fleet.audit(PARTITION, results, 'cern-gdr-pilot-audit')
    rates = fleet.summarize(results)
    gain = rates[LANES[1]]['wall_output_tokens_per_second']/rates[LANES[0]]['wall_output_tokens_per_second'] - 1
    destination = OUT/('results.json' if PARTITION == 'hphd' else f'{PARTITION}-results.json')
    write(destination, {'pilot': True, 'rates': rates, 'gain': gain,
          **({'advance_to_confirmation': gain >= THRESHOLD} if PARTITION == 'hphd' else {}),
          'audit': audit, 'median_decode_step_ms': steps, 'median_between_batches_seconds': gaps,
          'matched_host_staged_rates': {lane: prior['rates'][lane] for lane in LANES},
          'protocol_sha256': digest(receipt)})
    print(f'Audited {PARTITION.upper()} pilot:', {lane: rates[lane]['wall_output_tokens_per_second'] for lane in LANES}, f'{gain:+.2%}', flush=True)


if __name__ == '__main__':
    try:
        main()
    finally:
        cleanup()

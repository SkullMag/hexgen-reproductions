"""Bounded GPUDirect check on the three reserved CERN nodes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from campaign import HERE, NODES, read, write

OUT = Path(__file__).resolve().parent
REMOTE = 'trace13b/local-kv-20261108/cern-gdr-pilot'
CLEANUP = '--cleanup' in sys.argv
sys.argv = [sys.argv[0], str(OUT)]
write(OUT/'fleet.json', dict(read(HERE/'cern-fixed-quick/fleet.json'), prefix=OUT.name))
from common import CD, checked, collect, put, run

GDR = {'NCCL_NET_GDR_LEVEL': 'PHB', 'NCCL_NET_GDR_READ': '1', 'NCCL_CUMEM_ENABLE': '0',
       'NCCL_CUMEM_HOST_ENABLE': '0', 'NCCL_IB_PCI_RELAXED_ORDERING': '0'}


def parallel(action, nodes=NODES):
    with ThreadPoolExecutor(len(nodes)) as pool:
        futures = {node: pool.submit(action, node) for node in nodes}
        return {node: future.result() for node, future in futures.items()}


def cleanup():
    before = read(OUT/'network-before.json')

    def restore(node):
        if not before[node]:
            return checked(run(node, 'if test -d /sys/module/nvidia_peermem; then sudo -n modprobe -r nvidia_peermem; fi', 20), node)
        return ''
    parallel(restore)
    write(OUT/'network-restored.json', {'nodes': NODES})


def qualify():
    profiles = OUT/'profiles'
    profiles.mkdir(exist_ok=True)
    original = HERE/'profiles'
    rankmap = read(original/'rankmap.json')
    write(profiles/'rankmap.json', rankmap)
    source = (original/'launch_links.py').read_text().replace('trace13b/cern-ampere/profiles', REMOTE+'/profiles')
    source = source.replace("'29929'", "'31850'").replace("    env['LD_PRELOAD']", f"    env.update({GDR!r})\n    env['LD_PRELOAD']")
    (profiles/'launch_links.py').write_text(source)
    (profiles/'profile_links.py').write_text((original/'profile_links.py').read_text()
        .replace('trace13b/cern-ampere/profiles', REMOTE+'/profiles')
        .replace('timedelta(minutes=10)', 'timedelta(seconds=55)'))
    before = parallel(lambda node: checked(run(node, 'if test -d /sys/module/nvidia_peermem; then echo yes; else echo no; fi', 15), node).strip() == 'yes')
    write(OUT/'network-before.json', before)

    def probe(node):
        checked(run(node, CD+'mkdir -p '+REMOTE+'/profiles && sudo -n modprobe nvidia_peermem', 20), node)
        for name in ('rankmap.json', 'launch_links.py', 'profile_links.py'):
            put(node, profiles/name, REMOTE+'/profiles/'+name)
        cmd = ['timeout', '--kill-after=12s', '85s', '.venv/bin/python', '-u', REMOTE+'/profiles/launch_links.py', node]
        reply = run(node, CD+shlex.join(cmd), 105)
        row = next(row for row in rankmap if row['node'] == node)
        name = f"link-rank-{row['rank']}.log"
        write(profiles/f'{node}-exit.json', {'status': reply.returncode, 'stdout': reply.stdout, 'stderr': reply.stderr})
        collect(node, REMOTE+'/profiles/'+name, profiles/name)
        return reply.returncode

    try:
        statuses = parallel(probe)
        paths = {}
        for row in rankmap:
            rank = row['rank']
            log = (profiles/f'link-rank-{rank}.log').read_text()
            paths[rank] = {}
            for peer in range(len(rankmap)):
                if peer == rank:
                    continue
                for source, target in ((rank, peer), (peer, rank)):
                    pattern = rf'\b{source}\[\d+\] -> {target}\[\d+\] \[(?:send|receive)\] via NET/IB[^\n]*'
                    lines = re.findall(pattern, log)
                    paths[rank][f'{source}->{target}'] = bool(lines) and all('GDRDMA' in line for line in lines)
        qualified = all(code == 0 for code in statuses.values()) and all(ok for p in paths.values() for ok in p.values())
        if qualified:
            collect(NODES[0], REMOTE+'/profiles/links.json', profiles/'links.json')
            qualified = len(read(profiles/'links.json')['records']) == 12
        write(OUT/'network-qualification.json', {'qualified': qualified, 'status': statuses, 'direct_by_rank_and_direction': paths})
        print('GPUDirect qualification:', qualified, 'rank exit statuses:', statuses, flush=True)
        if qualified:
            return
    except BaseException:
        cleanup()
        raise
    cleanup()
    raise SystemExit(1)


if __name__ == '__main__':
    if CLEANUP:
        cleanup()
    else:
        qualify()

import json,os,signal,subprocess,sys,time
from pathlib import Path
node=sys.argv[1];ranks=json.loads(Path('original13b/plans/rankmap-seven.json').read_text())
root=Path('original13b/profiles');root.mkdir(parents=True,exist_ok=True);children=[]
def stop(*unused):
    for child in children:
        if child.poll() is None:child.terminate()
signal.signal(signal.SIGTERM,stop)
for row in ranks:
    if row['node']!=node:continue
    env=os.environ.copy();env.update(RANK=str(row['rank']),LOCAL_RANK=str(row['device']),WORLD_SIZE=str(len(ranks)),
        MASTER_ADDR=ranks[0]['ip'],MASTER_PORT='29919',CUDA_MODULE_LOADING='LAZY',
        GLOO_SOCKET_IFNAME=row['interface'],NCCL_SOCKET_IFNAME=row['interface'],NCCL_IB_DISABLE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env['LD_PRELOAD']=str(Path('artifacts/nccl-runtime/nvidia/nccl/lib/libnccl.so.2').resolve())
    log=(root/f'link-rank-{row["rank"]}.log').open('wb')
    children.append(subprocess.Popen([sys.executable,'-u','original13b/runtime/profile_links.py'],env=env,stdout=log,stderr=subprocess.STDOUT))
(root/f'links-pids-{node}.json').write_text(json.dumps({'launcher':os.getpid(),'children':[p.pid for p in children]}))
while any(c.poll() is None for c in children):
    if any(c.poll() not in [None,0] for c in children):stop();break
    time.sleep(.2)
for c in children:
    try:c.wait(10)
    except subprocess.TimeoutExpired:c.kill();c.wait()
(root/f'links-exit-{node}.json').write_text(json.dumps([c.returncode for c in children]))

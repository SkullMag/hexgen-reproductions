"""Launch a fixed released-code qualification plan on one owned guest."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--node',required=True);p.add_argument('--plan',required=True)
p.add_argument('--output',required=True);p.add_argument('--port',type=int,required=True);a=p.parse_args()
plan=json.loads(Path(a.plan).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
children=[]
def stop(*unused):
    for child in children:
        if child.poll() is None:child.terminate()
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
for row in plan['ranks']:
    if row['node']!=a.node:continue
    env=os.environ.copy();env.update(RANK=str(row['rank']),LOCAL_RANK=str(row['device']),WORLD_SIZE=str(len(plan['ranks'])),
        MASTER_ADDR=plan['ranks'][0]['ip'],MASTER_PORT=str(a.port),NCCL_SOCKET_IFNAME=row['interface'],GLOO_SOCKET_IFNAME=row['interface'],
        NCCL_IB_DISABLE='1',NCCL_LAUNCH_ORDER_IMPLICIT='1',NCCL_LAUNCH_RACE_FATAL='1',CUDA_MODULE_LOADING='LAZY',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',NCCL_DEBUG='WARN')
    env['LD_PRELOAD']=str(Path('artifacts/nccl-runtime/nvidia/nccl/lib/libnccl.so.2').resolve())
    log=(out/f'rank-{row["rank"]}-console.log').open('wb')
    children.append(subprocess.Popen([sys.executable,'-u','original13b/runtime/native_runner.py','--plan',a.plan,'--output',a.output],env=env,stdout=log,stderr=subprocess.STDOUT))
(out/f'pids-{a.node}.json').write_text(json.dumps({'launcher':os.getpid(),'children':[c.pid for c in children]}))
while any(c.poll() is None for c in children):
    if any(c.poll() not in [None,0] for c in children):stop();break
    time.sleep(.2)
for c in children:
    try:c.wait(15)
    except subprocess.TimeoutExpired:c.kill();c.wait()
codes=[c.returncode for c in children]
(out/f'exit-{a.node}.json').write_text(json.dumps(codes))
sys.exit(0 if all(c==0 for c in codes) else 3)

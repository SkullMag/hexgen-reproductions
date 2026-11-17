"""Measured GPU NCCL payload paths, including same-host PCIe and private LAN."""
import os,time,json,statistics
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from transport_compat import send,recv

def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(1)
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    dist.init_process_group('nccl',timeout=timedelta(minutes=10),device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    wire=dist.group.WORLD
    control=dist.new_group(list(range(world)),backend='gloo')
    torch.cuda.init();x=torch.ones(1,device='cuda');dist.all_reduce(x,group=wire);torch.cuda.synchronize()
    records=[]
    for source in range(world):
        for target in range(source+1,world):
            for size in [16,1<<20,8<<20,64<<20]:
                tensor=torch.ones(size//2,device='cuda',dtype=torch.float16);torch.cuda.synchronize();times=[]
                for i in range(8):
                    dist.barrier(group=control);started=time.perf_counter()
                    if rank==source:
                        send(tensor,dst=target,group=wire);recv(tensor,src=target,group=wire);torch.cuda.synchronize()
                    elif rank==target:
                        recv(tensor,src=source,group=wire);send(tensor,dst=source,group=wire);torch.cuda.synchronize()
                    if rank==source and i>=3:times.append((time.perf_counter()-started)/2)
                if rank==source:records.append({'source':source,'target':target,'bytes':size,'seconds':statistics.median(times),'samples':times})
                del tensor
    gathered=[None]*world if rank==0 else None
    dist.gather_object(records,gathered,dst=0,group=control)
    if rank==0:
        result={'kind':'measured_gpu_nccl_roundtrip_halved','records':[r for group in gathered for r in group],'timing':'same sender perf_counter for round-trip; no subtraction across hosts'}
        Path('original13b/profiles/links-seven.json').write_text(json.dumps(result,indent=2)+'\n');print('LINK_PROFILES_DONE',flush=True)
    torch.cuda.synchronize();dist.barrier(group=control);dist.destroy_process_group(control);dist.destroy_process_group()
if __name__=='__main__':main()

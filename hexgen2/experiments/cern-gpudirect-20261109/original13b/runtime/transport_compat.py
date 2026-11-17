"""PyTorch 2.5 NCCL P2P communicator-cleanup compatibility, not serving policy.

Single blocking send/receive semantics and identical tensors are preserved.
The released pipeline already uses batch_isend_irecv. Using that primitive for
the released KV helper avoids PyTorch interpreting a peer key such as '5:6'
as CUDA device 5 while destroying its individual-peer communicator.
"""
from types import SimpleNamespace
import torch.distributed as dist

def send(tensor,dst,group=None,tag=0):
    requests=dist.batch_isend_irecv([dist.P2POp(dist.isend,tensor,dst,group=group,tag=tag)])
    for request in requests:request.wait()

def recv(tensor,src,group=None,tag=0):
    requests=dist.batch_isend_irecv([dist.P2POp(dist.irecv,tensor,src,group=group,tag=tag)])
    for request in requests:request.wait()
    return src

def install_in_released_kv_helper():
    import kv_cache_communication
    kv_cache_communication.dist=SimpleNamespace(send=send,recv=recv)

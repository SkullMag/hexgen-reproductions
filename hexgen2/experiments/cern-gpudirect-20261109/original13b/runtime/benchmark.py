"""Offline measurement around released full-request generation functions.

Fixed queues/batches match the supplied examples. CUDA events observe native
forward completions without synchronizing or scheduling tokens individually.
"""
import functools,gc,hashlib,json,time
from pathlib import Path
import torch
import torch.distributed as dist
from hexgen_core.generation import decode
from inference_phases import prefill_only,decode_only
from kv_cache_batch import batch_logits_and_key_value_memory_dict
from arguments import clear_kv_cache

class Meter:
    def __init__(self,model,tail,rank):
        self.model=model;self.tail=tail;self.rank=rank;self.events=[]
        self.current=[];self.epoch=None;self.enabled=False
        original=model.gpipe_forward
        @functools.wraps(original)
        def measured(*args,**kwargs):
            result=original(*args,**kwargs)
            if self.enabled and rank==tail:
                event=torch.cuda.Event(enable_timing=True);event.record()
                self.events.append((event,list(self.current),'decode' if args[2][0][1]==1 else 'prefill'))
            return result
        model.gpipe_forward=measured
    def start(self):
        self.events=[];self.epoch=torch.cuda.Event(enable_timing=True)
        self.epoch.record();self.epoch.synchronize();self.utc_ns=time.time_ns();self.enabled=True
    def finish(self,path):
        self.enabled=False;torch.cuda.synchronize()
        if self.rank!=self.tail:return
        with path.open('w') as f:
            for event,ids,phase in self.events:
                f.write(json.dumps({'utc_ns':self.utc_ns+round(self.epoch.elapsed_time(event)*1e6),
                    'ids':ids,'phase':phase,'native_output_tokens':len(ids)})+'\n')

def tensor_input(row):
    return torch.tensor([row['input_ids']],device='cuda',dtype=torch.long)

def batches(rows,size):
    assert len(rows)%size==0
    return [rows[i:i+size] for i in range(0,len(rows),size)]

def run(plan,rep,rank,model,journal,control,output,emit,forward_step,transfer_one):
    rows=json.loads(Path(plan['workload']).read_text())
    warm=json.loads(Path(plan['warmup_workload']).read_text())
    allreps=plan['replicas'];tail=rep['stages'][-1]['ranks'][0]
    meter=Meter(model,tail,rank);results=[]
    batch_size=int(plan.get('batch_size',4))

    def collocated(work,measured):
        replicas=[r for r in allreps if r['role']=='all']
        index=replicas.index(rep);groups=batches(work,batch_size)
        for n,group in enumerate(groups):
            if n%len(replicas)!=index:continue
            lengths={len(r['input_ids']) for r in group};outputs={r['max_new_tokens'] for r in group}
            assert len(lengths)==len(outputs)==1, 'Released generation requires equal sequence lengths within each static batch'
            clear_kv_cache();length=next(iter(lengths));count=next(iter(outputs))
            ids=torch.tensor([r['input_ids'] for r in group],device='cuda',dtype=torch.long)
            shapes=[[-1,length,5120],[-1,length],[-1,length,5120]]
            meter.current=[r['id'] for r in group];started=time.time_ns()
            with torch.inference_mode():
                result=decode(ids,shapes,model,forward_step,length+count,tail,top_k=1,top_p=0,
                              temperature=1,fused_ft_kernel=False,timing=False)
            torch.cuda.synchronize();ended=time.time_ns()
            if measured and rank==tail:
                tokens=result.sequences[:,length:].cpu().tolist()
                assert all(len(t)==count for t in tokens)
                for row,token_ids in zip(group,tokens):
                    record={'id':row['id'],'started_ns':started,'completed_ns':ended,'input_tokens':length,
                            'output_tokens':len(token_ids),'token_ids':token_ids,'replica':rep['id']}
                    results.append(record);emit(journal,'benchmark_request_complete',**record)
            del result,ids;clear_kv_cache()

    def disaggregated(work,measured):
        groups=batches(work,batch_size)
        assert len([r for r in allreps if r['role']=='prefill'])==1
        assert len([r for r in allreps if r['role']=='decode'])==1
        for group in groups:
            assert len({len(r['input_ids']) for r in group})==1 and len({r['max_new_tokens'] for r in group})==1
            started=time.time_ns();pending=[]
            if rep['role']=='prefill':
                for row in group:
                    clear_kv_cache();ids=tensor_input(row);length=ids.shape[1]
                    shapes=[[-1,length,5120],[-1,length],[-1,length,5120]]
                    meter.current=[row['id']]
                    with torch.inference_mode():
                        item=prefill_only(ids,shapes,model,forward_step,length+row['max_new_tokens'],tail,
                                          top_k=1,top_p=0,temperature=1,fused_ft_kernel=False,timing=False)
                    pending.append(item);clear_kv_cache()
            arrivals=[]
            for i,row in enumerate(group):
                item=pending[i] if pending else None
                value=transfer_one(plan,rep,rank,model,row,item,journal,False)
                if rep['role']=='decode':arrivals.append(value)
                if pending:pending[i]=None
                del item,value
            if rep['role']=='decode':
                tokens,caches=batch_logits_and_key_value_memory_dict([v[0] for v in arrivals],[v[1] for v in arrivals])
                del arrivals;clear_kv_cache();row=group[0];length=len(row['input_ids'])
                meter.current=[r['id'] for r in group]
                with torch.inference_mode():
                    result=decode_only(tokens,len(group),length,caches,model,5120,forward_step,
                        length+row['max_new_tokens'],tail,top_k=1,top_p=0,temperature=1,fused_ft_kernel=False,timing=False)
                torch.cuda.synchronize();ended=time.time_ns()
                if measured and rank==tail:
                    token_rows=result.sequences.reshape(len(group),-1).cpu().tolist()
                    for row,token_ids in zip(group,token_rows):
                        assert len(token_ids)==row['max_new_tokens']
                        record={'id':row['id'],'started_ns':started,'completed_ns':ended,
                                'input_tokens':len(row['input_ids']),'output_tokens':len(token_ids),
                                'token_ids':token_ids,'replica':rep['id']}
                        results.append(record);emit(journal,'benchmark_request_complete',**record)
                del result,tokens,caches
            del pending;clear_kv_cache()
    execute=collocated if plan['method']=='hexgen' else disaggregated
    execute(warm,False)
    dist.barrier(group=control);torch.cuda.synchronize()
    if rank==0:start=time.time_ns()
    else:start=None
    anchor=[start];dist.broadcast_object_list(anchor,src=0,group=control)
    started_utc=anchor[0];meter.start();emit(journal,'benchmark_start',started_utc_ns=started_utc,
        requests=len(rows),batch_size=batch_size,method=plan['method'],label=plan.get('label'))
    execute(rows,True)
    meter.enabled=False;dist.barrier(group=control);ended=time.time_ns()
    meter.finish(output/f'native-token-events-rank-{rank}.jsonl')
    gathered=[None]*len(plan['ranks']) if rank==0 else None
    dist.gather_object(results,gathered,dst=0,group=control)
    if rank==0:
        result=[r for group in gathered for r in group]
        assert len(result)==len(rows) and {r['id'] for r in result}=={r['id'] for r in rows}
        seconds=(ended-started_utc)/1e9;tokens=sum(r['output_tokens'] for r in result)
        summary={'label':plan.get('label'),'method':plan['method'],'requests':len(rows),
                 'completed_requests':len(result),'failed_requests':0,'output_tokens':tokens,
                 'started_utc_ns':started_utc,'ended_utc_ns':ended,'wall_seconds':seconds,
                 'wall_output_tokens_per_second':tokens/seconds,'static_decode_batch_size':batch_size,
                 'rows':result,'interpretation':'Fixed offered workload with full drain; original request generation and original static batch policy. Not yet a Figure 6 trace-wide confirmation.'}
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('BENCHMARK_COMPLETED',json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)
    emit(journal,'benchmark_complete')

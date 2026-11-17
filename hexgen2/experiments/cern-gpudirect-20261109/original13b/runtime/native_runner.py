"""FABRIC configuration and measurement harness for released full-request paths.

There is no continuous admission, paged cache, replacement attention, or new
token-by-token serving scheduler. PipelineParallel and generation are released
artifact implementations, with the explicit correctness/import repairs recorded
in provenance/required-repairs.patch.
"""
import argparse, faulthandler, gc, hashlib, importlib, json, os, signal, sys, time
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from transformers import LlamaConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.cwd()))
sys.modules.setdefault('torch.distributed.fsdp.flat_param',importlib.import_module('torch.distributed.fsdp._flat_param'))
# The existing SM75 port changes hardware dispatch only. It adds no cache or
# batching policy. Both arms use this exact same checked FP16 kernel dispatch.
from hexgen2 import fa_adapter
fa_adapter.install()
VENDOR = ROOT/'vendor'
sys.path[:0] = [str(VENDOR/'hexgen'), str(VENDOR/'hexgen/llama'),
               str(VENDOR/'hexgen/llama/modules'),str(VENDOR/'third_party/megatron')]
from hexgen_core.heterogeneous_pipeline import PipelineParallel, PipeSequential
from hexgen_core import CommGroup, generate_send_recv_lists
from hexgen_core.gen_hetero_groups import get_pp_groups,gen_tp_pp_rank_whole_model
from hexgen_core.models import gpt
from hexgen_core.generation import decode
from inference_phases import prefill_only,decode_only
from llama_config_utils import llama_config_to_gpt2_config
from Llamamodel_pipeline import LlamaEmbeddings_,LlamaLayers_,LlamaPreNorm_,LlamaCls_
from arguments import clear_kv_cache
from kv_cache_communication import send_key_value_memory_dict,recv_key_value_memory_dict
from kv_cache_batch import batch_logits_and_key_value_memory_dict
from transport_compat import send as wire_send,recv as wire_recv,install_in_released_kv_helper
install_in_released_kv_helper()

def emit(path,event,**values):
    with path.open('a') as f:
        f.write(json.dumps({'event':event,'utc_ns':time.time_ns(),**values})+'\n')

def forward_step(inputs, inference_params, position_ids, model):
    return model(*inputs,position_ids=position_ids,inference_params=inference_params)

def init_groups(plan, rank):
    # All ranks create every communicator in the same order; artifact constructors
    # otherwise create incompatible overlapping PP groups in different orders.
    original = dist.new_group
    cache = {}; me = None
    groups = [tuple(range(len(plan['ranks'])))]
    for rep in plan['replicas']:
        stages = [s['ranks'] for s in rep['stages']]
        members = sorted(r for stage in stages for r in stage)
        groups.append(tuple(members));groups.extend(tuple(s) for s in stages)
        groups.extend(tuple(path) for path in get_pp_groups(stages))
        if rank in members: me = rep
    for key in dict.fromkeys(groups):
        group = original(list(key),backend='nccl',timeout=timedelta(minutes=8))
        cache[tuple(sorted(key))] = group
        if rank in key:
            z=torch.ones(1,device='cuda');dist.all_reduce(z,group=group)
        dist.barrier()
    assert me is not None
    def cached(ranks=None,*args,**kwargs):
        key=tuple(sorted(ranks if ranks is not None else range(dist.get_world_size())))
        assert key in cache, ('undeclared process group',key)
        return cache[key]
    return me,cache,original,cached

def create_model(plan,replica,rank,cache,original,cached,directory):
    hf=LlamaConfig.from_pretrained(directory,local_files_only=True)
    assert hf.num_hidden_layers==40 and hf.hidden_size==5120 and hf.num_key_value_heads==40
    config=llama_config_to_gpt2_config(hf)
    for key,value in dict(n_head_kv=hf.num_key_value_heads,rotary_emb_base=hf.rope_theta,
                          use_flash_attn=True,fused_bias_fc=True,fused_mlp=False,
                          fused_dropout_add_ln=True,sequence_parallel=False).items():
        setattr(config,key,value)
    stages=replica['stages'];rank_groups=[s['ranks'] for s in stages]
    stage=next(s for s in stages if rank in s['ranks'])
    tp=cache[tuple(sorted(stage['ranks']))]
    tp_size=len(stage['ranks']);tp_rank=stage['ranks'].index(rank)
    # Full released model on meta; only its actual stage is materialized. Native
    # TP block factories and the checkpoint's native TP weight sharder are used.
    full=gpt.GPTLMHeadModel(config,device='meta',dtype=torch.float16)
    for i in range(40):
        full.transformer.layers[i]=gpt.create_block(config,layer_idx=i,
            process_group=tp if tp_size>1 else None,device='meta',dtype=torch.float16)
    seq=PipeSequential();seq.add_module('embeddings',LlamaEmbeddings_(full))
    starts=[s['start'] for s in stages]
    for i in range(40):seq.add_module(f'layer_{i}',LlamaLayers_(full,i,i+1,starts))
    seq.add_module('prenorm',LlamaPreNorm_(full));seq.add_module('cls',LlamaCls_(full))
    partitions=[s['stop']-s['start'] for s in stages]
    mapping,_=gen_tp_pp_rank_whole_model(len(stages),partitions,rank_groups)
    paths=get_pp_groups(rank_groups)
    path=next(path for path in reversed(paths) if rank in path)
    p2p=generate_send_recv_lists(paths,paths[0])
    shapes=[None]+[[[-1,2048,5120],[-1,2048],[-1,2048,5120]]]*40+[None,None]
    dtypes=[None]+[[torch.float16,torch.long,torch.float16]]*40+[None,None]
    dist.new_group=cached
    try:
        model=PipelineParallel(seq,mapping,shapes,dtypes,[1]*43,
            chunks=1,p2p_lists=p2p,process_group=path,broadcast_group=tp,
            broadcast_group_list=stage['ranks'],nproc_per_node=torch.cuda.device_count(),
            show_process=bool(plan.get('debug',False)))
        group=CommGroup(stage['ranks'])
        model.wrap_pipeline_modules_data_parallel([0]*43,[group]*43,
            ['embed']+['gpt_dec']*40+['norm','cls'],mixed_precision=torch.float16)
    finally:dist.new_group=original
    for module in model.model_cur_stage:
        label=module.label()
        if label[0]==0:
            module.embeddings.word_embeddings.load_state_dict(load_file(str(directory/'embedding.safetensors')),strict=True)
        elif label[0]==1:
            i=label[1];sd=load_file(str(directory/f'layer-{i:02}.safetensors'))
            if tp_size>1:
                prefix=f'transformer.layers.{i}.'
                sd=gpt.shard_state_dict_tp({prefix+k:v for k,v in sd.items()},config,tp_size,tp_rank)
                sd={k.removeprefix(prefix):v.contiguous() for k,v in sd.items()}
            layer=module.layers[0];layer.load_state_dict(sd,strict=True)
            rotary=layer.mixer.rotary_emb
            rotary.inv_freq.copy_(1.0/(hf.rope_theta**(torch.arange(0,128,2,device='cuda',dtype=torch.float32)/128)))
        elif label[0]==2:
            module.ln_f.load_state_dict(load_file(str(directory/'norm.safetensors')),strict=True)
        else:module.lm_head.load_state_dict(load_file(str(directory/'head.safetensors')),strict=True)
    model.replica_group=cache[tuple(sorted(r for s in stages for r in s['ranks']))]
    model.replica_members=sorted(r for s in stages for r in s['ranks'])
    model.eval();del full,seq;gc.collect();torch.cuda.empty_cache()
    return model,config

def input_batch(case,batch=1):
    ids=case['input_ids']
    return torch.tensor([ids for _ in range(batch)],dtype=torch.long,device='cuda')

def transfer_routes(source,target):
    """Map existing lossless transfer functions to asymmetric native TP shards."""
    pairs={}
    for s in source['stages']:
        for d in target['stages']:
            lo=max(s['start'],d['start']);hi=min(s['stop'],d['stop'])
            for si,sr in enumerate(s['ranks']):
                for di,dr in enumerate(d['ranks']):
                    hs=max(si*40//len(s['ranks']),di*40//len(d['ranks']))
                    he=min((si+1)*40//len(s['ranks']),(di+1)*40//len(d['ranks']))
                    if hi>lo and he>hs:
                        pairs.setdefault((sr,dr),[]).extend((layer,hs-si*40//len(s['ranks']),he-si*40//len(s['ranks']),hs-di*40//len(d['ranks']),he-di*40//len(d['ranks'])) for layer in range(lo,hi))
    return pairs

def transfer_one(plan,rep,rank,model,case,item,journal,audit):
    source=next(r for r in plan['replicas'] if r['role']=='prefill')
    target=next(r for r in plan['replicas'] if r['role']=='decode')
    pairs=transfer_routes(source,target);length=len(case['input_ids'])
    max_length=length+case['max_new_tokens'];received={};digests=[]
    if rep['role']=='decode':
        stage=next(s for s in rep['stages'] if rank in s['ranks'])
        for layer in range(stage['start'],stage['stop']):
            received[layer]=torch.empty((1,max_length,2,40//len(stage['ranks']),128),device='cuda',dtype=torch.float16)
    for (sr,dr),routes in sorted(pairs.items()):
        if rank==sr:
            payload={layer:item[1][layer][:,:,:,a:b].contiguous() for layer,a,b,c,d in routes}
            if audit:
                for layer,a,b,c,d in routes:
                    raw=payload[layer][:,:length].contiguous().cpu().numpy().tobytes()
                    digests.append({'src':sr,'dst':dr,'layer':layer,'sha256':hashlib.sha256(raw).hexdigest()})
            send_key_value_memory_dict(payload,dr)
            del payload
        elif rank==dr:
            payload=recv_key_value_memory_dict(sr)
            assert len(payload)==len(routes)
            for i,(layer,a,b,c,d) in enumerate(routes):
                received[layer][:,:,:,c:d].copy_(payload[i])
                if audit:
                    raw=received[layer][:,:length,:,c:d].contiguous().cpu().numpy().tobytes()
                    digests.append({'src':sr,'dst':dr,'layer':layer,'sha256':hashlib.sha256(raw).hexdigest()})
            del payload
    stail=source['stages'][-1]['ranks'][0];dtail=target['stages'][-1]['ranks'][0]
    token=item[0] if rank==stail else torch.empty(1,dtype=torch.long,device='cuda')
    if rank==stail:wire_send(token.contiguous(),dst=dtail)
    elif rank==dtail:wire_recv(token,src=stail)
    if rep['role']=='decode':dist.broadcast(token,src=dtail,group=model.replica_group)
    if audit:emit(journal,'cache_transfer_audit',case=case['id'],digests=digests)
    return token,received

def run_disaggregated(plan,rep,rank,model,cases,journal,control):
    tail=rep['stages'][-1]['ranks'][0]
    for case in cases:
        for batch in plan.get('batches',[1,4]):
            emit(journal,'request_start',case=case['id'],batch=batch)
            dist.barrier(group=control);torch.cuda.synchronize();start=time.monotonic_ns()
            pending=[]
            # Released example: queue B single-request prefills, then transfer,
            # concatenate native contiguous caches, decode a fixed batch.
            if rep['role']=='prefill':
                for i in range(batch):
                    clear_kv_cache();ids=input_batch(case)
                    shapes=[[-1,ids.shape[1],5120],[-1,ids.shape[1]],[-1,ids.shape[1],5120]]
                    with torch.inference_mode():
                        item=prefill_only(ids,shapes,model,forward_step,ids.shape[1]+case['max_new_tokens'],tail,
                                          top_k=1,top_p=0,temperature=1,fused_ft_kernel=False,timing=False)
                    pending.append(item);clear_kv_cache()
            arrivals=[]
            for i in range(batch):
                item=pending[i] if pending else None
                value=transfer_one(plan,rep,rank,model,case,item,journal,plan.get('audit',True))
                if rep['role']=='decode':arrivals.append(value)
            if rep['role']=='decode':
                tokens,caches=batch_logits_and_key_value_memory_dict([v[0] for v in arrivals],[v[1] for v in arrivals])
                del arrivals;clear_kv_cache()
                with torch.inference_mode():
                    result=decode_only(tokens,batch,len(case['input_ids']),caches,model,5120,forward_step,
                        len(case['input_ids'])+case['max_new_tokens'],tail,top_k=1,top_p=0,temperature=1,fused_ft_kernel=False,timing=False)
                torch.cuda.synchronize();seconds=(time.monotonic_ns()-start)/1e9
                if rank==tail:
                    output=result.sequences.reshape(batch,-1).cpu().tolist();expected=case.get('token_ids')
                    correct=expected is None or all(v==expected for v in output)
                    emit(journal,'request_complete',case=case['id'],batch=batch,tokens=output,seconds=seconds,correct=correct)
                    assert correct,(case['id'],output,expected)
                del result,tokens,caches
            del pending;clear_kv_cache();gc.collect();torch.cuda.empty_cache()
            dist.barrier(group=control)

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model-dir',default='models/native13b');a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());rank=int(os.environ['RANK'])
    manifest_path=ROOT/'provenance/deployed-source-sha256.json'
    source_manifest=json.loads(manifest_path.read_text())
    manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if plan.get('expected_source_manifest_sha256'):
        assert manifest_sha==plan['expected_source_manifest_sha256']
    for name,expected in source_manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,('source changed',name)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(1)
    faulthandler.register(signal.SIGUSR1,all_threads=True)
    if plan.get('debug'):faulthandler.dump_traceback_later(90,repeat=True)
    dist.init_process_group('nccl',timeout=timedelta(minutes=8),device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    warm=torch.ones(1,device='cuda');dist.all_reduce(warm);dist.barrier()
    control=dist.new_group(list(range(len(plan['ranks']))),backend='gloo',timeout=timedelta(minutes=8))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);journal=out/f'rank-{rank}.jsonl'
    emit(journal,'source_identity',manifest_sha256=manifest_sha,source_files=len(source_manifest),
         method=plan['method'],plan_sha256=hashlib.sha256(Path(a.plan).read_bytes()).hexdigest())
    torch.manual_seed(123);torch.cuda.manual_seed(123)
    rep,cache,original,cached=init_groups(plan,rank)
    emit(journal,'groups_ready',replica=rep)
    model,config=create_model(plan,rep,rank,cache,original,cached,Path(a.model_dir))
    emit(journal,'model_loaded',weight_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
         allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
         native_pipeline_file=importlib.import_module('hexgen_core.heterogeneous_pipeline').__file__,
         native_generation_file=importlib.import_module('hexgen_core.generation').__file__,
         cache_policy='released contiguous cache; fused_ft_kernel=False; no continuous admission')
    tail=rep['stages'][-1]['ranks'][0]
    if plan.get('benchmark'):
        from benchmark import run
        run(plan,rep,rank,model,journal,control,out,emit,forward_step,transfer_one)
        dist.barrier(group=control);emit(journal,'complete');dist.destroy_process_group();return
    cases=json.loads(Path(plan['cases']).read_text())
    if plan['method']=='hexgen2':
        run_disaggregated(plan,rep,rank,model,cases,journal,control)
        dist.barrier(group=control);emit(journal,'complete');dist.destroy_process_group();return
    for case in cases:
        for batch in plan.get('batches',[1]):
            clear_kv_cache();ids=input_batch(case,batch);length=ids.shape[1]
            shapes=[[-1,length,5120],[-1,length],[-1,length,5120]]
            emit(journal,'request_start',case=case['id'],batch=batch)
            dist.barrier(group=control);torch.cuda.synchronize();started=time.monotonic_ns()
            with torch.inference_mode():
                result=decode(ids,shapes,model,forward_step,length+case['max_new_tokens'],tail,
                              top_k=1,top_p=0,temperature=1,fused_ft_kernel=False,timing=False)
            torch.cuda.synchronize();seconds=(time.monotonic_ns()-started)/1e9
            if rank==tail:
                tokens=result.sequences[:,length:].cpu().tolist()
                expected=case.get('token_ids')
                correct=None if expected is None else all(v==expected for v in tokens)
                emit(journal,'request_complete',case=case['id'],batch=batch,tokens=tokens,seconds=seconds,correct=correct,reference_checked=expected is not None)
                assert correct is not False,(case['id'],tokens,expected)
            del result;clear_kv_cache();gc.collect();torch.cuda.empty_cache()
    dist.barrier();emit(journal,'complete')
    dist.destroy_process_group()

if __name__=='__main__':main()

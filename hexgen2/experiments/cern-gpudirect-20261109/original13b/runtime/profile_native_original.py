"""Measured inputs for the released contiguous-cache implementation, not a serving result."""
import argparse,gc,json,os,statistics,time
from pathlib import Path
from datetime import timedelta
from native_runner import gpt,llama_config_to_gpt2_config,LlamaConfig,torch,dist,load_file
from inference_phases import InferenceParams

p=argparse.ArgumentParser();p.add_argument('--tp',type=int,default=1);p.add_argument('--output',required=True);a=p.parse_args()
rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0))
torch.cuda.set_device(local);torch.set_num_threads(1)
group=None
if a.tp>1:
    dist.init_process_group('nccl',timeout=timedelta(minutes=5),device_id=torch.device('cuda',local));group=dist.group.WORLD
directory=Path('models/native13b');hf=LlamaConfig.from_pretrained(directory,local_files_only=True)
cfg=llama_config_to_gpt2_config(hf)
for k,v in dict(n_head_kv=40,rotary_emb_base=hf.rope_theta,use_flash_attn=True,fused_bias_fc=True,
                fused_mlp=False,fused_dropout_add_ln=True,sequence_parallel=False).items():setattr(cfg,k,v)
layer=gpt.create_block(cfg,layer_idx=0,process_group=group,device='cuda',dtype=torch.float16).eval()
sd=load_file(str(directory/'layer-00.safetensors'))
if a.tp>1:
    prefix='transformer.layers.0.'
    sd=gpt.shard_state_dict_tp({prefix+k:v for k,v in sd.items()},cfg,a.tp,rank)
    sd={k.removeprefix(prefix):v.contiguous() for k,v in sd.items()}
layer.load_state_dict(sd,strict=True);del sd
rows=[]
def measure(fn):
    for _ in range(4):fn()
    torch.cuda.synchronize();samples=[]
    for _ in range(11):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();fn();end.record();end.synchronize();samples.append(start.elapsed_time(end)/1000)
    value=statistics.median(samples)
    if group is not None:
        t=torch.tensor([value],device='cuda');dist.all_reduce(t,op=dist.ReduceOp.MAX);value=float(t.item())
    return value
with torch.inference_mode():
    for phase in ['prefill','decode']:
        for batch in [1,4,8]:
            for context in ([512,1024,1536] if phase=='prefill' else [1024,1152,1792]):
                state=InferenceParams(max_sequence_len=2048,max_batch_size=batch,fused_ft_kernel=False)
                state.key_value_memory_dict[0]=layer.allocate_inference_cache(batch,2048,fused_ft_kernel=False)
                state.key_value_memory_dict[0].normal_(std=.1)
                state.sequence_len_offset=0 if phase=='prefill' else context
                x=torch.randn(batch,context if phase=='prefill' else 1,5120,device='cuda',dtype=torch.float16)*.1
                seconds=measure(lambda:layer(x,None,mixer_kwargs={'inference_params':state}))
                rows.append({'phase':phase,'batch':batch,'context':context,'seconds_per_layer':seconds})
                del x,state;gc.collect();torch.cuda.empty_cache()
if rank==0:
    report={'kind':'measured_native_layer_calibration','gpu':torch.cuda.get_device_name(),'tp':a.tp,
            'native_model_source':gpt.__file__,'cache_backend':'released contiguous tensor; fused_ft_kernel=False',
            'checkpoint':'openlm-research/open_llama_13b@b6d7fde8392250730d24cc2fcfa3b7e5f9a03ce8',
            'memory_total_bytes':torch.cuda.get_device_properties(local).total_memory,
            'parameter_bytes_per_rank':sum(p.numel()*p.element_size() for p in layer.parameters()),'rows':rows}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print('NATIVE_CONTIGUOUS_PROFILE_COMPLETE',a.output,flush=True)
if group is not None:dist.destroy_process_group()

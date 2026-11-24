"""Direct FABRIC data-plane load generator with bounded in-flight work.

Offline maintains a fixed backlog for a fixed window; online replays a frozen
arrival trace. All offered requests, drain outcomes and server records are retained.
"""
import argparse,asyncio,json,math,random,time,socket
from .record import digest
from pathlib import Path
import aiohttp


def generate_request(request_format, request_id, input_ids, count):
    if request_format == 'sglang':
        return {
            'rid': request_id,
            'input_ids': input_ids,
            'sampling_params': {
                'temperature': 0,
                'max_new_tokens': count,
                'ignore_eos': True,
            },
        }
    return {'id': request_id, 'input_ids': input_ids, 'max_new_tokens': count}


def validate_completion(request_format, status, data, input_count, output_count):
    if status != 200:
        raise RuntimeError(f'Request returned HTTP {status}')
    if request_format == 'sglang':
        metadata = data['meta_info']
        if metadata.get('finish_reason', {}).get('type') != 'length':
            raise RuntimeError('SGLang generation did not reach the requested length')
        prompt_tokens = metadata['prompt_tokens']
        output_tokens = data['output_ids']
    else:
        if data.get('status') != 'success':
            raise RuntimeError('Server did not complete request')
        prompt_tokens = data['input_tokens']
        output_tokens = data['token_ids']
    if prompt_tokens != input_count or len(output_tokens) != output_count:
        raise RuntimeError('Token count validation failed')


def poisson_arrivals(rate, duration, seed):
    if not math.isfinite(rate) or rate <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Poisson rate and duration must be positive finite numbers')
    rng = random.Random(seed)
    offsets = []
    offset = 0
    while True:
        offset += rng.expovariate(rate)
        if offset >= duration:
            break
        if len(offsets) == 10000:
            raise ValueError('Arrival trace exceeds the 10,000-request replay limit')
        offsets.append(offset)
    if not offsets:
        raise ValueError('Poisson schedule has no arrivals in the requested window')
    return {'rate': rate, 'duration_seconds': duration, 'seed': seed, 'offsets_s': offsets}


async def execute(args):
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(s) for s in args.trace.read_text().splitlines() if s.strip()]
    arrivals_path = args.arrivals
    poisson_rate = getattr(args, 'poisson_rate', None)
    request_format = getattr(args, 'request_format', 'hexgen2')
    if poisson_rate is not None:
        if args.mode != 'online' or args.arrival_seed is None:
            raise ValueError('Generating a Poisson schedule requires online mode and an arrival seed')
        arrivals_path = args.output / 'arrivals.json'
        arrivals_path.write_text(json.dumps(poisson_arrivals(poisson_rate, args.duration, args.arrival_seed), indent=2) + '\n')
    arrivals = json.loads(arrivals_path.read_text()) if arrivals_path else None
    if not rows:raise ValueError('Empty trace')
    routing=json.loads(args.routing.read_text())['endpoint_indices'] if args.routing else list(range(len(args.endpoint)))
    if not routing or any(type(i) is not int or not 0<=i<len(args.endpoint) for i in routing):raise ValueError('Invalid frozen routing')
    results=[];offered=[];origin=time.monotonic();origin_ns=time.time_ns();sequence=0
    semaphore=asyncio.Semaphore(args.inflight)
    timeout=aiohttp.ClientTimeout(total=args.request_timeout)
    connector=aiohttp.TCPConnector(limit=0,force_close=False)
    summary={'method':args.method,'mode':args.mode,'duration_seconds':args.duration,'endpoints':args.endpoint,
             'inflight_limit':args.inflight,'origin_utc_ns':origin_ns,'origin_monotonic':origin,'run_id':args.run_id,
             'trace_sha256':digest(args.trace),'arrivals_sha256':digest(arrivals_path) if arrivals_path else None,
             'request_format':request_format,
             'routing_sha256':digest(args.routing) if args.routing else None,
             'clock':'Client durations use only this process monotonic clock','client_ttft':None,
             'clock_domain':{'hostname':socket.gethostname(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip() if Path('/proc/sys/kernel/random/boot_id').exists() else None},
             'throughput_definition':'Successful completed output tokens received within the fixed measurement window / window seconds; full drain reported separately'}
    (args.output/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    async with aiohttp.ClientSession(timeout=timeout,connector=connector,trust_env=False) as session:
        async def one(index,endpoint,offset=0,phase='measured'):
            row=rows[index%len(rows)];rid=f'{args.run_id}-{phase}-{index:07}'
            scheduled=origin+offset;await asyncio.sleep(max(0,scheduled-time.monotonic()))
            record={'id':rid,'index':index,'prompt_id':row['id'],'phase':phase,'endpoint':endpoint,
                    'input_tokens':len(row['input_ids']),'requested_output_tokens':row['max_new_tokens'],
                    'scheduled_offset_s':offset,'scheduled_utc_ns':origin_ns+round(offset*1e9),'offered_utc_ns':time.time_ns()}
            offered.append(record)
            with (args.output/'offered.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            try:
                async with semaphore:
                    submitted=time.monotonic();record.update(submitted_utc_ns=time.time_ns(),submission_lag_seconds=submitted-scheduled)
                    payload = generate_request(request_format, rid, row['input_ids'], row['max_new_tokens'])
                    async with session.post(endpoint+'/generate',json=payload) as response:
                        data=await response.json();record['http_status']=response.status;record['server']=data
                        validate_completion(request_format, response.status, data, record['input_tokens'], record['requested_output_tokens'])
                        record['success']=True
            except BaseException as error:
                record.update(success=False,error_type=type(error).__name__,error=str(error))
                if isinstance(error,asyncio.CancelledError):record['cancelled']=True
            completed=time.monotonic();record.update(completed_utc_ns=time.time_ns(),completed_offset_s=completed-origin,
                latency_seconds=completed-scheduled,within_measurement_window=completed-origin<=args.duration)
            if record['success']:
                record['submit_to_completion_seconds'] = completed - submitted
            results.append(dict(record))
            with (args.output/'requests.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()
            return record
        if args.mode=='finite':
            await asyncio.gather(*(one(i,args.endpoint[routing[i%len(routing)]],0,phase='qualification') for i in range(len(rows))))
        elif args.mode=='offline':
            # Each endpoint has a dedicated share of the fixed global backlog so a
            # slower replica cannot leave a faster baseline replica starved.
            async def lane(lane_id):
                nonlocal sequence
                endpoint=args.endpoint[lane_id%len(args.endpoint)]
                while time.monotonic()<origin+args.duration:
                    index=sequence;sequence+=1
                    await one(index,endpoint,time.monotonic()-origin)
            tasks=[asyncio.create_task(lane(i)) for i in range(args.inflight)]
            try:await asyncio.wait_for(asyncio.gather(*tasks),args.duration+args.drain_timeout)
            except asyncio.TimeoutError:
                for task in tasks:task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
        else:
            if not arrivals or len(arrivals['offsets_s'])>10000:raise ValueError('Missing or oversized frozen arrivals')
            offsets = [offset for offset in arrivals['offsets_s'] if offset < args.duration]
            tasks=[asyncio.create_task(one(i,args.endpoint[routing[i%len(routing)]],offset)) for i,offset in enumerate(offsets)]
            try:await asyncio.wait_for(asyncio.gather(*tasks),args.duration+args.drain_timeout)
            except asyncio.TimeoutError:
                for task in tasks:task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
        summary.update(end_utc_ns=time.time_ns(),wall_seconds=time.monotonic()-origin,
            offered=len(offered),completed=len(results),successful=sum(r.get('success',False) for r in results),
            failures=sum(not r.get('success',False) for r in results),
            completed_output_tokens=sum(r['requested_output_tokens'] for r in results if r.get('success')),
            measurement_output_tokens=sum(r['requested_output_tokens'] for r in results if r.get('success') and r['within_measurement_window']))
        summary['measurement_window_applicable']=args.mode!='finite'
        summary['throughput_tokens_per_second']=summary['measurement_output_tokens']/args.duration if args.mode!='finite' else None
        summary['throughput_requests_per_second'] = (
            sum(r['success'] and r['within_measurement_window'] for r in results) / args.duration
            if args.mode != 'finite' else None
        )
        latencies = sorted(r['submit_to_completion_seconds'] for r in results if r['success'])
        summary['successful_requests_p99_submit_to_completion_seconds'] = (
            latencies[math.ceil(0.99 * len(latencies)) - 1] if latencies else None
        )
        if args.mode=='finite':
            summary['measurement_output_tokens']=None
            summary['throughput_definition']='Finite diagnostic run; only full-drain throughput applies'
        summary['full_drain_tokens_per_second']=summary['completed_output_tokens']/summary['wall_seconds']
        summary['capture_complete']=len(offered)==len(results)
        if args.mode == 'online':
            summary['scheduled_requests'] = len(offsets)
            summary['capture_complete'] &= len(offered) == len(offsets)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary),flush=True)
        if summary['failures'] or not summary['capture_complete']:raise SystemExit(2)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--mode',choices=['offline','online','finite'],required=True);p.add_argument('--endpoint',action='append',required=True)
    p.add_argument('--routing',type=Path);p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    schedule = p.add_mutually_exclusive_group()
    schedule.add_argument('--arrivals', type=Path)
    schedule.add_argument('--poisson-rate', type=float, help='Generate and save a Poisson schedule at this rate in requests/sec')
    p.add_argument('--arrival-seed', type=int)
    p.add_argument('--request-format', choices=['hexgen2', 'sglang'], default='hexgen2')
    p.add_argument('--duration',type=float,default=600);p.add_argument('--inflight',type=int,default=32)
    p.add_argument('--request-timeout',type=float,default=1800);p.add_argument('--drain-timeout',type=float,default=1200)
    a=p.parse_args();asyncio.run(execute(a))

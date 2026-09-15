import gc,gzip,time,sys
from .common import *
from paper_active_set_v1.measure import g,recovery,probes,anchor_gradients,mixed_forward,flat,project,unflatten
torch=g.torch
def run(mode):
    verify();p=read(OUT/'PROTOCOL_FROZEN.json');SCRATCH.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(1);assert torch.cuda.get_device_name(0)==p['resources']['GPU']
    g.init_ddp(OUT/('ddp_recover_'+mode));recovery.OUT=OUT
    with gzip.open(ROOT/'reports/gradient_relation_4090_v2/DESIGN.json.gz','rt') as f:design=json.load(f)
    pts={x['point']:x for x in design['points']}
    members={x['point']:x for x in read(ROOT/'reports/sensitive_component_compatibility_v1/PROBE_MEMBERSHIP.json')['points']}
    origins={x['point']:x for s in p['seeds'] for x in rows(ROOT/f'reports/functional_geometry_diagnostic_v1/carrier/seed{s}/origins.jsonl')}
    names=p['preflight6'] if mode=='preflight' else p['points'];current=None;algo=None
    try:
        for name in names:
            out=OUT/'recovery'/name;out.mkdir(parents=True,exist_ok=True)
            if (out/'COMPLETE.json').exists():
                for a,x in read(out/'COMPLETE.json')['checkpoints'].items():assert sha(x['path'])==x['sha256']
                continue
            start=time.time();pt=pts[name]
            if current!=(pt['seed'],pt['task']):
                if algo is not None:del algo,ds;gc.collect();torch.cuda.empty_cache()
                algo,ds,plan=recovery.build_without_monitors(pt['seed'],pt['task'],design);current=pt['seed'],pt['task']
            origin=origins[name];assert sha(origin['anchor']['path'])==origin['anchor']['sha256']
            base=torch.load(origin['anchor']['path'],map_location='cpu');g.restore(algo,base)
            assert all(g.digest(g.capture(algo)[k])==h for k,h in origin['witness'].items())
            P=probes(algo,ds,members[name]);Pold={k:v for k,v in P.items() if k[1]<pt['task']}
            expected={(x['arm'],x['horizon']):x['state_hashes'] for x in rows(PRIOR/'measurements'/name/'ENDPOINT_STATES.jsonl')}
            decisions={x['step']:x for x in rows(PRIOR/'measurements'/name/'DECISIONS_FROZEN.jsonl') if x['arm']=='TASK_GRAD_QP'}
            checkpoints={};checks=[]
            for arm in ARMS:
                g.restore(algo,base)
                for h in range(1,21):
                    stage=pt['stage_step']+h-1
                    if h>1:g.ensure_next_epoch_rng(algo,ds,plan,stage-1,stage)
                    b=g.batch(ds,pt['future20_record_indices'][h-1])
                    if arm=='MLR':g.native(algo,b,pt['task'],stage//plan['batches_per_epoch'])
                    else:
                        before=g.capture(algo);H,tasks,_=anchor_gradients(algo,Pold,fullcheck=h==1);fw=mixed_forward(algo,b);ps=g.params(algo);raw=flat(fw['mixed'],ps)
                        g.apply_first(algo,fw['mixed'],fw)
                        c,sol=project(H,-raw,fullcheck=False)
                        old=decisions[h]
                        assert g.digest(H.cpu())==old['gradient_hash'] and sol['lam']==old['solver']['lam'],('QP replay mismatch',name,h)
                        if any(v>0 for v in sol['lam']):g.restore(algo,before);g.apply_first(algo,unflatten(raw+c,ps,fw['mixed']),fw)
                        del before,H,fw,raw,c;gc.collect();torch.cuda.empty_cache()
                    if h in [1,5,20]:
                        state=g.capture(algo);actual={k:g.digest(v) for k,v in state.items()}
                        checks.append({'arm':arm,'horizon':h,'matches':{k:actual[k]==v for k,v in expected[arm,h].items()}})
                        write(out/'PARITY.json',checks)
                        assert all(checks[-1]['matches'].values()),('Full original endpoint mismatch',name,arm,h,checks[-1])
                        if h==20:
                            path=SCRATCH/(name+'__'+arm+'.pth');assert not path.exists()
                            g.atomic_save(path,{'state_dict':state['model'],'full_state_hashes':actual,'point':name,'arm':arm,'horizon':20,'protocol_sha256':sha(OUT/'PROTOCOL_FROZEN.json')})
                            checkpoints[arm]={'path':str(path),'sha256':sha(path),'model_hash':actual['model'],'bytes':path.stat().st_size}
                        del state
                print('RECOVERED',name,arm,flush=True)
            write(out/'COMPLETE.json',{'point':name,'status':'PASS','checkpoints':checkpoints,'parity':checks,'seconds':time.time()-start})
            del base,P,Pold;gc.collect();torch.cuda.empty_cache()
            write(OUT/'RECOVERY_STATUS.json',{'mode':mode,'point':name,'completed':len(list((OUT/'recovery').glob('*/COMPLETE.json'))),'time':time.time()})
        write(OUT/('RECOVERY_PREFLIGHT.json' if mode=='preflight' else 'RECOVERY_COMPLETE.json'),{'status':'PASS','points':len(names),'full_original_witnesses_per_point':6})
    finally:torch.distributed.destroy_process_group()
if __name__=='__main__':run(sys.argv[1])

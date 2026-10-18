"""Frozen post-run accounting and exact comparison; no rollouts or training."""
import os,json,time
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import torch
from coreset_exp.common import O,verify,write,sha,hashes
from coreset_exp.model import cfg,strict_downstream
from libero.lifelong.models.brief import BriefPolicy

def js(p):return json.loads(p.read_text())
def matrix(root,arm):
    rates={};records={}
    for s in range(6,10):
        for t in range(6,s+1):
            rows=[json.loads(x) for x in (root/arm/'evaluation'/f'trials_stage{s}_task{t}.jsonl').read_text().splitlines()]
            assert len(rows)==20 and {r['trial'] for r in rows}==set(range(20))
            assert all(r['seed']==42 and r['stage']==s and r['task']==t and r['arm']==arm and type(r['success']) is bool for r in rows)
            rates[s,t]=F(sum(r['success'] for r in rows),20)
            for r in rows:records[s,t,r['trial']]=r
    return rates,records
def metrics(r):return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
def pack(m):return {k:dict(fraction=str(v),percent=float(100*v)) for k,v in m.items()}
def main():
    verify();P=Path(os.environ['SC_PARENT']);root=O/'sc_coreset';binding=js(O/'PARENT_BINDING.json')
    assert all(sha(p)==h for p,h in binding['protected'].items())
    epochs=js(root/'epochs.json');assert len(epochs)==400 and {(r['task'],r['epoch']) for r in epochs}=={(t,e) for t in range(6,10) for e in range(100)}
    assert sum(r['batches'] for r in epochs)==76100 and epochs[-1]['steps']==76100
    assert js(O/'TASK6_TENSOR_PARITY.json')['equal']
    p=BriefPolicy(cfg('sc_coreset'),None).cpu();loaded=[];selections={};ifa={}
    for t in range(6,10):
        cp=root/'models'/f'mlrifa_{t}_model_ep99.pth';raw=torch.load(cp,map_location='cpu')
        assert raw['epoch_index']==99 and raw['task']==t and raw['contract_sha256']==sha(O/'CONTRACT.json') and raw['modal_sha256']==sha(O/'MODAL_CONTRACT.json')
        loaded.append(dict(task=t,sha256=sha(cp),loaded_keys=strict_downstream(p,cp)))
        m=js(root/'storage'/f'task{t}.json');b=js(O/'bindings'/f'task{t}.json')['windows'];selected=m['selected'];report=m['selection'];n=len(b);k=m['K']
        assert k==[499,445,452,985][t-6] and len(selected)==len({r['destination'] for r in selected})==k
        counts=Counter(str(b[Path(r['source']).name]['group']) for r in selected)
        for g,c in report['population'].items():assert counts[g]==report['target'][g] and abs(counts[g]*n-k*c)<n
        assert len(report['swaps'])==report['minimum_swaps']
        original=set(report['original_ids']);final=set(report['selected_ids']);assert len(original-final)==report['minimum_swaps']
        for r in selected:assert sha(r['source'])==sha(root/'buffer'/r['destination'])==r['sha256']
        selections[str(t)]={k:report[k] for k in ['population','original','target','final','minimum_swaps','retained_original','radius_before','radius_after','mean_distance_before','mean_distance_after','demo_full','demo_original','demo_final','seconds']}
        current=js(root/f'ifa_task{t}.json');parent=js(P/'coreset'/f'ifa_task{t}.json');ifa[str(t)]=dict(current=current,parent=parent,equal=current==parent)
    assert hashes(root/'buffer')==js(root/'buffer_after_task9.json')['hashes']
    r,new=matrix(O,'sc_coreset');c,old=matrix(P,'coreset');random,_=matrix(P,'random')
    for key,row in new.items():
        assert row['initial_state_sha256']==old[key]['initial_state_sha256'] and row['modal_sha256']==old[key]['modal_sha256']
        assert row['checkpoint_sha256']==loaded[row['stage']-6]['sha256']
    events=Counter()
    for f in (root/'evaluation').glob('attempts*.jsonl'):
        for line in f.read_text().splitlines():events[json.loads(line)['event']]+=1
    assert events==Counter(vector_attempt_start=20,vector_attempt_complete=20,environment_construct_start=10,environment_construct_complete=10,environment_closed=10)
    nm,cm,rm=metrics(r),metrics(c),metrics(random);criteria=dict(AUC=nm['AUC']>cm['AUC'],OldFinal=nm['OldFinal']>=cm['OldFinal'])
    write(O/'AUDIT.json',dict(status='PASS',epochs=400,optimizer_steps=76100,formal_records=200,vector_attempts=20,checkpoints=loaded,inputs_unchanged=True,criteria=criteria,metric_gate=all(criteria.values()),metrics=dict(sc_coreset=pack(nm),coreset=pack(cm),random=pack(rm)),delta_vs_coreset=pack({k:nm[k]-cm[k] for k in nm}),matrix=[dict(stage=s,task=t,random=float(random[s,t]*100),coreset=float(c[s,t]*100),sc_coreset=float(r[s,t]*100)) for s,t in r],selection=selections,ifa=ifa,updated=time.time()))
    write(O/'RUN_STATUS.json',dict(status='AWAITING_PROCESS_EXIT_ACCEPTANCE',metric_gate=all(criteria.values()),time=time.time()))
if __name__=='__main__':main()

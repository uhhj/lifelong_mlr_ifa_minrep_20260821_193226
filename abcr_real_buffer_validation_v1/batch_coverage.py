"""Native sampler-only reconstruction. No feature forward, optimizer or ABCR training."""
import argparse,csv,math,random
from pathlib import Path
import numpy as np
import torch
from .common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    write(a.out/'NATIVE_BATCH_COVERAGE_PROTOCOL.json',dict(epochs=list(range(100)),dataset='sorted old + sorted current, random.Random(seed).shuffle, exact native build_dataset_for_task',sampler='DistributedSampler shuffle=True default seed0, world_size1/rank0, set_epoch(epoch); batch_size10/drop_lastFalse',validation='historical lr_protocol.csv batch counts must equal ceil(dataset_size/10)',scope='R1 actual training access rate only; no P4 ABCR training',formal_training_resumed=False),True)
    allrows=[];summary=[]
    for s in read(a.out/'STAGE_MANIFEST.json')['stages']:
        seed,task=s['seed'],s['task'];folder=a.out/('seed%d_task%d'%(seed,task));order=read(folder/'record_order.json');z=np.load(folder/'candidate_pairs.npz')
        nm={r['record']:i for i,r in enumerate(order['new'])};om={'t%d__%s'%(r['task'],r['record']):i for i,r in enumerate(order['old'])}
        current=sorted(p.name for p in (a.repo/'features'/('object_seed%d'%seed)/str(task)).glob('*.pt'));old=sorted(s['old_files']);mix=[('o',om.get(n,-1)) for n in old]+[('n',nm.get(n,-1)) for n in current];random.Random(seed).shuffle(mix)
        cfg=read(a.repo/'runs/own_reference_control/ifa'/('seed%d'%seed)/'run_001/config.json');assert cfg['train']['batch_size']==10
        lr=list(csv.DictReader((a.repo/'runs/own_reference_control/ifa'/('seed%d'%seed)/'run_001/lr_protocol.csv').open()))
        records=[r for r in lr if int(r['stage'])==task];assert records and all(int(r['batches'])==math.ceil(len(mix)/10) for r in records),('NATIVE_BATCH_COUNT_MISMATCH',seed,task)
        for gate,mask in [('quantile',np.ones(len(z['share']),bool)),('physical',z['physical_pass']),('physical_object_proxy',z['object_proxy_pass'])]:
            share=np.zeros((len(nm),len(om)),bool);bif=share.copy()
            for label,matrix in [('share',share),('bif',bif)]:
                ok=mask&z[label];matrix[z['new_local'][ok],z['old_local'][ok]]=True
            for epoch in range(100):
                sampler=torch.utils.data.DistributedSampler(mix,num_replicas=1,rank=0,shuffle=True);sampler.set_epoch(epoch);indices=list(sampler);counts=[0,0,0,0,0]
                for start in range(0,len(indices),10):
                    rows=[mix[k] for k in indices[start:start+10]];ni=[k for role,k in rows if role=='n' and k>=0];oi=[k for role,k in rows if role=='o' and k>=0]
                    sh=bool(ni and oi and share[np.ix_(ni,oi)].any());bi=bool(ni and oi and bif[np.ix_(ni,oi)].any())
                    counts[0]+=1;counts[1]+=sh;counts[2]+=bi;counts[3]+=sh and bi;counts[4]+=bool(ni and oi)
                allrows.append(dict(seed=seed,task=task,gate=gate,epoch=epoch,batches=counts[0],share_batches=counts[1],bif_batches=counts[2],both_batches=counts[3],mixed_batches=counts[4]))
            rows=allrows[-100:];total=sum(r['batches'] for r in rows);summary.append(dict(seed=seed,task=task,gate=gate,total_batches=total,share_batch_fraction=sum(r['share_batches'] for r in rows)/total,bif_batch_fraction=sum(r['bif_batches'] for r in rows)/total,both_batch_fraction=sum(r['both_batches'] for r in rows)/total,buffer_anchor_both_fraction=float((share.any(1)&bif.any(1)).sum()/len(current)),historical_batch_count_verified=True))
        print('NATIVE_SAMPLER',seed,task,flush=True)
    table(a.out/'R1_NATIVE_BATCH_EPOCHS.csv',allrows);table(a.out/'R1_NATIVE_BATCH_COVERAGE.csv',summary)

if __name__=='__main__':main()

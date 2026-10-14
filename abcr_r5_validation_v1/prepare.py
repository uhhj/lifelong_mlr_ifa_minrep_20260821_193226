import argparse,collections,platform,subprocess,time
from pathlib import Path
import numpy as np
import torch
from .common import *

def load_raw(repo):
    pilot=repo/'reports/abcr_object_pilots_v1';dm=read(pilot/'DATA_MANIFEST.json');raw={};meta={};visual={};keys=collections.defaultdict(list)
    for r in dm['demos']:
        if r['task']<6:continue
        p=pilot/(r['id']+'_raw.npz');assert sha(p)==r['raw_sha256']
        z=dict(np.load(p));raw[r['task'],r['demo']]=z;meta[r['task'],r['demo']]=r
        fp=pilot/(r['id']+'_features.npz');assert sha(fp)==read(pilot/(r['id']+'_FEATURE_COMPLETE.json'))['sha256']
        visual[r['task'],r['demo']]=np.load(fp)['visual']
        for w in range(r['n']):
            a=np.ascontiguousarray(z['actions'][np.minimum(np.arange(w,w+8),r['n']-1)],dtype=np.float32)
            keys[r['task'],a.tobytes()].append(dict(task=r['task'],demo=r['demo'],window=w,t=min(w+7,r['n']-1)))
    return raw,meta,visual,keys

def pair_metrics(n,o,raw,visual,norm,th):
    a=raw[n['task'],n['demo']];b=raw[o['task'],o['demo']];i,j=n['t'],o['t']
    e=float(np.linalg.norm(a['eef'][i]-b['eef'][j]));qa=a['quat'][i]/np.linalg.norm(a['quat'][i]);qb=b['quat'][j]/np.linalg.norm(b['quat'][j]);rot=float(np.degrees(2*np.arccos(min(1,abs(qa@qb)))))
    gi=(a['state'][i,7]-a['state'][i,8])/.08;gj=(b['state'][j,7]-b['state'][j,8])/.08
    grip=float(abs(gi-gj));ds=float(np.linalg.norm((a['state'][i]-b['state'][j])/norm['state_std']))
    va=visual[n['task'],n['demo']][i].astype(float);vb=visual[o['task'],o['demo']][j].astype(float)
    vs=float(np.mean(np.sum(va*vb,axis=-1)/(np.linalg.norm(va,axis=-1)*np.linalg.norm(vb,axis=-1))))
    valid=i+4<len(a['actions']) and j+4<len(b['actions'])
    da=float(np.sqrt(np.mean(((a['actions'][i+1:i+5].astype(float)-b['actions'][j+1:j+5].astype(float))/norm['action_scale'])**2))) if valid else None
    cfg=next(r for r in th if r['task']==n['task'] and r['beta']==.5)
    phys=bool(0<=gi<=1 and 0<=gj<=1 and e<=.03 and rot<=15 and grip<=.1)
    eligible=bool(valid and phys and vs>=cfg['tau_v'] and ds<=cfg['tau_s'])
    label=('SHARE' if da<=cfg['tau_plus'] else ('BIF' if da>=cfg['tau_minus'] else 'MID')) if valid else 'TAIL'
    return dict(action_distance=da,state_distance=ds,eef_m=e,rotation_deg=rot,gripper_difference=grip,object_relative_m=float(np.linalg.norm(a['target_relative'][i]-b['target_relative'][j])),visual_score=vs,physical_pass=phys,eligible=eligible,label=label)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();repo,out=a.repo,a.out;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);write(out/'PROTOCOL.json',PROTOCOL,True)
    starts=read(repo/'reports/actual_update_fidelity_v1/START_MANIFEST.json')['starts'];chosen=[]
    for seed in [45,46,47]:
        for task in [7,8]:
            rows=sorted([s for s in starts if s['seed']==seed and s['task_id']==task and s['anchor']['stage_step']==s['stage_step']],key=lambda s:s['stage_step'])
            assert len(rows)>=2;chosen.extend([rows[0],rows[-1]])
    prev=repo/'reports/abcr_real_buffer_validation_v1';pairs=read(prev/'AUDIT_SELECTION_PRIVATE.json')['pairs'];raw,meta,visual,keys=load_raw(repo)
    norm=read(repo/'reports/abcr_object_pilots_v1/NORMALIZATION.json');th=read(repo/'reports/abcr_object_pilots_v1/FROZEN_STAGE_THRESHOLDS.json')
    # New blind identifiers without previous labels. Mapping is inaccessible to the review page.
    ordered=sorted(pairs,key=lambda r:stable(['r5-blind',r['pair_id']]))
    for i,r in enumerate(ordered):r['audit_id']='A%03d'%(i+1)
    write(out/'AUDIT_SELECTION_PRIVATE.json',dict(pairs=ordered,source_sha=sha(prev/'AUDIT_SELECTION_PRIVATE.json')),True)
    mapped={};plan_cache={};record_audit=[];man=[]
    for s in chosen:
        seed,task=s['seed'],s['task_id'];key=(seed,task);pp=repo/s['plan']['path'];assert sha(pp)==s['plan']['sha256'];plan=read(pp)
        if key not in plan_cache:
            lookup=collections.defaultdict(list);records=[]
            for idx,(p,h) in enumerate(zip(plan['immutable_feature_origins'],plan['record_sha256'])):
                if p not in mapped:
                    z=torch.load(repo/p,map_location='cpu');t=int(Path(p).parent.name);hits=keys[t,np.ascontiguousarray(z['actions'].numpy().reshape(8,7),dtype=np.float32).tobytes()]
                    mapped[p]=dict(hits[0],origin=p,sha256=h) if len(hits)==1 else None
                    record_audit.append(dict(seed=seed,origin=p,identity_matches=len(hits)))
                r=mapped[p];records.append(r)
                if r:lookup[r['task'],r['demo'],r['window']].append(dict(r,index=idx))
            plan_cache[key]=(lookup,records)
        lookup,records=plan_cache[key];anchor=Path(s['anchor']['anchor']['path']);assert sha(anchor)==s['anchor']['anchor']['sha256']
        selected=[];missing=[]
        for p in ordered:
            if p['task']!=task:continue
            n=lookup[p['task'],p['new_demo'],p['new_window']];o=lookup[p['old_task'],p['old_demo'],p['old_window']]
            if len(n)==1 and len(o)==1:
                selected.append(dict(pair_id=p['pair_id'],audit_id=p['audit_id'],new=n[0],old=o[0],**pair_metrics(n[0],o[0],raw,visual,norm,th)))
            else:missing.append(dict(pair_id=p['pair_id'],new_matches=len(n),old_matches=len(o)))
        batch=[records[i] for i in s['record_indices']];assert len(batch)==10
        batch_unmapped=[i for i,r in enumerate(batch) if r is None]
        bp=[]
        for i,n in enumerate(batch):
            if n is None or n['task']!=task:continue
            for j,o in enumerate(batch):
                if o is None or o['task']>=task:continue
                bp.append(dict(new_slot=i,old_slot=j,**pair_metrics(n,o,raw,visual,norm,th)))
        probes={}
        for kind,tasks in [('old',list(range(6,task))),('new',[task])]:
            per_task=12//len(tasks);sel=[]
            for t in tasks:
                banned={r['demo'] for r in batch if r and r['task']==t};available=[r for r in records if r and r['task']==t and r['demo'] not in banned]
                groups=collections.defaultdict(list)
                for r in available:groups[r['demo']].append(r)
                for d in groups:groups[d]=sorted(groups[d],key=lambda r:stable(['probe',r['task'],r['demo'],r['window']]))
                took=[]
                for k in range(per_task):
                    nonempty=[d for d in sorted(groups) if groups[d]]
                    if not nonempty:break
                    d=nonempty[k%len(nonempty)];took.append(groups[d].pop(0))
                sel.extend(took)
            probes[kind]=sel
        for r in [q for p in selected for q in [p['new'],p['old']]]+[r for r in batch if r]+probes['old']+probes['new']:
            assert sha(repo/r['origin'])==r['sha256']
        man.append(dict(start=s,pairs=selected,excluded_pairs=missing,batch_records=batch,batch_unmapped_slots=batch_unmapped,batch_pairs=bp,probes=probes))
        print('PREPARED',s['start_id'],'B',len(selected),'Celigible',sum(p['eligible'] and p['label']!='MID' for p in bp),'probe',[len(v) for v in probes.values()],flush=True)
    write(out/'START_MANIFEST.json',dict(starts=man,task9='NOT_RUN_NO_COMPLETE_ADAMW_STATE'),True)
    table(out/'IDENTITY_AUDIT.csv',record_audit)
    context=dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(repo),text=True).strip(),git_branch=subprocess.check_output(['git','branch','--show-current'],cwd=str(repo),text=True).strip(),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda_initialized=torch.cuda.is_initialized(),device='CPU',formal_training_resumed=False,source_actual_manifest_sha=sha(repo/'reports/actual_update_fidelity_v1/START_MANIFEST.json'))
    write(out/'EXPERIMENT_CONTEXT.json',context,True)
    write(out/'PREPARATION_COMPLETE.json',dict(manifest_sha=sha(out/'START_MANIFEST.json'),protocol_sha=sha(out/'PROTOCOL.json'),code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},anchors=len(man),audit_pairs=len(ordered)),True)

if __name__=='__main__':main()

"""Paired transactional BC/ABCR probes; exactly restored original optimizer state."""
import argparse,contextlib,gc,io,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from abcr_real_buffer_validation_v1 import gradient_probe as native
from .common import *

def load(repo,r):
    assert sha(repo/r['origin'])==r['sha256'];return torch.load(repo/r['origin'],map_location='cpu')
def stack(z):return {k:torch.stack([r[k] for r in z]) for k in z[0]}
def aux(last,pairs,arm):
    z=F.normalize(last.float(),dim=-1);parts={};counts={}
    for label in ['SHARE','BIF']:
        selected=[p for p in pairs if p['eligible'] and p['label']==label];values=[]
        for p in selected:
            sim=(z[p['new_slot']]*z[p['old_slot']].detach()).sum()
            values.append(1-sim if label=='SHARE' else torch.relu(sim-PROTOCOL['C']['bif_cosine_margin']))
        parts[label]=torch.stack(values).mean() if values else last.sum()*0;counts[label]=len(values)
    added=last.sum()*0
    if arm in ['SHARE_ONLY','ABCR']:added=added+.05*parts['SHARE']
    if arm in ['BIF_ONLY','ABCR']:added=added+.05*parts['BIF']
    return added,{k:float(v.detach()) for k,v in parts.items()},counts
def displacement(a,s,go=None):
    norm=0.;hd=0.
    for idx,(name,p) in enumerate(a.policy.named_parameters()):
        d=p.detach().double()-s['model'][name].double();norm+=float(d.square().sum())
        if go is not None and go[idx] is not None:hd+=float((go[idx].double()*d).sum())
    return norm**.5,hd
def eval_losses(a,z):
    native.eval_mode(a)
    with torch.no_grad():return [float(native.loss(a,v)) for v in z]
def counter(a,s):
    c=dict(optimizer_delta=a._optimizer_step_count-s['optimizer_step'],scheduler_delta=a._scheduler_step_count-s['scheduler_step']);assert c==dict(optimizer_delta=1,scheduler_delta=1);return c
def restore_checked(a,s):
    native.restore(a,s);c=native.checks(a,s);assert all(c.values()),c;return c
def evidence(a):return {k:native.digest(v) for k,v in [('model',a.policy.state_dict()),('optimizer',a.optimizer.state_dict()),('scheduler',a.scheduler.state_dict())]}

def run_b(a,s,entry,repo,out,binding):
    start=entry['start'];rows=[];dest=out/'B'/start['start_id'];dest.mkdir(parents=True,exist_ok=True)
    for r in entry['pairs']:
        path=dest/(r['pair_id']+'.json')
        if path.exists():
            prev=read(path);assert prev['binding']==binding;rows.append(prev['row']);continue
        begin=time.monotonic();zn,zo=load(repo,r['new']),load(repo,r['old']);restore_checked(a,s);native.eval_mode(a)
        old0,go=native.grad(a,zo);new0,gn=native.grad(a,zn);cos=native.dot(go,gn)/np.sqrt(native.dot(go,go)*native.dot(gn,gn))
        a.optimizer.zero_grad(set_to_none=True)
        for p,g in zip(a.policy.parameters(),gn):p.grad=None if g is None else g.clone()
        pre=float(torch.nn.utils.clip_grad_norm_(a.policy.parameters(),a.cfg.train.grad_clip));a.optimizer_step_and_maybe_schedule();counts=counter(a,s);dn,hd=displacement(a,s,go)
        old1,new1=eval_losses(a,[zo,zn]);restored=restore_checked(a,s)
        row=dict(start_id=start['start_id'],seed=start['seed'],task=start['task_id'],epoch=start['epoch'],pair_id=r['pair_id'],audit_id=r['audit_id'],new_trajectory=qualified_trajectory(r['new']['task'],r['new']['demo']),old_trajectory=qualified_trajectory(r['old']['task'],r['old']['demo']),new_window=r['new']['window'],old_window=r['old']['window'],label=r['label'],physical_pass=r['physical_pass'],eligible=r['eligible'],action_distance=r['action_distance'],state_distance=r['state_distance'],eef_m=r['eef_m'],rotation_deg=r['rotation_deg'],gripper_difference=r['gripper_difference'],object_relative_m=r['object_relative_m'],negative_gradient_cosine=-float(cos),h_dot_d=hd,old_base=old0,new_base=new0,old_delta=old1-old0,new_delta=new1-new0,deadband=deadband(old0),d_norm=dn,preclip_norm=pre,seconds=time.monotonic()-begin,**counts)
        assert all(np.isfinite(v) for v in row.values() if isinstance(v,(float,np.floating)));write(path,dict(binding=binding,row=row,restored=restored));rows.append(row)
        del go,gn,zn,zo;gc.collect()
    return rows

def run_c(a,s,entry,repo,out,binding):
    start=entry['start'];dest=out/'C'/start['start_id'];dest.mkdir(parents=True,exist_ok=True)
    if entry['batch_unmapped_slots']:
        write(dest/'NOT_RUN.json',dict(reason='AMBIGUOUS_BATCH_IDENTITIES',slots=entry['batch_unmapped_slots']));return []
    data=stack([load(repo,r) for r in entry['batch_records']]);zs={k:[load(repo,r) for r in entry['probes'][k]] for k in ['old','new']}
    restore_checked(a,s);base={k:eval_losses(a,zz) for k,zz in zs.items()};restore_checked(a,s);results=[];original=None
    for arm in PROTOCOL['C']['arms']:
        path=dest/(arm+'.json')
        if path.exists():
            p=read(path);assert p['binding']==binding;results.append(p)
            if arm=='MLR':original=p['candidate_evidence']
            continue
        begin=time.monotonic();restore_checked(a,s);a.optimizer.zero_grad(set_to_none=True)
        bc,last=a.policy.compute_loss(data,None,None);added,parts,counts=aux(last,entry['batch_pairs'],arm)
        loss=a.loss_scale*bc+added;loss.backward();pre=float(torch.nn.utils.clip_grad_norm_(a.policy.parameters(),a.cfg.train.grad_clip));a.optimizer_step_and_maybe_schedule();deltas=counter(a,s)
        ev=evidence(a);dn,_=displacement(a,s);rng_candidate=native.digest(torch.get_rng_state());values={k:eval_losses(a,zz) for k,zz in zs.items()}
        if arm=='MLR':
            original=ev;restore_checked(a,s)
            with contextlib.redirect_stdout(io.StringIO()):a.observe([data,[str(r['task']) for r in entry['batch_records']]],start['task_id'],start['epoch'],None)
            parity={k:ev[k]==evidence(a)[k] for k in ev};parity['rng']=rng_candidate==native.digest(torch.get_rng_state());assert all(parity.values()),parity
        else:parity=None
        no_aux=(arm=='SHARE_ONLY' and counts['SHARE']==0) or (arm=='BIF_ONLY' and counts['BIF']==0) or (arm=='ABCR' and sum(counts.values())==0)
        if no_aux:assert ev==original,'NO_PAIR_NATIVE_PARITY_FAILED'
        restored=restore_checked(a,s);observations=[]
        for kind in ['old','new']:
            for r,b,v in zip(entry['probes'][kind],base[kind],values[kind]):observations.append(dict(kind=kind,task=r['task'],trajectory=qualified_trajectory(r['task'],r['demo']),window=r['window'],origin=r['origin'],base=b,updated=v,delta=v-b,deadband=deadband(b)))
        row=dict(binding=binding,start_id=start['start_id'],seed=start['seed'],task=start['task_id'],epoch=start['epoch'],arm=arm,bc=float(bc.detach()),auxiliary=float(added.detach()),auxiliary_components=parts,pair_counts=counts,preclip_norm=pre,d_norm=dn,seconds=time.monotonic()-begin,counts=deltas,observations=observations,candidate_evidence=ev,native_parity=parity,no_active_pair=no_aux,bitwise_equal_native=ev==original,restored=restored,rng_candidate=rng_candidate)
        write(path,row);results.append(row);print('C',start['start_id'],arm,counts,'native_equal',ev==original,flush=True)
        del bc,last,loss,added;gc.collect()
    return results

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();repo,out=a.repo,a.out
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);assert not torch.cuda.is_initialized()
    files=[Path(__file__),Path(__file__).with_name('common.py'),Path(native.__file__),out/'START_MANIFEST.json']
    native_files=[repo/p for p in ['libero/lifelong/models/modules/transformer_modules.py','libero/lifelong/models/policy_head.py','libero/lifelong/models/brief.py','libero/lifelong/models/base_policy.py','libero/lifelong/algos/base.py','libero/lifelong/utils.py']]
    binding=bind(out,files+native_files);write(out/'ENGINE_BINDING.json',binding,True);B=[];C=[]
    for entry in read(out/'START_MANIFEST.json')['starts']:
        start=entry['start'];p=Path(start['anchor']['anchor']['path']);assert sha(p)==start['anchor']['anchor']['sha256'];s=torch.load(p,map_location='cpu');algo=native.build(repo,s);restore_checked(algo,s)
        assert s['config'].get('ifa_enabled',False)==False,'ANCHOR_IS_NOT_PURE_MLR'
        B.extend(run_b(algo,s,entry,repo,out,binding));C.extend(run_c(algo,s,entry,repo,out,binding))
        assert sha(p)==start['anchor']['anchor']['sha256'];print('DONE',start['start_id'],'B rows',len(B),'C',len(C),flush=True)
        del algo,s;gc.collect()
    table(out/'R5_B_RAW.csv',B);write(out/'R5_C_RAW.json',C);write(out/'MEASUREMENT_COMPLETE.json',dict(status='COMPLETE',B_rows=len(B),C_branches=len(C),cuda_initialized=torch.cuda.is_initialized(),formal_training_resumed=False,binding=binding))

if __name__=='__main__':main()

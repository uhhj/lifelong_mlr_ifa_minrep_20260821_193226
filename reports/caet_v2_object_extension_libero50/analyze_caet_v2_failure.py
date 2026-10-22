"""Local metadata-only analysis of completed CAET-v2; no model or remote access."""
from pathlib import Path
import json, hashlib, datetime
from fractions import Fraction as F

R = Path(__file__).resolve().parent
V = R.parent / '20261021_caet_v2_object_seed42'
PACKS = {41:R/'final_seed41_20261022_022751',42:V/'final_20261021_231144',43:R/'final_seed43_20261022_052808'}
OUT = R/'caet_v2_failure_analysis_20261022'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p): return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    inputs = {}
    ap = R/'caet_v2_object_three_seed_final_20261022/AGGREGATE_AUDIT.json'
    agg = js(ap); inputs[str(ap.relative_to(R))] = sha(ap)
    assert agg['status']=='THREE_SEED_LOCAL_RECOMPUTATION_PASS'
    stages = []; task_deltas = {}
    for seed,p in PACKS.items():
        manifest = js(p/'PACKAGE_MANIFEST.json')['entries']
        for filename,key in [('PACKAGE_MANIFEST.json','manifest_sha256'),('LOCAL_FINAL_AUDIT.json','audit_sha256'),('TASK_COST_AUDIT.json','cost_sha256')]:
            assert sha(p/filename)==agg['inputs'][str(seed)][key]
            inputs[f'seed{seed}/{filename}']=sha(p/filename)
        audit=js(p/'LOCAL_FINAL_AUDIT.json')
        def read(rel):
            f=p/rel; item=manifest[rel]
            assert f.stat().st_size==item['bytes'] and sha(f)==item['sha256'],rel
            inputs[f'seed{seed}/{rel}']=sha(f)
            return js(f)
        contract=read('current/CONTRACT.json')
        for filename in ['observe.py','fit.py','model.py','functional_reference.py','science.py','dataset.py','train.py']:
            local=V/'additions/caet_v2_exp'/filename
            matches=[h for name,h in contract['source_sha256'].items() if name=='caet_v2_exp/'+filename]
            # Deployment serializes text with LF; preserve local CRLF files.
            lf_sha=hashlib.sha256(local.read_text(encoding='utf8').encode('utf8')).hexdigest()
            assert len(matches)==1 and lf_sha==matches[0], filename
            inputs['frozen_source/'+filename]=sha(local)
            inputs['frozen_source_LF/'+filename]=lf_sha
        matrices={m:{(x['stage'],x['task']):F(x['rate']) for x in agg['matrices'][str(seed)][m]} for m in ['v2','ra']}
        assert all(matrices['v2'][k]==matrices['ra'][k] for k in matrices['ra'] if k[0]<=7)
        task_deltas[str(seed)]=[]
        for t in range(6,10):
            d={s:matrices['v2'][s,t]-matrices['ra'][s,t] for s in range(t,10)}
            task_deltas[str(seed)].append(dict(task=t,diagonal_delta_pp=float(d[t]*100),final_delta_pp=float(d[9]*100),auc_contribution_pp=float(sum(d.values())/len(d)/4*100)))
        for t in [8,9]:
            accept=read(f'current/effects/models/task{t}/ACCEPTANCE.json')
            graph=read(f'current/effects/current/task{t}/SUPPORT.json')
            counts=dict(applied=0,blocked=0,retention=0,steps=0)
            max_mu=0
            for ep in range(100):
                rows=read(f'current/ra_coreset/auxiliary/task{t}_epoch{ep:02d}.json')
                counts['steps']+=len(rows)
                counts['applied']+=sum(bool(x['transfer_applied']) for x in rows)
                counts['blocked']+=sum(x['transfer_dot'] is not None and x['transfer_dot']<0 for x in rows)
                counts['retention']+=sum(x['retention_loss']>0 for x in rows)
                c=read(f'current/ra_coreset/constraints/task{t}_epoch{ep:02d}.json')
                max_mu=max(max_mu,*c['mu_after'].values())
            expected=audit['activation'][str(t)]
            assert counts['applied']==expected['transfer'] and counts['blocked']==expected['blocked'] and counts['retention']==expected['retention']
            violated=sum(v>c['bounds'][k]+c['epsilon'] for k,v in c['values'].items())
            assert violated==agg['costs'][str(seed)]['constraints'][str(t)]['final_violated_groups']
            stages.append(dict(seed=seed,task=t,**counts,blocked_fraction=counts['blocked']/(counts['applied']+counts['blocked']),
                supported_current=graph['supported_current'],eligible_unique_current=len(graph['current_nodes']),
                current_support_fraction=graph['supported_current']/len(graph['current_nodes']),
                current_demos=len(graph['supporting_current_demos']),heldout_supported=graph['heldout_supported'],
                predictor_status=accept['status'],heldout_mae=accept['heldout_mae'],permuted_mae=accept['permuted_action_mae'],
                relative_gain_vs_permutation=1-accept['heldout_mae']/accept['permuted_action_mae'],
                relative_gain_vs_mean=1-accept['heldout_mae']/accept['training_mean_baseline_mae'],
                final_violated_groups=violated,groups=len(c['bounds']),max_mu=max_mu))
    totals={k:sum(x[k] for x in stages) for k in ['applied','blocked','retention','steps']}
    totals['blocked_fraction']=totals['blocked']/(totals['applied']+totals['blocked'])
    result=dict(status='LOCAL_EXISTING_METADATA_ANALYSIS_PASS',observed_at=datetime.datetime.now().astimezone().isoformat(),
                new_training=0,new_environment_steps=0,new_model_forwards=0,server_writes=0,
                means=agg['means'],paired_deltas=agg['paired_deltas'],task_deltas=task_deltas,stages=stages,totals=totals,
                inference_limit='Observed associations and source-level structural findings; no causal ablation or significance claim.')
    OUT.mkdir(exist_ok=True)
    (OUT/'EVIDENCE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (OUT/'INPUT_HASHES.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(stages=stages,totals=totals,task_deltas=task_deltas,verified_inputs=len(inputs)),indent=2))

if __name__=='__main__': main()

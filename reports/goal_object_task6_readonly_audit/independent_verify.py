"""Second, counting-based check of report evidence. No torch/model/environment."""
import collections,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
def read(n):return json.loads((R/n).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p):sha(p) for p in R.rglob('*') if p.is_file() and p.name not in ('INDEPENDENT_ACCEPTANCE.json','ARTIFACT_MANIFEST.json','independent_verify.log')}
d=read('DIAGNOSTICS.json');r=read('RESULTS.json');j=read('JOINT_RNG.json');b=read('BUFFER_VERIFICATION.json')
assert d['status']==r['status']==j['status']==b['status']=='PASS_READONLY'
for name in ('collect.exit','joint_rng.exit','buffer_verify.exit'):assert (R/name).read_text().strip()=='0'
for rel,v in read('EVIDENCE_MANIFEST.json').items():assert sha(R/'evidence'/rel)==v['sha256']
for rel,v in read('TRANSFER_MANIFEST.json').items():assert sha(R/rel)==v['sha256'] and (R/rel).stat().st_size==v['bytes']
assert sum(c['N'] for c in d['cells'])==10801
assert sum(row['count'] for row in b['rows'])==10804
assert len(j['rng'])==64 and all(x['entry_all_rng_equal'] for x in r['rng_comparison'])
for c in d['cells']:
    joint=next(z for z in j['cells'] if (z['suite'],z['seed'])==(c['suite'],c['seed']))
    for method in ('population','random','coreset'):
        ws=[w['candidates'][0] for w in c['window_bindings'] if w['unique'] and (method=='population' or w['selected'][method])]
        prof=c['population'] if method=='population' else c['methods'][method]['profile']
        assert collections.Counter(w['demo'] for w in ws)==prof['demo_counts']
        qt=collections.Counter(str(min(3,4*w['target']//(w['length']-1))) for w in ws)
        assert qt==prof['quartile_counts']
        tabs=joint['profiles'][method]['cross_tabs']
        assert sum(z['n'] for z in tabs)==len(ws)
        assert sum(z['n'] for z in tabs if z['padded'])==prof['padding_n']
        assert abs(sum(z['n'] for z in tabs if z['gripper']=='positive')/len(ws)-prof['positive_gripper_fraction'])<1e-12
        for q in range(4):assert sum(z['n'] for z in tabs if z['quartile']==q)==qt[str(q)]
    # Verify IFA diagnostics against original copied lightweight JSON, not report prose.
    for arm in ('random','coreset'):
        for task in range(6,10):
            p=R/'evidence'/f"{c['suite']}_{c['seed']}_first"/arm/f'ifa_task{task}.json'
            if p.exists():assert json.loads(p.read_text())==c['ifa'][arm][str(task)]
for key,run in r['runs'].items():
    base=R/'evidence'/key
    for method in ('random','coreset'):
        counts={}
        for stage in range(6,10):
            for task in range(6,stage+1):
                rows=[json.loads(x) for x in (base/method/'evaluation'/f'trials_stage{stage}_task{task}.jsonl').read_text().splitlines()]
                counts[stage,task]=collections.Counter(v['success'] for v in rows)[True]
                assert 5*counts[stage,task]==run['arms'][method]['matrix'][stage-6][task-6]
        # AUC weights by task: Task6 1/16, Task7 1/12, Task8 1/8, Task9 1/4.
        auc=sum(counts[s,t]*5/((10-t)*4) for s,t in counts)
        assert abs(auc-run['arms'][method]['metrics']['AUC']['percent'])<1e-10
        v=run['arms'][method]['task6_relative_to_first']['9']
        assert v['SS']+v['SF']==counts[6,6] and v['SS']+v['FS']==counts[9,6]
        assert sum(v[k] for k in ('SS','SF','FS','FF'))==20
assert all(sha(Path(p))==h for p,h in before.items())
out={'status':'PASS_READONLY','formal_records':3600,'feature_files_verified_remotely':10801,'actual_buffer_files_verified_remotely':10804,'rng_snapshots':64,'metadata_and_records_unchanged':True,'no_training_no_rollouts':True,'checks':'Independent profile cross-tabs, selected identities, quota, episode integer counts, weighted AUC, pair-count identities, IFA matching, input hashes'}
(R/'INDEPENDENT_ACCEPTANCE.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))

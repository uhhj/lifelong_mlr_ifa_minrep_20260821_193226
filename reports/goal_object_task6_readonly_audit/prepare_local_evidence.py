"""Copy only existing light metadata/outcome evidence into this independent audit."""
import hashlib,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parent;EX=R.parent
dest=R/'evidence';dest.mkdir(exist_ok=False)
manifest={}
roots={('GOAL',s,'first'):EX/'20261017_coreset_goal_three_seed/final_evidence'/f'seed{s}' for s in (42,43,44)}
for s in range(40,45):
    roots['OBJECT',s,'first']=(EX/'20261016_seeds40_41_three_arm/completion_evidence'/f'seed{s}/rc' if s<42 else EX/'20261013_coreset_extension/final_evidence/remote'/f'seed{s}')
roots['OBJECT',43,'repeat1']=EX/'20261016_seed43_three_arm_repeat_v1/completion_evidence/RC'
for (suite,s,run),base in roots.items():
    target=dest/f'{suite}_{s}_{run}'
    files=[base/n for n in ('SUMMARY.json','CONTRACT.json','MODAL_CONTRACT.json','TASK6_TENSOR_PARITY.json','PRETRAIN_METADATA.json') if (base/n).exists()]
    for arm in ('random','coreset'):
        files+=list((base/arm/'evaluation').glob('trials*.jsonl'))
        files+=list((base/arm).glob('ifa_task*.json'))
        files+=list((base/arm/'storage').glob('task6.json'))
        files += [base/arm/n for n in ('INITIAL_STATE.json','epochs.json') if (base/arm/n).exists()]
    assert len(list((base/'random/evaluation').glob('trials*.jsonl')))==10
    for p in files:
        b=p.read_bytes();h=hashlib.sha256(b).hexdigest();q=target/p.relative_to(base);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b)
        manifest[q.relative_to(dest).as_posix()]={'source':str(p),'bytes':len(b),'sha256':h}
    assert all(hashlib.sha256(Path(v['source']).read_bytes()).hexdigest()==v['sha256'] for v in manifest.values())
(R/'EVIDENCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Copied',len(manifest),'light files, unchanged inputs')

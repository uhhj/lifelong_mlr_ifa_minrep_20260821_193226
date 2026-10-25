"""Post-acceptance local scalar review; no model loads, remote commands or environments."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
D=R/'final_readonly_20261025_181226'
O=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
closure=read(D/'CLOSURE_INDEX.json')
assert all(sha(R/n)==x['sha256'] for n,x in closure['files'].items())
a=read(D/'LOCAL_FINAL_AUDIT.json');assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
assert a['scientific_main_exploratory_criterion'] is False
raw=read(D/'diagnostics/G_9.json');scored=read(D/'RECOMPUTED_G_9.json')['rows']
assert [(x['index'],x['task']) for x in raw]==[(x['index'],x['task']) for x in scored]
summary={}
for t in [6,7,8]:
    rr=[x for x in raw if x['task']==t]; ss=[x for x in scored if x['task']==t]
    summary[str(t)]=dict(count=len(rr),quality_weight_mean=float(np.mean([x['quality_weight'] for x in rr])),
        projection_fraction_mean=float(np.mean([x['projection_fraction'] for x in ss])),
        weighted_anchor_mean=float(np.mean([x['weighted_anchor'] for x in ss])),
        expert_mean=float(np.mean([x['expert_exec'] for x in ss])))
source=[D/'CLOSURE_INDEX.json',D/'LOCAL_FINAL_AUDIT.json',D/'MECHANISM_DIAGNOSTICS.json',D/'diagnostics/G_9.json',D/'RECOMPUTED_G_9.json',R/'directional_prototype.py',R/'losses.py']
out=dict(status='PASS_LOCAL_POST_ACCEPTANCE_SCALAR_REVIEW',source_sha256={p.relative_to(R).as_posix():sha(p) for p in source},
    G_final_old_task_summary=summary,remote_commands=0,model_forwards=0,training_updates=0,environment_episodes=0,
    interpretation_limits=['Mean weights are not causal protection strengths or success probabilities.',
        'Projection positions in replay do not identify the actions causing closed-loop failure.',
        'The segment algebra holds for fixed teacher/expert and an interior projection in normalized output coordinates.'])
(O/'EVIDENCE.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(out,ensure_ascii=False,indent=2))

"""Create a small reproducible setup archive; excludes weights, data and old drafts."""
from pathlib import Path
import tarfile,json,hashlib
R=Path(__file__).resolve().parent
s=(R.parent/'20261013_coreset_extension/aggregate_three_seeds.py').read_text(encoding='utf-8')
s=s.replace('/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed{seed}_20261013','/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed{seed}_20261017').replace('optimizer_steps=633000','optimizer_steps=456600').replace('OBJECT Random','GOAL Random')
s=s.replace('Exploratory extension authorized after observing seed42; n=3 descriptive results, not an independent confirmatory study.','Three GOAL seeds specified before these new outcomes; cross-suite exploratory study. n=3 descriptive results, not proof of significance.')
s=s.replace('Extension was authorized after seed42 results.','Three seeds were authorized together before this GOAL comparison.')
s=s.replace("(W/'THREE_SEED_REPORT.md').write_text", "for seed,groups in allseeds.items():\n    lines.extend(['',f'## Seed {seed}','| Group | AUC | FWT | NBT |','|---|---:|---:|---:|'])\n    for arm,v in groups.items():\n        lines.append('| '+arm+' | '+' | '.join(f\"{v[\"metrics_percent\"][k]:.6f}\" for k in ('AUC','FWT','NBT'))+' |')\n        lines.extend(['',f'{arm}: rows=stage6-9, columns=task6-9',json.dumps(v['matrix']),''])\n(W/'THREE_SEED_REPORT.md').write_text")
# Python 3.8 nested quote correction.
s=s.replace('f\"{v[\"metrics_percent\"][k]:.6f}\"', 'format(v[\"metrics_percent\"][k],\".6f\")')
(R/'aggregate_three_seeds.py').write_text(s,encoding='utf-8',newline='\n')
for p in R.glob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
for p in R.glob('*.sh'):p.write_text(p.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
with tarfile.open(R/'setup.tar.gz','w:gz') as t:
    for p in sorted(R.glob('*')):
        if p.is_file() and p.suffix in ('.py','.sh','.md','.json'):t.add(p,arcname=p.name)
    for seed in (42,43,44):
        for p in sorted((R/f'seed{seed}').rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:t.add(p,arcname=str(p.relative_to(R)).replace('\\','/'))
print('Setup archive bytes', (R/'setup.tar.gz').stat().st_size)

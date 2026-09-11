"""Audit candidate seed provenance before caches or confirmation outcomes exist."""
import csv
import json
import os
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/fixed_kappa_confirmation_v2'
SEEDS={45,46,47}
name_pattern=re.compile(r'(?:seed|seeds)[_ =:-]*(?:45|46|47)(?!\d)',re.I)
content_pattern=re.compile(r'''(?i)(?:seed[s]?\s*["']?\s*[:=]\s*["']?\s*(?:45|46|47)(?!\d)|seed[_-](?:45|46|47)(?!\d))''')
SKIP={'confirmation_experiment','.git','.conda','__pycache__','node_modules','runtime'}
def seed_values(x,path=''):
    found=[]
    if isinstance(x,dict):
        for k,v in x.items():
            if str(k).lower() in ('seed','seeds','train_seed','training_seed','eval_seed'):
                vs=v if isinstance(v,list) else [v]
                for z in vs:
                    if str(z) in ('45','46','47'):found.append({'key':path+'.'+str(k),'value':z})
            found.extend(seed_values(v,path+'.'+str(k)))
    elif isinstance(x,list):
        for i,v in enumerate(x):
            if isinstance(v,(dict,list)):found.extend(seed_values(v,path+f'[{i}]'))
    return found
def run():
    OUT.mkdir(parents=True,exist_ok=True)
    hits=[];scanned=0;json_count=0;text_count=0;skipped=[];csv_count=0
    for parent,dirs,files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in SKIP and not (Path(parent)==ROOT/'reports' and d=='fixed_kappa_confirmation_v2')]
        for d in dirs:
            rel=(Path(parent)/d).relative_to(ROOT).as_posix()
            if name_pattern.search(d):hits.append({'path':rel,'kind':'directory_name'})
        for name in files:
            p=Path(parent)/name;rel=p.relative_to(ROOT).as_posix();scanned+=1
            if name_pattern.search(name):hits.append({'path':rel,'kind':'file_name'})
            if p.suffix.lower() not in ('.json','.jsonl','.yaml','.yml','.toml','.py','.sh','.md','.txt','.csv','.log'):continue
            if p.stat().st_size>100*1024*1024:
                skipped.append({'path':rel,'reason':'text_over_100MiB'});continue
            try:
                if p.suffix=='.json':
                    json_count+=1
                    for hit in seed_values(json.loads(p.read_text(encoding='utf-8-sig'))):hits.append(dict(path=rel,kind='structured_json',**hit))
                elif p.suffix=='.csv':
                    csv_count+=1
                    with p.open(encoding='utf-8-sig',newline='') as f:
                        reader=csv.DictReader(f)
                        keys=[k for k in (reader.fieldnames or []) if k.lower() in ('seed','train_seed','training_seed','eval_seed')]
                        if keys:
                            for i,r in enumerate(reader,2):
                                for k in keys:
                                    if r[k] in ('45','46','47'):hits.append({'path':rel,'kind':'csv_seed','line':i,'seed':r[k]})
                else:
                    text_count+=1
                    with p.open(encoding='utf-8-sig',errors='replace') as f:
                        for i,line in enumerate(f,1):
                            if content_pattern.search(line):hits.append({'path':rel,'kind':'text_seed','line':i,'text':line[:500].rstrip()})
            except (UnicodeError,json.JSONDecodeError,csv.Error) as e:skipped.append({'path':rel,'reason':str(e)[:200]})
    history=subprocess.run(['git','log','--all','--format=%H %s','--regexp-ignore-case','--extended-regexp','--grep=seed[^0-9]*(45|46|47)([^0-9]|$)'],cwd=ROOT,text=True,capture_output=True,check=True)
    result={'status':'REVIEW_REQUIRED' if hits or skipped or history.stdout.strip() else 'PASS',
            'seeds':[45,46,47],'scanned_file_names':scanned,'json_files':json_count,'csv_files':csv_count,
            'text_files':text_count,'hits':hits,'unparsed':skipped,'git_history_subject_hits':history.stdout.splitlines(),
            'excluded':['.git objects (git subjects searched separately)','runtime dependency bundle','confirmation package and its own reports','Python caches/node_modules/conda'],
            'rule':'Any prior same-method selection or tuning use fails; never replace seeds automatically.'}
    (OUT/'SEED_AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False)[:20000])
if __name__=='__main__':run()

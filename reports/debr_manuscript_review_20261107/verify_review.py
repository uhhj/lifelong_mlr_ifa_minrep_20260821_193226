"""Read-only numeric verification; private manuscript text is never published."""
from pathlib import Path
import csv, datetime, hashlib, json, statistics, zipfile
from lxml import etree

ROOT=Path(__file__).resolve().parent
W=ROOT.parents[1]
DOC=Path('C:/Users/Lenovo/Downloads/DEBR_Complete_Manuscript.docx')
DATA=W/'execution/20261103_lambda_r_goal_sensitivity/publication_figures_20261106/data'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with zipfile.ZipFile(DOC) as z:
    body=etree.fromstring(z.read('word/document.xml')).find('w:body',NS)
    tables=[]
    for t in body.findall('w:tbl',NS):
        tables.append([[''.join(c.xpath('.//w:t/text()',namespaces=NS)) for c in r.findall('w:tc',NS)] for r in t.findall('w:tr',NS)])
checks=[]
sources={}
for fname,key in [('method_metrics_per_seed.csv','method'),('lambda_metrics_per_seed.csv','lambda_R')]:
    p=DATA/fname;sources[str(p.relative_to(W))]=sha(p)
    rows=list(csv.DictReader(p.open(encoding='utf-8-sig')))
    for suite in ['OBJECT','GOAL']:
        for v in sorted({r[key] for r in rows}):
            a=[r for r in rows if r['suite']==suite and r[key]==v]
            assert {int(r['seed']) for r in a}=={41,42,43}
            vals={m:[statistics.mean(float(r[m]) for r in a),statistics.stdev(float(r[m]) for r in a)] for m in ['FWT','NBT','AUC']}
            if key=='lambda_R':
                target=tables[3][1+(0 if suite=='OBJECT' else 3)+['0.05','0.1','0.2'].index(v)][2:5]
            elif v=='Q':
                target=tables[1][5][(1 if suite=='OBJECT' else 4):(4 if suite=='OBJECT' else 7)]
            else:
                # Table 3: two header rows, standard C, matched C, S.
                target=tables[2][{'C':3,'S':4}[v]][(2 if suite=='OBJECT' else 6):(5 if suite=='OBJECT' else 9)]
            actual=[f'{x:.1f}±{s:.1f}' for x,s in vals.values()]
            assert actual==target,(suite,v,actual,target)
            checks.append(dict(suite=suite,configuration=v,source=fname,metrics=vals,display_verified=actual))
audit=dict(status='PASS_SELECTED_TABLE_NUMERIC_RECOMPUTATION',observed=datetime.datetime.now().astimezone().isoformat(),manuscript_sha256=sha(DOC),manuscript_bytes=DOC.stat().st_size,tables=len(tables),checks=checks,source_sha256=sources,scope='Table 2 Q, Table 3 matched C/S, and all Table 4 coefficients; does not verify every historical row',original_modified=False,full_page_render_verified=False,new_training_or_environment_calls=0)
(ROOT/'VERIFICATION.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
print(audit['status'],len(checks))

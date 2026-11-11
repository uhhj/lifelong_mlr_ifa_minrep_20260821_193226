"""Audit manuscript configuration and numbers without changing the document."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
import csv, datetime, hashlib, io, json, math, re, statistics

R=Path(__file__).resolve().parent
W=R.parents[1]
bindings={}
def read(path):
    p=Path(path)
    if not p.is_absolute():p=W/p
    raw=p.read_bytes()
    key=p.relative_to(W).as_posix() if p.is_relative_to(W) else str(p)
    bindings[key]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    return raw
def J(path):return json.loads(read(path).decode('utf-8-sig'))
def CSV(path):return list(csv.DictReader(io.StringIO(read(path).decode('utf-8-sig'))))
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
docs=J(R/'SOURCE_DOCUMENTS.json')
tables={}; trees={}
for name,meta in docs.items():
    raw=read(meta['path']);assert hashlib.sha256(raw).hexdigest()==meta['sha256']
    with ZipFile(io.BytesIO(raw)) as z: tree=etree.fromstring(z.read('word/document.xml'))
    trees[name]=tree
    tables[name]=[[[''.join(c.xpath('.//w:t/text()|.//m:t/text()',namespaces=ns)) for c in row.findall('w:tc',ns)] for row in tbl.findall('w:tr',ns)] for tbl in tree.find('w:body',ns).findall('w:tbl',ns)]
assert len(tables['latest'])==len(tables['previous'])==5
pattern=re.compile(r'(-?\d+\.\d+)\s*±\s*(\d+\.\d+)')
def pairs(row):return [(float(a),float(b)) for a,b in pattern.findall(' | '.join(row))]
for a,b in zip(tables['latest'],tables['previous']):
    assert len(a)==len(b)
    assert [pairs(row) for row in a]==[pairs(row) for row in b]
checked=0
def compare(actual,expected):
    global checked
    assert len(actual)==len(expected),(actual,expected)
    for x,y in zip(actual,expected):
        for aa,bb in zip(x,y):assert abs(aa-bb)<=.05000001,(aa,bb)
        checked+=1
def agg(rows,group,key):
    out=[]
    for suite in ['OBJECT','GOAL']:
        subset=[r for r in rows if r[group]==key and r['suite']==suite]
        assert sorted(int(r['seed']) for r in subset)==[41,42,43]
        out.extend((statistics.mean(float(r[m]) for r in subset),statistics.stdev(float(r[m]) for r in subset)) for m in ['FWT','NBT','AUC'])
    return out
main=J('execution/20261107_main_comparison_table/TABLE_AUDIT.json')
assert main['status']=='PASS_BASELINE_TRANSCRIPTION_AND_OURS_RECOMPUTATION'
for row,ref in zip(tables['latest'][0][2:],main['rows']):
    compare(pairs(row),[(v['mean'],v['sd']) for v in ref['values']])
baseline=[(v['mean'],v['sd']) for v in main['rows'][7]['values']]
ab=CSV('execution/20261107_ablation_table/per_seed_metrics.csv')
kc=CSV('execution/20261109_kcenter_ra_matched_comparison/METRICS.csv')
for row,expected in zip(tables['latest'][1][2:],[baseline,agg(kc,'method','K-Center'),agg(ab,'configuration','RA-Coreset')]):compare(pairs(row),expected)
component_expected=[baseline]+[agg(ab,'configuration',key) for key in ['RA-Coreset','+ Execution Supervision','+ Support-Weighted Retention','Ours']]
for row,expected in zip(tables['latest'][2][2:],component_expected):compare(pairs(row),expected)
budget=CSV('execution/20261107_optimizer_budget_table/per_seed_metrics.csv')
for row,key in zip(tables['latest'][3][2:],['+Execution Supervision(Standard Budget)','+Execution Supervision(Matched Budget)','Ours']):compare(pairs(row),agg(budget,'configuration',key))
for row,counts in zip(tables['latest'][3][2:],[(105500,76100),(195300,142200),(195300,142200)]):
    observed=[int(v.replace(',','')) for v in row if re.fullmatch(r'\d{2,3},\d{3}',v)]
    assert observed==list(counts)
lam=CSV('execution/20261107_lambda_sensitivity_table/per_seed_metrics.csv')
for row,(suite,value) in zip(tables['latest'][4][1:],[(s,v) for s in ['OBJECT','GOAL'] for v in [.05,.1,.2]]):
    rr=[r for r in lam if r['suite']==suite and math.isclose(float(r['lambda_R']),value)]
    assert sorted(int(r['seed']) for r in rr)==[41,42,43]
    compare(pairs(row),[(statistics.mean(float(r[m]) for r in rr),statistics.stdev(float(r[m]) for r in rr)) for m in ['FWT','NBT','AUC']])

# Text extraction omits implicit OMML delimiters; verify the original structure.
eval_par=trees['latest'].find('w:body',ns)[57]
delimiters=eval_par.xpath('.//m:d',namespaces=ns)
assert any(''.join(x.xpath('.//m:t/text()',namespaces=ns))=='1+2+3+4' for x in delimiters)
assert 20*sum([1,2,3,4])==200

sources=[
 'execution/20261107_align_cqs_audit/REPORT.md',
 'execution/20261107_mlr_ifa_reproduction_audit/REPORT.md',
 'execution/20261030_equal_optimizer_budget_cs_two_suites/PROTOCOL.md',
 'execution/20261030_equal_optimizer_budget_cs_two_suites/FINAL_REPORT.md',
 'execution/20261031_full_q_vs_latest_s_two_suites/PROTOCOL.md',
 'execution/20261109_kcenter_ra_matched_comparison/REPORT.md',
 'execution/20261109_q1_user_resume/ACCEPTANCE.json',
 'execution/20261109_q1_user_resume/FINAL_REPORT.md',
 'execution/20261029_support_gate_full_s_goal/PROTOCOL.md',
 'execution/20261029_support_gate_full_s_goal/seed41/train.py',
 'execution/20261029_support_gate_full_s_goal/seed41/collect.py',
 'execution/20261029_support_gate_full_s_goal/seed41/support_memory.py',
 'execution/20261029_support_gate_full_s_goal/seed41/fork_tools.py',
 'execution/20261029_support_gate_full_s_goal/seed41/updates.py',
 'execution/20261031_full_q_vs_latest_s_two_suites/engineering_startup_v2/GOAL_41/runtime.py',
 'execution/20261022_align_object_extension_goal_conditional/additions/caet_v3_exp/model.py',
 'execution/20261103_lambda_r_goal_sensitivity/publication_figures_20261106/data/gate_condition_patterns.csv',
 'execution/20261103_lambda_r_goal_sensitivity/publication_figures_20261106/data/stage_task_differences.csv',
]
for path in sources:read(path)
out={
 'status':'PASS_NUMERIC_CROSSCHECK_WITH_TEXTUAL_CONFIGURATION_FINDINGS',
 'observed':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'documents':docs,'tables_checked':5,'mean_sd_cells_checked':checked,
 'table_numeric_values_unchanged_between_versions':True,
 'budget_counts_verified':{'GOAL':[76100,142200],'OBJECT':[105500,195300]},
 'native_evaluation_formula_has_parentheses':True,
 'new_training':0,'new_model_forwards':0,'new_environment':0,'manuscript_modified':False,
 'finding_classes':{'direct_claim_contradiction':['Abstract all-metrics-improved versus GOAL NBT'],
 'reproducibility_ambiguity':['Scale population and freezing time','Masking versus removing training population','Fixed versus maximum history length'],
 'scope_clarification':['External versus locally executed results','Enabled components and collection applicability','Cumulative configurations versus a single loss switch','Historical versus new evaluation random streams'],
 'version_regression':['Latest rewritten version reverted CADL naming'],
 'minor_rounding':['OBJECT FWT gain 13.7 from displayed values versus 13.8 from original means']},
 'source_bindings':bindings,
}
(R/'AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in out.items() if k not in ['source_bindings','documents','finding_classes']},ensure_ascii=False,indent=2))

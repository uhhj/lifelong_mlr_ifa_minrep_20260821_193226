import json,sys,hashlib
from pathlib import Path
from independent_audit import audit_run
p=Path(sys.argv[1]);m=json.loads((p/'PACKAGE_MANIFEST.json').read_text());assert m['status']=='LIGHT_MANIFEST_READY' and not m['residual_workers']
for n,v in m['entries'].items():
 raw=(p/n).read_bytes();assert len(raw)==v['bytes'] and hashlib.sha256(raw).hexdigest()==v['sha256']
e=m['entry'];result=audit_run(p/'current',p/'baseline',e['seed'],e['suite'])
r=json.loads((p/'launcher/receipt.json').read_text());server=json.loads((p/'launcher/server_independent.json').read_text())
assert r['key']==e['key'] and r['acceptance_sha256']==result['acceptance_sha256'] and r['contract_sha256']==result['contract_sha256']
assert r['independent_audit_sha256']==hashlib.sha256((p/'launcher/server_independent.json').read_bytes()).hexdigest() and result==server
result['status']='COMPLETED_LOCAL_INDEPENDENT_PASS';result['package_manifest_sha256']=hashlib.sha256((p/'PACKAGE_MANIFEST.json').read_bytes()).hexdigest()
(p/'LOCAL_FINAL_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# '+e['key']+' 最终独立验收','','完整轻量manifest/SHA、400epoch、主预算、4CP服务器严格加载凭据、原RA成员载荷/IFA/初始化、逐步mask账目、200episode/20vector/10cell与有效退出复算PASS。单seed探索性。','','|指标|RA %|ALIGN %|差 pp|','|---|---:|---:|---:|']
for k,v in result['metrics'].items():lines.append('|'+k+'|'+f"{result['baseline'][k]['percent']:.6f}|{v['percent']:.6f}|{result['delta'][k]['percent']:+.6f}| ")
(p/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps(result,indent=2))

"""Lossless structured JSON splitting for reviewed scalar diagnostic evidence."""
import hashlib,json,sys
from pathlib import Path
def main():
 p=Path(sys.argv[1]);raw=(p/'EVIDENCE.json').read_bytes();d=json.loads(raw);s=p/'evidence_parts';s.mkdir(exist_ok=True);mapping={};dicts=[k for k in ('stages','calibrations','ood') if k in d];lists=[k for k in ('models','predictors') if k in d]
 for key in dicts+lists:
  entries=d[key].items() if key in dicts else enumerate(d[key])
  for sub,value in entries:
   name=key+'_'+str(sub).replace('/','_')+'.json';(s/name).write_text(json.dumps(value,ensure_ascii=False)+'\n',encoding='utf8');mapping[key+'/'+str(sub)]=name
 meta={k:v for k,v in d.items() if k not in dicts+lists};(s/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False)+'\n',encoding='utf8')
 manifest=dict(original_sha256=hashlib.sha256(raw).hexdigest(),original_keys=list(d),mapping=mapping,dict_keys={k:list(d[k]) for k in dicts},list_counts={k:len(d[k]) for k in lists},metadata='metadata.json',files={f.name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in s.glob('*.json') if f.name!='MANIFEST.json'})
 (s/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8');rebuilt={}
 for key in manifest['original_keys']:
  if key in dicts:rebuilt[key]={sub:json.loads((s/mapping[key+'/'+sub]).read_text(encoding='utf8')) for sub in manifest['dict_keys'][key]}
  elif key in lists:rebuilt[key]=[json.loads((s/mapping[key+'/'+str(i)]).read_text(encoding='utf8')) for i in range(manifest['list_counts'][key])]
  else:rebuilt[key]=meta[key]
 assert (json.dumps(rebuilt)+'\n').encode()==raw
 report=dict(status='LOSSLESS_STRUCTURED_SPLIT_PASS',original_sha256=manifest['original_sha256'],files=len(manifest['files']),max_bytes=max(x['bytes'] for x in manifest['files'].values()),reconstructed_raw_bytes_exact=True)
 assert report['max_bytes']<8*1024**2
 (p/'LIGHT_PACKAGE_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

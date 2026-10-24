"""Local provenance/coverage audit for a finite stopped semantic screen."""
import json,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'20261024_ra_align_paired'
def js(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wr(n,x):(R/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a=js(R/'ASSET_PREFLIGHT.json'); logs=[json.loads(x) for x in (R/'VIEW_LOG.jsonl').read_text().splitlines()]
assert a['source_unchanged'] and a['worker_exit'].strip()==a['supervisor_exit'].strip()=='0'
assert a['initial_raw_windows_equal_across_four_drivers']==list(range(20))
assert all(x['request']['seed']==41 for x in logs)
# One large PNG response was truncated before display; replaced by JPEG without new env/model calls.
rejected='b86cd1b7409e784504c714266b1dcb87738fc2069838f3e10da6244cd39a1929'
seen=set()
for x in logs:
 if x['png_sha256']==rejected:continue
 q=x['request']
 for i in q['indices']:
  for m in (['RA','ALIGN'] if q['window']=='first_mean_delta' else [q['method']]):
   for d in [8,9]:seen.add((m,d,i,q['window']))
cases=[];pairs=js(P/'final_20261024_193008/PAIRED_TRAJECTORIES.json')
tr={(p['method'],p['initial_index']):p['transition'] for p in pairs if p['seed']==41}
for c in a['discovery_window_coverage']:
 m=c['run'].split('_')[0];d=int(c['run'].split('_')[2][6:]);i=c['initial_index']
 for w in c['windows']:assert (m,d,i,w['window']) in seen
 intervals=sorted((w['start'],w['end']) for w in c['windows']);covered=set(t for lo,hi in intervals for t in range(lo,hi+1))
 gaps=[];start=None
 for t in range(c['steps']+1):
  if t not in covered and start is None:start=t
  if t in covered and start is not None:gaps.append([start,t-1]);start=None
 if start is not None:gaps.append([start,c['steps']])
 cases.append({'method':m,'driver':d,'initial_index':i,'transition':tr[(m,i)],'saved_window_count':len(intervals),'unique_rgb_times':len(covered),'largest_rgb_gap':max((hi-lo+1 for lo,hi in gaps),default=0),'gaps':gaps,'windows':c['windows'],'semantic_scope':'Sampled first/middle/last frames per noninitial window; no full continuous video.'})
assert len(cases)==80
wr('WINDOW_COVERAGE_AUDIT.json',{'status':'PASS_IMAGE_SOURCE_HASH_AND_DISCOVERY_COVERAGE','cases':cases,'source_npz_windows':sum(len(c['windows']) for c in cases),'unique_pairs':40,'trajectories':80,'holdout_images_viewed':0,'rejected_display_sha256':rejected,'note':'Coverage proves inspected contact-sheet requests exist and each source window was hash checked remotely; semantic correctness is not machine-verified.'})
wr('SCREEN_DECISION.json',{'status':'STOP_DESIGN_INSUFFICIENT_TEMPORAL_LOCALIZATION','candidate_considered':'Target-plate vicinity with little apparent progress in terminal window','admitted_rule':None,'stop_clause':'PROTOCOL step3: terminal-only phenomenon without a localized preceding decision stage','holdout_evaluation':'NOT_ENTERED','teacher_replacement':'NOT_TESTED','new_environments':0,'new_model_forwards':0,'training_updates':0,'replay_attempts':0,'simulator_restoration':'NOT_VERIFIED; original full simulator/controller states were not stored; verified replay would be required if a semantic trigger existed.','no_claims':['Teacher replacement ineffective','Mechanism nonexistent','Grasp/release causal failure identified']})
files=['PROTOCOL.md','ASSET_PREFLIGHT.json','SEMANTIC_NOTES.md','VIEW_LOG.jsonl','WINDOW_COVERAGE_AUDIT.json','SCREEN_DECISION.json','visual_readonly.py','visual_bridge.py','view_client.py','REPORT.md','IMPLEMENTATION_NOTES.md']
first_view=min(datetime.datetime.fromisoformat(x['time']).timestamp() for x in logs)
assert (R/'PROTOCOL.md').stat().st_mtime<first_view
wr('REGISTRATION_AND_CLOSURE.json',{'status':'CLOSED_NO_ACTIVE_EXPERIMENT','protocol_sha256':sha(R/'PROTOCOL.md'),'protocol_file_mtime':(R/'PROTOCOL.md').stat().st_mtime,'first_display_request':min(x['time'] for x in logs),'registration_note':'Protocol written before first display, as supported by session execution and local file mtime; SHA recorded during closure, not an external timestamp authority.','bridge':'Explicit close request succeeded; local process exit0 observed','automation_created':False,'private_payload_local_files':False,'scientific_stop':'Design-stage insufficient temporal localization, not a negative intervention result.'})
files.append('REGISTRATION_AND_CLOSURE.json')
wr('LOCAL_SCREEN_AUDIT.json',{'status':'PASS_FINITE_SCREEN_PROVENANCE_NOT_CAUSAL_ACCEPTANCE','time':datetime.datetime.now().astimezone().isoformat(),'evidence':{f:sha(R/f) for f in files},'prior_final_audit_sha256':sha(P/'final_20261024_193008/LOCAL_FINAL_AUDIT.json'),'no_private_image_files_local':not any(p.suffix.lower() in ['.png','.jpg','.jpeg','.npz','.npy','.pth'] for p in R.rglob('*'))})
wr('RUN_STATUS.json',{'status':'COMPLETE_SCREEN_STOPPED_NO_ADMITTED_MECHANISM','observation':datetime.datetime.now().astimezone().isoformat(),'discovery_pairs':40,'discovery_trajectories':80,'holdout_images_viewed':0,'intervention_episodes':0,'model_forwards':0,'training_updates':0,'automation':'NONE_NO_RUNNING_JOB','next_action':'No automatic continuation or protection development around this unlocalized phenomenon.'})
print(json.dumps({'status':'PASS_FINITE_SCREEN_PROVENANCE_NOT_CAUSAL_ACCEPTANCE','windows':sum(len(c['windows']) for c in cases),'cases':len(cases),'decision':'STOP_DESIGN_INSUFFICIENT_TEMPORAL_LOCALIZATION'}))

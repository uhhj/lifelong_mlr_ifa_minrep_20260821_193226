"""Independent read-only COVR launch/early training audit; not final acceptance."""
import argparse,collections,datetime,hashlib,json,statistics
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);a=ap.parse_args();p=a.directory
 raw=(p/'COVR_DETAILS.json').read_bytes();d=json.loads(raw)
 assert hashlib.sha256(raw).hexdigest()==json.loads((p/'COVR_DETAILS_RECEIPT.json').read_text())['sha256']
 assert d['decision']['method']=='COVR' and d['decision']['seed']==42
 assert d['decision']['deferred']['not_completed'] and d['controller_exit'] is None and d['blocked'] is None
 assert d['contract']['hashes']=='PASS' and d['contract']['sources']==204
 pre=d['preflight'];assert pre['status']=='PASS' and pre['optimizer_calls']==4 and pre['environment_steps']==28
 assert pre['zero_aux_native_update_exact'] and pre['full_state_epoch_recovery_exact']
 rows=d['epochs'];assert rows and len({(x['task'],x['epoch']) for x in rows})==len(rows)
 assert all(x['task']==6 and x['epoch']==i and x['batches']==157 for i,x in enumerate(rows))
 assert sum(x['batches'] for x in rows)==rows[-1]['steps']
 assert all(v=='0' for v in d['exits'].values())
 starts=collections.Counter(x['phase'] for x in d['phase_events'] if x['event']=='start')
 assert starts=={'train6':1} and any('covr_exp.train' in x for x in d['processes'])
 avg=statistics.mean(x['seconds'] for x in rows)
 result=dict(status='COVR_EARLY_TRAINING_PASS',experiment_status='PARTIAL',
   observed_at=datetime.datetime.fromtimestamp(d['observed_end_unix'],datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
   epochs=len(rows),optimizer_steps=rows[-1]['steps'],formal_records=0,
   mean_epoch_seconds=avg,remaining_task6_minutes=(100-len(rows))*avg/60,
   full_run_eta=None,eta_limitation='No measured COVR collection throughput yet; no reliable complete-run ETA',
   source_input_hashes='PASS',preflight_optimizer_calls=4,preflight_engineering_environment_steps=28,
   local_optimizer_calls=0,local_environment_steps=0,details_sha256=hashlib.sha256(raw).hexdigest())
 (p/'COVR_LAUNCH_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

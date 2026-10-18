"""Read-only ETA from actual completed vector attempts; no scientific changes."""
from pathlib import Path
import json,datetime,hashlib
P=Path(__file__).resolve().parent
def main():
    s=json.loads((P/'snapshot.json').read_text())
    rows=[json.loads(x) for x in (P/'attempts_stage6_task6.jsonl').read_text().splitlines()]
    starts={r['group']:r['unix'] for r in rows if r['event']=='vector_attempt_start'}
    done=[r for r in rows if r['event']=='vector_attempt_complete']
    assert len(done)==s['completed_vectors']==1 and s['formal_records']==10
    assert s['epochs']==40 and s['optimizer_steps']==10550 and s['technical_exceptions']==0
    measured=done[0]['unix']-starts[done[0]['group']]
    active=max(starts);elapsed=max(0,s['time']-starts[active])
    remaining=(20-len(done))*measured-min(elapsed,measured)
    tz=datetime.timezone(datetime.timedelta(hours=8))
    out=dict(status='PARTIAL',observed_CST=datetime.datetime.fromtimestamp(s['time'],tz).isoformat(),completed_episodes=10,planned_episodes=200,
        measured_vector_seconds=measured,policy_steps=done[0]['steps'],seconds_per_vector_step=measured/done[0]['steps'],
        estimated_remaining_eval_minutes=remaining/60,estimated_acceptance_minutes=[5,10],
        estimate_with_acceptance_CST=[datetime.datetime.fromtimestamp(s['time']+remaining+v*60,tz).isoformat() for v in [5,10]],
        user_facing_conservative_remaining_minutes=[90,120],uncertainty='Only the first completed vector group; later early successes, environment creation and throughput vary.',
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [P/'snapshot.json',P/'attempts_stage6_task6.jsonl']})
    (P/'ETA.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()

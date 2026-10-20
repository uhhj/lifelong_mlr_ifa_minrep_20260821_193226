"""Synthetic accounting only; no experimental outcomes or simulation."""
import tempfile,json,itertools
from pathlib import Path
from analyze import analyze
def main():
 ids=["normal"]+["a%d_%s_r%s"%(a,s,r) for a in range(3) for s in ("m","p") for r in ("original",0,1,2,3,4)]
 assert len(ids)==len(set(ids))==37
 with tempfile.TemporaryDirectory() as d:
  p=Path(d);rows=[];attempts=[]
  for i,rep,k in itertools.product([20,21,22],range(3),ids):
   success=k.endswith("r0")
   rows.append(dict(initial_index=i,repeat=rep,candidate={"id":k},status="COMMITTED",success=success,
    steps=1,noise_sha256=str((i,rep))))
   attempts.extend([dict(event="branch_start"),dict(event="branch_complete")])
  (p/"episodes.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rows))
  (p/"attempts.jsonl").write_text("".join(json.dumps(x)+"\n" for x in attempts))
  (p/"EXECUTION_COMPLETE.json").write_text(json.dumps(dict(eligible_states=[20,21,22],elapsed_seconds=1,
   counts=dict(screening_steps=333,prefix_steps=0,engineering_steps=0,optimizer_calls=0))))
  (p/"ENGINEERING_GATE.json").write_text('{"status":"PASS"}')
  r=analyze(p);assert r["branches"]==333 and len(r["directional_signal"])==6
  with (p/"episodes.jsonl").open("a") as f:f.write(json.dumps(rows[0])+"\n")
  try:analyze(p)
  except AssertionError:pass
  else:raise AssertionError("duplicate must fail")
 print("PASS: candidate cardinality, complete-grid accounting, pairing direction, signal flag, duplicate rejection. SYNTHETIC ONLY.")
if __name__=="__main__":main()

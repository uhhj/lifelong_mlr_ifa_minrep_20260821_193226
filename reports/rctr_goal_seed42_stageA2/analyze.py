"""Read-only outcome recomputation; never imports simulation or training libraries."""
import json,sys,csv,hashlib,itertools
from pathlib import Path
def read(p):return json.loads(p.read_text())
def analyze(root):
 rows=[json.loads(x) for x in (root/"episodes.jsonl").read_text().splitlines()]
 complete=read(root/"EXECUTION_COMPLETE.json");gate=read(root/"ENGINEERING_GATE.json")
 assert gate["status"]=="PASS"
 eligible=complete["eligible_states"]
 ids=["normal"]+["a%d_%s_r%s"%(a,s,r) for a in range(3) for s in ("m","p") for r in ("original",0,1,2,3,4)]
 expected=set(itertools.product(eligible,range(3),ids))
 by={(r["initial_index"],r["repeat"],r["candidate"]["id"]):r for r in rows}
 assert len(by)==len(rows) and set(by)==expected
 assert len(rows)==111*len(eligible)<=333
 assert sum(r["steps"] for r in rows)==complete["counts"]["screening_steps"]
 assert all(0<=r["steps"]<=600 and r["status"] in ("COMMITTED","SKIPPED") for r in rows)
 for i in eligible:
  for rep in range(3):
   assert len({by[i,rep,k]["noise_sha256"] for k in ids})==1
 attempts=[json.loads(x) for x in (root/"attempts.jsonl").read_text().splitlines()]
 assert not any(r["event"]=="technical_exception" for r in attempts)
 starts=[r for r in attempts if r["event"]=="branch_start"]
 ends=[r for r in attempts if r["event"]=="branch_complete"]
 assert len(starts)==len(ends)==len(rows)
 counts=complete["counts"];assert counts["optimizer_calls"]==0
 assert sum(r["event"]=="control_step_complete" and r["kind"]=="prefix" for r in attempts)==counts["prefix_steps"]<=135
 assert sum(r["event"]=="control_step_complete" and r["kind"]=="engineering" for r in attempts)==counts["engineering_steps"]<=48
 tables=[]
 for i in eligible:
  for ident in ids:
   rr=[by[i,r,ident] for r in range(3)];ok=[x for x in rr if x["status"]=="COMMITTED"]
   tables.append(dict(initial_index=i,candidate=ident,committed=len(ok),skipped=3-len(ok),
    successes=sum(x["success"] for x in ok),success_frequency=sum(x["success"] for x in ok)/len(ok) if ok else None,
    steps=sum(x["steps"] for x in rr)))
 pairs=[]
 for i in eligible:
  for ident in ids[1:]:
   cand=by[i,0,ident]["candidate"];ref=ident.rsplit("_r",1)[0]+"_roriginal"
   for reference in dict.fromkeys(["normal",ref]):
    if reference==ident:continue
    bins={k:0 for k in ("SS","SF","FS","FF")};missing=0
    for rep in range(3):
     a=by[i,rep,reference];b=by[i,rep,ident]
     if a["status"]!="COMMITTED" or b["status"]!="COMMITTED":missing+=1;continue
     label=("S" if a["success"] else "F")+("S" if b["success"] else "F")
     bins[label]+=1
    pairs.append(dict(initial_index=i,candidate=ident,reference=reference,**bins,missing=missing,
     net_success_gain=bins["FS"]-bins["SF"]))
 flagged={}
 for p in pairs:
  if p["reference"]!="normal" and p["FS"]>=2 and p["SF"]==0 and p["missing"]==0:
   flagged.setdefault(p["candidate"],[]).append(p["initial_index"])
 signal={k:v for k,v in flagged.items() if len(v)>=2}
 for name,values in [("candidate_summary.csv",tables),("paired_summary.csv",pairs)]:
  with (root/name).open("w",newline="") as f:
   w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
 report=dict(status="SCREENING_ANALYZED_NOT_TRAINING_CERTIFIED",branches=len(rows),eligible_states=eligible,
  committed=sum(r["status"]=="COMMITTED" for r in rows),skipped=sum(r["status"]=="SKIPPED" for r in rows),
  counts=counts,engineering_gate="PASS",directional_signal=signal,
  independent_certification="NOT_AUTHORIZED_NOT_RUN",training_authorized=False,
  elapsed_seconds=complete["elapsed_seconds"],all_candidate_rows=len(tables),all_pair_rows=len(pairs))
 (root/"ANALYSIS.json").write_text(json.dumps(report,indent=2)+"\n")
 (root/"REPORT.md").write_text("# RCTR Stage A2 exploratory screening\n\n"+json.dumps(report,indent=2)+
  "\n\nAll outcomes: candidate_summary.csv; matched comparisons: paired_summary.csv. "
  "SS/SF/FS/FF order is reference then candidate. Three repeats are exploratory; "
  "candidate selection is multiple comparison and does not establish significance or transferable control rules. "
  "No training or additional certification authorized. Historical Stage A remains BLOCKED.\n")
 print(json.dumps(report,indent=2));return report
if __name__=="__main__":analyze(Path(sys.argv[1]))

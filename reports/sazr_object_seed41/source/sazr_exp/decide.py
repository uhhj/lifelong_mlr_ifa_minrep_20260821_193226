from .common import *
def main():
    verify();assert not (O/"DECISION.json").exists()
    rs=[]
    for name in ("validation_parent","validation_candidate"):
        assert js(O/name/"COMPLETE.json")["episodes"]==10
        rs.append({r["initial_index"]:r for r in [json.loads(l) for l in (O/name/"episodes.jsonl").read_text().splitlines()]})
    assert set(rs[0])==set(rs[1])==set(range(30,40))
    pairs=[]
    for i in range(30,40):
        a,b=rs[0][i],rs[1][i]
        for k in ("initial_sha256","parent_sha256","noise_seed","noise_sha256","env_seed"):assert a[k]==b[k]
        pairs.append(dict(initial_index=i,parent=a["success"],candidate=b["success"],
            outcome=("S" if a["success"] else "F")+("S" if b["success"] else "F")))
    counts={k:sum(r["outcome"]==k for r in pairs) for k in ("SS","SF","FS","FF")}
    write(O/"DECISION.json",dict(enabled=counts["FS"]>=2 and counts["SF"]==0,rule="FS>=2 and SF=0",
        counts=counts,pairs=pairs,checkpoint_sha256=sha(O/"models/sazr_stage6_calibration_ep9.pth"),
        files={str(O/n/"episodes.jsonl"):sha(O/n/"episodes.jsonl") for n in ("validation_parent","validation_candidate")}))
    print("DECISION",counts,"enabled",counts["FS"]>=2 and counts["SF"]==0)
if __name__=="__main__":main()

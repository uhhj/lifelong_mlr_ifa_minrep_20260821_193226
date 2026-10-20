from .common import *

def main():
    verify();assert not (O/"ANCHOR_CONTRACT.json").exists()
    assert js(O/"anchor/COMPLETE.json")["episodes"]==10
    traces=torch.load(O/"successful_parent_features.pth",map_location="cpu")
    assert traces,"No successful parent trajectory: stop, do not collect additional episodes"
    ids=sorted(traces);times=choose_anchor_times([len(traces[i]) for i in ids])
    z=torch.stack([traces[i][t] for i,ts in zip(ids,times) for t in ts])
    q,info=basis(z)
    save(O/"anchor_basis.pth",dict(z=z,Q=q))
    write(O/"ANCHORS.json",dict(status="PASS",successful_initials=ids,anchors=[
        dict(initial_index=i,policy_step=t,history_length=min(t+1,8)) for i,ts in zip(ids,times) for t in ts],
        **info,encoder="frozen current parent stage6 temporal transformer"))
    write(O/"ANCHOR_CONTRACT.json",dict(status="FROZEN",parent_contract_sha256=sha(O/"CONTRACT.json"),
        files={str(O/n):sha(O/n) for n in ("anchor_basis.pth","successful_parent_features.pth","ANCHORS.json","anchor/episodes.jsonl")}))
    print("ANCHORS",len(z),"rank",info["rank"])
if __name__=="__main__":main()

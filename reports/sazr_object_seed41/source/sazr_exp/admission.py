from .common import *
from .reference_core import extend_gates

def main():
    assert not (O/"ADMISSION.json").exists()
    setup()
    assert js(P/"AUTONOMOUS_EXIT_ACCEPTANCE.json")["status"]=="PASS"
    old=js("/data/mlr_ifa/experiments/ZRv3_OBJECT_RA_seed41_20261020_v3/CONTRACT.json")
    for p,h in old["files"].items():assert sha(p)==h,p
    c=cfg("eval");b,_,shape=data_setup(c,False);c.shape_meta=shape
    task=b.get_task(6);init=Path(c.init_states_folder)/task.problem_folder/task.init_states_file
    states=torch.load(init,map_location="cpu")
    assert len(states)>=40
    hs=[hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in states[:40]]
    assert len(set(hs))==40,"Initial state partitions overlap by payload"
    p=complete(c,shape).eval().requires_grad_(False)
    tokens=[b.get_task_emb(t) for t in range(7)]
    def encode(ts):
        ids=torch.stack([tokens[t][0] for t in ts]).cuda();mask=torch.stack([tokens[t][1] for t in ts]).cuda()
        return p.language_encoder(ids,mask).detach().cpu().numpy()
    with torch.no_grad():
        canonical=np.concatenate([encode([t]) for t in range(7)])
        gate,geometry=extend_gates(canonical[:6],canonical[6],np.empty((0,512)))
        np.savez(W/"language.npz",canonical=canonical,gate=gate[0])
        probe=make_bank(torch.empty(512,0,dtype=torch.float64))
        checks=[]
        for t in range(7):
            for batch in (1,10,32):
                e=torch.from_numpy(encode([t]*batch)).cuda()
                hit=probe.route(e)
                assert bool(hit.all()) if t==6 else not bool(hit.any())
                checks.append(dict(task=t,batch=batch,recognized_task6=bool(hit.all())))
        files=sorted((P/"features/6").glob("*.pt"));assert len(files)==1562
        for f in files:
            x=torch.load(f,map_location="cpu")
            assert probe.route(x["feature"][0,:,0,:].cuda()).all()
    fm=js(P/"FEATURES.json")
    inputs={str(CP):sha(CP),str(init):sha(init),str(W/"language.npz"):sha(W/"language.npz")}
    for f in ("/data/mlr_ifa/experiments/ZRv3_OBJECT_RA_seed41_20261020_v3/ADMISSION.json",
              "/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019/seed41/env.sh",
              "/data/mlr_ifa/activate.sh"):
        inputs[f]=sha(f)
    for f in files:
        rel=f.relative_to(P/"features").as_posix();assert sha(f)==fm["hashes"][rel]
        inputs[str(f)]=sha(f)
    inputs.update({p:h for p,h in old["files"].items() if str(P) in p or str(S) in p})
    write(O/"INPUT_SHA256.json",inputs)
    write(O/"BINDING.json",dict(parent=str(P),checkpoint=str(CP),checkpoint_sha256=sha(CP),
        initial_file=str(init),initial_sha256=sha(init),initial_hashes=hs,available_initials=len(states),
        partitions=dict(formal=list(range(20)),anchor=list(range(20,30)),validation=list(range(30,40))),
        modal_sha256=sha(P/"MODAL_CONTRACT.json"),features=1562,task_language=task.language))
    write(O/"ADMISSION.json",dict(status="PASS",gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        geometry=geometry,route_checks=checks,environment_steps=0,optimizer_calls=0,
        independent_payload_hashes=True,initial_distribution_limitation="Disjoint released initial states; no claim of IID statistical draws"))
    print("ADMISSION PASS: 40 disjoint initial payloads, real canonical/cache routing, 0 environment")
if __name__=="__main__":main()

"""Offline GOAL-to-PiPER base adaptation. All observations come from archived files."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import random
import time
from decimal import Decimal

import cv2
import numpy as np
import torch
from torch import nn
from tokenizers import Tokenizer
from transformers import CLIPVisionConfig, CLIPVisionModel, CLIPTextConfig, CLIPTextModel, GPT2Config, GPT2Model

GOAL_SHA = "241d3370acfc04d694df04256956a97aa13cfdc2e651cf258d338f8e72988be7"
TOKEN_SHA = "a83e0809aa4c3af7208b2df632a7a69668c6d48775b3c3fe4e1b1199d1f8b8f4"
SETS = [
    ("B1_top50_20261120_163349", "10fc15aff94bc6deb746c4f838ad04fef8156cddaea153b49301376aec3c38cb",
     "pick up the wooden block and place it in the blue dish on the left"),
    ("B2_top50_20261120_173919", "481469382a4725d659874b33575bf9a254deff91b7292c8509431a06271d7dc7",
     "pick up the wooden block and place it in the light-colored dish on the right"),
    ("B3_top50_20261120_211843", "82e47f0fa65545ac8c3500a1560aa2d08233e25fe99f526edec71dbb156e9fe6",
     "push the wooden block into the yellow area on the left"),
]
SEED, HISTORY = 42, 8


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4<<20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def save_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False)+"\n")
    temp.replace(path)


def save_pt(path, value):
    temp = path.with_suffix(".partial")
    torch.save(value, temp)
    temp.replace(path)


def read_episodes(base):
    episodes, split = [], {}
    for task, (set_name, sha, _) in enumerate(SETS):
        folder = base/"expert_sets"/set_name
        assert digest(folder/"manifest.json") == sha
        manifest = json.loads((folder/"manifest.json").read_text())
        entries = sorted(manifest["episodes"], key=lambda e: e["episode_id"])
        assert manifest["task"] == f"B{task+1}" and len(entries) == 50
        val = set(random.Random(SEED+task).sample([e["episode_id"] for e in entries], 10))
        for e in entries:
            name = e["episode_id"]
            ep = (folder/e["episode_dir"]).resolve(strict=True)
            assert ep == (base/"recordings"/name).resolve(strict=True) and name not in split
            raw = (ep/"alignment.jsonl").read_bytes()
            assert hashlib.sha256(raw).hexdigest() == e["alignment_sha256"]
            rows = [json.loads(s) for s in raw.splitlines()]
            start, last = e["first_action_row"], e["last_action_row"]
            assert last+1 == len(rows) == e["recorded_pairs"] and last-start+1 == e["action_count"]
            assert all(r["target"] is None for r in rows[:start])
            stamps = np.array([r["sample_monotonic_ns"] for r in rows], dtype=np.int64)
            cam = []
            for role in ("overhead","wrist"):
                files = sorted((p for p in (ep/"episode0/camera/color"/role).iterdir()
                                if p.stem.replace(".", "",1).isdigit()), key=lambda p: Decimal(p.stem))
                assert len(files) == len(rows)
                assert all(abs(int(Decimal(p.stem)*10**9)-int(t)) <= 500 for p,t in zip(files, stamps))
                cam.append(files[start:last+1])
            assert [p.stem for p in cam[0]] == [p.stem for p in cam[1]]
            valid = rows[start:last+1]
            values = [np.asarray([r[k]["joints_rad"]+[r[k]["gripper_mm"]] for r in valid], np.float32)
                      for k in ("feedback","target")]
            assert all(a.shape==(len(valid),7) and np.isfinite(a).all() for a in values)
            split[name] = "val" if name in val else "train"
            episodes.append(dict(id=name, task=task, split=split[name], cameras=cam, q=values[0], a=values[1]))
    assert len(split)==150 and sum(len(e["q"]) for e in episodes)==43824
    return episodes, split


def fit_stats(episodes):
    train = [e for e in episodes if e["split"]=="train"]
    q = np.concatenate([e["q"] for e in train])
    residual = np.concatenate([e["a"]-e["q"] for e in train])
    return dict(q_mean=q.mean(0,dtype=np.float64).tolist(), q_std=q.std(0,dtype=np.float64).clip(1e-5).tolist(),
                delta_mean=residual.mean(0,dtype=np.float64).tolist(),
                delta_std=residual.std(0,dtype=np.float64).clip(1e-5).tolist())


def clip_load(model, state, prefix, scale):
    original = {k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
    mapped = {}
    for k in model.state_dict():
        if k in original:
            mapped[k] = original[k]
        else:
            stem, kind = k.rsplit(".",1)
            mapped[k] = original[stem+".base_layer."+kind]
            if kind=="weight":
                a,b=(original[stem+f".lora_{m}.default.weight"] for m in ("A","B"))
                assert a.shape[0]==b.shape[1]==8
                mapped[k] = mapped[k]+scale*(b@a)
    model.load_state_dict(mapped,strict=True)


def official_load(module, state, prefix):
    module.load_state_dict({k:state[prefix+k] for k in module.state_dict()},strict=True)


def film():
    return nn.Sequential(nn.Linear(512,512),nn.GELU(),nn.Linear(512,512))


class Frontend(nn.Module):
    def __init__(self,state,scale,tokenizer):
        super().__init__()
        self.vision=CLIPVisionModel(CLIPVisionConfig(patch_size=16))
        text=CLIPTextModel(CLIPTextConfig())
        clip_load(self.vision,state,"image_encoders.base_model.model.clip.",scale)
        clip_load(text,state,"language_encoder.base_model.model.clip.",scale)
        self.proj=nn.Linear(768,512)
        self.scale,self.shift=film(),film()
        for module,prefix in ((self.proj,"image_proj."),(self.scale,"film_scale."),(self.shift,"film_shift.")):
            official_load(module,state,prefix)
        texts=[r[2] for r in SETS]
        text.eval()
        with torch.no_grad():
            pooled=[]
            for phrase in texts:
                ids=torch.tensor(tokenizer.encode(phrase).ids)[None]
                assert ids[0,0]==49406 and ids[0,-1]==49407
                pooled.append(text(input_ids=ids,attention_mask=torch.ones_like(ids)).pooler_output[0])
            self.register_buffer("language",torch.stack(pooled))
        self.register_buffer("mean",torch.tensor([.48145466,.4578275,.40821073])[None,:,None,None])
        self.register_buffer("std",torch.tensor([.26862954,.26130258,.27577711])[None,:,None,None])
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def encode(self, cams, indices, task):
        images=[]
        for i in indices:
            for view in cams:
                bgr=cv2.imread(str(view[i]),cv2.IMREAD_COLOR)
                if bgr is None or bgr.shape!=(480,640,3): raise ValueError(str(view[i]))
                # Keep the full workspace, including peripheral start points and dishes.
                rgb=cv2.cvtColor(cv2.resize(bgr,(224,168),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
                images.append(cv2.copyMakeBorder(rgb,28,28,0,0,cv2.BORDER_CONSTANT,value=(123,117,104)))
        pixels=torch.from_numpy(np.stack(images)).to(self.mean.device).permute(0,3,1,2).float()/255
        with torch.autocast("cuda",dtype=torch.bfloat16):
            v=self.vision(pixel_values=(pixels-self.mean)/self.std).pooler_output
            lang=self.language[task]
            v=self.proj(v)*(1+self.scale(lang))+self.shift(lang)
        assert torch.isfinite(v).all()
        return v.reshape(len(indices),2,512).half().cpu()


class Policy(nn.Module):
    def __init__(self,state,langs,stats):
        super().__init__()
        self.register_buffer("language",langs.clone())
        for k,v in stats.items(): self.register_buffer(k,torch.tensor(v,dtype=torch.float32))
        self.state_encoder=nn.Sequential(nn.Linear(7,512),nn.ReLU(),nn.Linear(512,512))
        official_load(self.state_encoder[2],state,"state_encoder.2.")
        self.scale,self.shift=film(),film()
        for module,name in ((self.scale,"state_scale."),(self.shift,"state_shift.")):
            official_load(module,state,name)
            module.requires_grad_(False)
        self.temporal=GPT2Model(GPT2Config(n_embd=512,n_layer=6,n_head=8,n_positions=32,
                                           vocab_size=1,resid_pdrop=.15,embd_pdrop=.15,attn_pdrop=.15,use_cache=False))
        official_load(self.temporal,state,"temporal_transformer.decoder.")
        self.head=nn.Sequential(nn.Linear(512,512),nn.ReLU(),nn.Linear(512,5*(7+7+1)))

    def forward(self, vision, q, task):
        lang=self.language[task]
        st=self.state_encoder((q-self.q_mean)/self.q_std)
        st=st*(1+self.scale(lang)[:,None])+self.shift(lang)[:,None]
        tokens=torch.cat([lang[:,None,None].expand(-1,q.size(1),-1,-1),vision,st[:,:,None]],dim=2)
        # The last causal token can attend to both current images and the current state.
        x=self.temporal(inputs_embeds=tokens.flatten(1,2)).last_hidden_state[:,-1]
        out=self.head(x).float().reshape(-1,5,15)
        means, raw_std, logits=out[:,:,:7],out[:,:,7:14],out[:,:,14]
        dist=torch.distributions
        mix=dist.MixtureSameFamily(dist.Categorical(logits=logits),
                    dist.Independent(dist.Normal(means,nn.functional.softplus(raw_std)+.02),1))
        mode=means[torch.arange(len(q),device=q.device),logits.argmax(-1)]
        absolute=q[:,-1]+self.delta_mean+self.delta_std*mode
        return mix,absolute


class Frames:
    def __init__(self, episodes, features):
        starts, tags = [], []
        offset=0
        for e in episodes:
            n=len(e["a"])
            starts.append(torch.full((n,),offset))
            tags.append(torch.full((n,),e["task"]))
            offset+=n
        self.q=torch.from_numpy(np.concatenate([e["q"] for e in episodes])).cuda()
        self.a=torch.from_numpy(np.concatenate([e["a"] for e in episodes])).cuda()
        self.v=torch.cat([features[e["id"]] for e in episodes]).cuda()
        self.start=torch.cat(starts).cuda()
        self.tag=torch.cat(tags).cuda()
        self.lag=torch.arange(1-HISTORY,1,device="cuda")

    def batch(self,idx):
        rows=torch.maximum(idx[:,None]+self.lag,self.start[idx,None])
        return self.v[rows],self.q[rows],self.tag[idx],self.a[idx]


def compute(policy,batch):
    v,q,t,a=batch
    with torch.autocast("cuda",dtype=torch.bfloat16):
        mix,pred=policy(v,q,t)
    label=(a-q[:,-1]-policy.delta_mean)/policy.delta_std
    nll=-mix.log_prob(label)
    if not torch.isfinite(nll).all() or not torch.isfinite(pred).all():
        raise ValueError("Nonfinite action prediction or loss")
    return nll,pred


def verify_tasks(front,policy,episodes):
    import copy
    probe=copy.deepcopy(policy).cuda().train()
    optim=torch.optim.AdamW(probe.parameters(), lr=.0001)
    evidence=[]
    for task in range(3):
        items=[e for e in episodes if e["task"]==task and e["split"]=="train"][:2]
        batches=[]
        for e in items:
            start=len(e["q"])//2
            ids=list(range(start,start+HISTORY))
            batches.append((front.encode(e["cameras"],ids,task),
                            torch.from_numpy(e["q"][ids]),torch.tensor(task),torch.from_numpy(e["a"][ids[-1]])))
        batch=[torch.stack(x).cuda() for x in zip(*batches)]
        optim.zero_grad(set_to_none=True)
        before=probe.head[-1].weight.detach().clone()
        nll,_=compute(probe,batch)
        nll.mean().backward()
        grad=torch.nn.utils.clip_grad_norm_(probe.parameters(),1.0,error_if_nonfinite=True)
        for part in (probe.head[-1].weight,probe.state_encoder[0].weight,probe.temporal.wpe.weight):
            assert part.grad is not None and part.grad.abs().max().item()>0
        optim.step()
        change=(before-probe.head[-1].weight).abs().max().item()
        assert np.isfinite(change) and change>0
        evidence.append(dict(task=f"B{task+1}",nll=nll.mean().item(),gradient_norm=grad.item(),head_update=change,
                             examples=[e["id"] for e in items],history=HISTORY,views=2))
    return evidence


def get_features(front,episodes,folder,identity):
    folder.mkdir(exist_ok=True)
    result={}
    for i,e in enumerate(episodes):
        path=folder/(e["id"]+".pt")
        if path.exists():
            record=torch.load(path,map_location="cpu",weights_only=True)
            if record["identity"]!=identity: raise ValueError("Cache provenance mismatch")
            v=record["vision"]
        else:
            n=len(e["q"])
            v=torch.cat([front.encode(e["cameras"],range(j,min(j+32,n)),e["task"]) for j in range(0,n,32)])
            save_pt(path,dict(identity=identity,vision=v))
        assert v.shape==(len(e["q"]),2,512) and torch.isfinite(v).all()
        result[e["id"]]=v
        if (i+1)%10==0 or i==0: print("encoded",i+1,"of",len(episodes),flush=True)
    return result


@torch.no_grad()
def evaluate(policy,frames,batch_size):
    policy.eval()
    totals=np.zeros((3,17),np.float64)
    for idx in torch.arange(len(frames.a),device="cuda").split(batch_size):
        batch=frames.batch(idx)
        nll,pred=compute(policy,batch)
        _,q,t,a=batch
        err=(pred-a).abs()
        hold=(q[:,-1]-a).abs()
        for task in range(3):
            mask=t==task
            if mask.any():
                totals[task]+=torch.cat([mask.sum()[None],nll[mask].sum()[None],
                         (err[mask]/policy.delta_std).mean(-1).sum()[None],
                         err[mask].sum(0),hold[mask].sum(0)]).double().cpu().numpy()
    metrics={}
    for task,line in enumerate(totals):
        n=line[0]
        assert n>0
        x=line/n
        metrics[f"B{task+1}"]=dict(rows=int(n),nll=float(x[1]),normalized_mae=float(x[2]),
                 joint_mae_deg=float(np.rad2deg(x[3:9]).mean()),gripper_mae_mm=float(x[9]),
                 hold_joint_mae_deg=float(np.rad2deg(x[10:16]).mean()),hold_gripper_mae_mm=float(x[16]))
    return metrics, float(np.mean([v["normalized_mae"] for v in metrics.values()]))


def train(policy,training,validation,out,run_config):
    epochs,batch_size=run_config["epochs"],run_config["batch_size"]
    optimizer=torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad),lr=1e-4,weight_decay=.01)
    n=len(training.a)
    weights=n/(3*torch.bincount(training.tag,minlength=3).float())
    history=[]
    best_epoch,best=0,float("inf")
    latest=out/"last.pt"
    if latest.is_file():
        old=torch.load(latest,map_location="cpu",weights_only=True)
        assert old["config"]==run_config
        policy.load_state_dict(old["policy"]); optimizer.load_state_dict(old["optimizer"])
        history,best_epoch,best=old["history"],old["best_epoch"],old["best"]
        if best_epoch==len(history) and history: save_pt(out/"best.pt",dict(config=run_config,epoch=best_epoch,policy=old["policy"]))
        del old
    for epoch in range(len(history)+1,epochs+1):
        if epoch-1-best_epoch>=run_config["patience"]: break
        policy.train()
        torch.manual_seed(SEED+epoch)
        order=torch.randperm(n,device="cuda")
        total_loss=0.0
        for step,idx in enumerate(order.split(batch_size),1):
            batch=training.batch(idx)
            optimizer.zero_grad(set_to_none=True)
            nll,_=compute(policy,batch)
            loss=(nll*weights[batch[2]]).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step()
            total_loss+=loss.detach().item()*len(idx)
            if epoch==1 and (step==1 or step%100==0): print("formal step",step,"nll",loss.item(),flush=True)
        metrics,score=evaluate(policy,validation,batch_size)
        improved=score<best
        if improved: best_epoch,best=epoch,score
        history.append(dict(epoch=epoch,train_nll=total_loss/n,val=metrics,macro_normalized_mae=score))
        save_pt(latest,dict(config=run_config,history=history,best_epoch=best_epoch,best=best,
                            policy=policy.state_dict(),optimizer=optimizer.state_dict()))
        if improved: save_pt(out/"best.pt",dict(config=run_config,epoch=epoch,policy=policy.state_dict()))
        save_json(out/"history.json",history)
        print("epoch",epoch,"score",score,"best",best_epoch,flush=True)
    checkpoint=torch.load(out/"best.pt",map_location="cpu",weights_only=True)
    assert checkpoint["epoch"]==best_epoch
    policy.load_state_dict(checkpoint["policy"])
    metrics,replayed=evaluate(policy,validation,batch_size)
    assert abs(replayed-best)<1e-5
    save_json(out/"finished.json",dict(epochs=len(history),best_epoch=best_epoch,best_score=replayed,
              metrics=metrics,best_sha256=digest(out/"best.pt"),completed_at=time.time()))
    print("complete",len(history),"best",best_epoch,flush=True)


class EasyDict(dict):
    """Only the legacy checkpoint configuration uses this mapping type."""


def run(base,out):
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    weight=out/"goal_multitask_model_ep96.pth"
    tokenizer_path=out/"clip_tokenizer_goal.json"
    assert digest(weight)==GOAL_SHA and digest(tokenizer_path)==TOKEN_SHA
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    with torch.serialization.safe_globals([(EasyDict,"easydict.EasyDict")]):
        checkpoint=torch.load(weight,map_location="cpu",weights_only=True)
    state={k.removeprefix("module."):v for k,v in checkpoint["state_dict"].items()}
    cfg=checkpoint["cfg"]["policy"]
    scale=cfg["alpha"]/cfg["rank"]
    assert cfg["rank"]==8
    episodes,split=read_episodes(base)
    stats=fit_stats(episodes)
    save_json(out/"split.json",split); save_json(out/"normalization.json",stats)
    config=dict(seed=SEED,history=HISTORY,epochs=30,patience=6,batch_size=128,learning_rate=1e-4,
                checkpoint_sha256=GOAL_SHA,tokenizer_sha256=TOKEN_SHA,script_sha256=digest(Path(__file__)),
                manifests=[r[1] for r in SETS],split_sha256=digest(out/"split.json"),normalization=stats,
                language=[r[2] for r in SETS],lora_scale=scale,device=torch.cuda.get_device_name(),
                observations="overhead,wrist,feedback joints (radians),gripper (mm)",
                action="absolute recorded joint and gripper target via standardized target minus feedback",
                readout="last state token after both current camera tokens")
    save_json(out/"config.json",config)
    front=Frontend(state,scale,tokenizer).cuda()
    policy=Policy(state,front.language.detach().cpu(),stats).cuda()
    del state,checkpoint
    smoke=verify_tasks(front,policy,episodes)
    save_json(out/"smoke.json",smoke)
    print("all three smoke batches passed",flush=True)
    features=get_features(front,episodes,out/"visual_cache",config["script_sha256"]+GOAL_SHA+TOKEN_SHA)
    del front
    trainset,valset=([e for e in episodes if e["split"]==s] for s in ("train","val"))
    training,validation=Frames(trainset,features),Frames(valset,features)
    del features
    print("formal start",len(trainset),len(valset),len(training.a),len(validation.a),flush=True)
    train(policy,training,validation,out,config)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root",type=Path)
    parser.add_argument("run_dir",type=Path)
    args=parser.parse_args()
    with (args.run_dir/"training.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.data_root.resolve(strict=True),args.run_dir.resolve(strict=True))

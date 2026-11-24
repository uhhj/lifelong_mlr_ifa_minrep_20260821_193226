"""Run independent B1 image interventions using the archived trainer and original split."""
import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
from types import MethodType

import cv2
import numpy as np
import torch
from torch import nn


MANIFEST = "130b51e723fe9839ed6986f0d9bcc7b8035e49ffe103cd8847b837607e1227b9"


def image_rgb(image, size):
    assert image.shape == (480,640,3)
    padded = cv2.copyMakeBorder(image,80,80,0,0,cv2.BORDER_CONSTANT,value=0)
    return cv2.cvtColor(cv2.resize(padded,(size,size),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB).transpose(2,0,1)


def high_cache(work, base, out):
    dest = out / "head_256.npy"
    meta = json.loads((base/"ode_cache/metadata.json").read_bytes())
    if dest.exists():
        assert np.load(dest,mmap_mode="r").shape == (meta["rows"],3,256,256)
        return dest
    ranks = {r["id"]:r for r in json.loads((base/"selection.json").read_bytes())["ranking"]}
    old = np.load(base/"ode_cache/head_camera.npy",mmap_mode="r")
    tmp = out/"head_256.incomplete.npy"
    store = np.lib.format.open_memmap(tmp,mode="w+",dtype=np.uint8,shape=(len(old),3,256,256))
    for ep in meta["episodes"]:
        r=ranks[ep["id"]]
        folder=work/"batches/B1_20261123"/r["source"]/"episode0/camera/color/overhead"
        fs=sorted(folder.glob("*.jpg"),key=lambda f:Decimal(f.stem))[r["first"]:]
        assert len(fs)==ep["stop"]-ep["start"]
        for k,f in enumerate(fs,ep["start"]):
            image=cv2.imread(str(f))
            assert np.array_equal(image_rgb(image,128),old[k])
            store[k]=image_rgb(image,256)
    store.flush()
    del store
    tmp.replace(dest)
    return dest


class WindowCrop(nn.Module):
    def __init__(self, factor, shared):
        super().__init__()
        self.factor,self.shared=factor,shared

    def forward(self, images):
        n,_,h,w=images.shape
        f=self.factor
        assert (h,w)==(128*f,128*f)
        size=112*f
        if not self.training:
            return images[:,:,8*f:120*f,8*f:120*f]
        # Draw the same number of random values, on the same device as the original trainer.
        # This preserves its RNG sequence for both arms. Share the first draw only within a view/window.
        y,x=[(torch.rand(n,1)*16).long().to(images.device).flatten()*f for _ in range(2)]
        if self.shared:
            assert n%8==0
            y,x=[v.reshape(-1,8)[:,:1].expand(-1,8).reshape(-1) for v in (y,x)]
        grid=torch.arange(size,device=images.device)
        pixel=(y[:,None,None]+grid[None,:,None])*w+x[:,None,None]+grid[None,None,:]
        idx=pixel.reshape(n,1,-1).expand(-1,3,-1)
        return torch.gather(images.flatten(2),2,idx).reshape(n,3,size,size)


def build_variant(builder, stats, cfg, variant):
    model=builder(stats,cfg)
    enc=model.obs_encoder
    for role in ("overhead","wrist"):
        f=2 if role=="overhead" and variant=="high" else 1
        enc.key_shape_map[role]=(3,128*f,128*f)
        enc.key_transform_map[role][1]=WindowCrop(f,variant=="shared")
    return model


def checkpoint_resnet(self,x):
    # GroupNorm is per image. Recomputing deterministic residual blocks changes only activation storage.
    from torch.utils.checkpoint import checkpoint
    if not self.training or not torch.is_grad_enabled():
        return self._forward_impl(x)
    x=self.maxpool(self.relu(self.bn1(self.conv1(x))))
    for layer in (self.layer1,self.layer2,self.layer3,self.layer4):
        x=checkpoint(layer,x,use_reentrant=False,preserve_rng_state=False)
    return self.fc(torch.flatten(self.avgpool(x),1))


def train(work,out,variant):
    base=work/"offline_diagnostics/b1_ode_lerobot_20261123"
    assert hashlib.sha256((base/"selection.json").read_bytes()).hexdigest()==MANIFEST
    sys.path.insert(0,str(base/"code"))
    import train_task as trainer
    original_builder=trainer.build_policy
    original_frames=trainer.TrainingFrames
    cache=base/"ode_cache"
    high=high_cache(work,base,out) if variant=="high" else None
    class Frames(original_frames):
        def __init__(self,path,task):
            super().__init__(path,task)
            # Keep camera arrays on the host to share the GPU with other local jobs.
            del self.x["head_camera"],self.x["wrist_camera"]
            torch.cuda.empty_cache()
            self.views={key:torch.from_numpy(np.load(file,mmap_mode="c")) for key,file in
                        (("overhead",high or path/"head_camera.npy"),("wrist",path/"wrist_camera.npy"))}

        def batch(self,idx):
            hist=(idx[:,None]+self.history).maximum(self.start[idx,None])
            fut=(idx[:,None]+self.future).minimum(self.end[idx,None])
            host=hist.cpu()
            obs={"agent_pos":self.x["state"][hist]}
            for role,images in self.views.items():
                obs[role]=images[host].cuda().float().div_(255)
            return {"obs":obs,"action":torch.cat([self.x["action"][hist],self.x["command"][fut]],1)}

    def build(stats,cfg):
        cfg["visual_intervention"]=variant
        cfg["overhead_input_px"]=256 if variant=="high" else 128
        cfg["temporal_crop_shared"]=variant=="shared"
        model=build_variant(original_builder,stats,cfg,variant)
        if os.environ.get("B1_CHECKPOINT_ACTIVATIONS")=="1":
            for net in model.obs_encoder.key_model_map.values():
                assert hasattr(net,"_forward_impl")
                net.forward=MethodType(checkpoint_resnet,net)
        return model
    trainer.TrainingFrames=Frames
    trainer.build_policy=build
    trainer.main(cache,out/variant,"B1")


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("work",type=Path)
    ap.add_argument("out",type=Path)
    ap.add_argument("variant",choices=["shared","high"])
    args=ap.parse_args()
    cv2.setNumThreads(1)
    try:
        train(args.work,args.out,args.variant)
    except torch.OutOfMemoryError:
        if os.environ.get("B1_CHECKPOINT_ACTIVATIONS")=="1":
            raise
        print("Retry with activation recomputation after GPU allocation failure",flush=True)
        os.environ["B1_CHECKPOINT_ACTIVATIONS"]="1"
        os.execv(sys.executable,[sys.executable,"-Bu",*sys.argv])

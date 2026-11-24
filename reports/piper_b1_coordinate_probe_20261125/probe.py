"""Extract frozen, overhead-only features from saved B1 images into a private working directory."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def load_model(work, variant):
    diag=work/"offline_diagnostics"
    prior=diag/"b1_vision_contrasts_20261124"
    base=diag/"b1_ode_lerobot_20261123"
    sys.path[:0]=[str(prior),str(base/"code")]
    from ode_policy_adapter import build_policy
    from train_contrast import build_variant
    path=(base/"run" if variant=="baseline" else prior/variant)/"epoch_50.pt"
    sha=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):
            sha.update(block)
    identity=sha.hexdigest()
    assert identity==json.loads((prior/"final.json").read_bytes())["checkpoint_ids"][variant]
    saved=torch.load(path,map_location="cpu",weights_only=True)
    model=(build_policy(saved["stats"],saved["config"]) if variant=="baseline"
           else build_variant(build_policy,saved["stats"],saved["config"],variant))
    model.load_state_dict(saved["ema"],strict=True)
    return model.eval(),identity


@torch.inference_mode()
def extract(work,out):
    sys.path.insert(0,str(work/"offline_diagnostics/b1_vision_contrasts_20261124"))
    from train_contrast import image_rgb
    torch.set_num_threads(2)
    rows=json.loads((out/"frames.json").read_bytes())
    keys=[r["key"] for r in rows]
    with zipfile.ZipFile(out/"frames.zip") as archive:
        imgs=[cv2.imdecode(np.frombuffer(archive.read(k+".jpg"),np.uint8),cv2.IMREAD_COLOR) for k in keys]
    for variant in ["baseline","high","shared"]:
        model,identity=load_model(work,variant)
        enc=model.obs_encoder
        net=enc.key_model_map["overhead"]
        size=256 if variant=="high" else 128
        values={n:[] for n in ["pixels","layer2_map","layer4_map","pooled"]}
        for k in range(0,len(imgs),12):
            arr=torch.from_numpy(np.stack([image_rgb(im,size) for im in imgs[k:k+12]])).float()/255
            x=enc.key_transform_map["overhead"](arr)
            values["pixels"].append(F.adaptive_avg_pool2d(x,(24,24)).flatten(1).numpy())
            stage2=net.layer2(net.layer1(net.maxpool(net.relu(net.bn1(net.conv1(x))))))
            values["layer2_map"].append(F.adaptive_avg_pool2d(stage2,(8,8)).flatten(1).numpy())
            stage4=net.layer4(net.layer3(stage2))
            pooled=net.fc(net.avgpool(stage4).flatten(1))
            torch.testing.assert_close(pooled,net(x),rtol=1e-5,atol=1e-6)
            values["layer4_map"].append(stage4.flatten(1).numpy())
            values["pooled"].append(pooled.numpy())
        np.savez(out/(variant+"_features.npz"),keys=keys,**{k:np.concatenate(v) for k,v in values.items()})
        print(variant,identity,{k:np.concatenate(v).shape for k,v in values.items()},flush=True)


if __name__=="__main__":
    extract(*map(Path,sys.argv[1:3]))

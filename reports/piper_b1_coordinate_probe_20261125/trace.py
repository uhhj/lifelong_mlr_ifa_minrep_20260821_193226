"""Trace real held-out history substitutions through the frozen policy in memory."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from probe import load_model


def encode_view(encoder,view,pixels):
    x=encoder.key_transform_map[view](pixels[0])
    net=encoder.key_model_map[view]
    l2=net.layer2(net.layer1(net.maxpool(net.relu(net.bn1(net.conv1(x))))))
    l4=net.layer4(net.layer3(l2))
    pooled=net.fc(net.avgpool(l4).flatten(1))
    return dict(input=x,layer2=l2,layer4=l4,pooled=pooled)


def norm(t):
    return torch.linalg.vector_norm(t.double()).item()


def measure(row,name,new,old):
    dist=norm(new-old)
    row[name+'_abs_l2']=dist
    row[name+'_rel_pct']=100*dist/max(1e-20,norm(old))


@torch.inference_mode()
def main(work,out):
    prior=work/'offline_diagnostics/b1_vision_contrasts_20261124'
    sys.path.insert(0,str(prior))
    from annotate import images
    from assess import geometry
    from train_contrast import image_rgb
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    meta,records,_,xy=geometry(work,prior)
    fixed=json.loads((prior/'final.json').read_bytes())
    assert meta['pairs']==fixed['geometry']['pairs']
    results={}
    for variant in ('baseline','high','shared'):
        model,_=load_model(work,variant)
        encoder=model.obs_encoder
        assert list(encoder.rgb_keys)==['overhead','wrist']
        hist={}
        for r in records:
            t=r['close']-18
            q=torch.from_numpy(r['q'][None,t-7:t+1])
            obs={'agent_pos':q}
            encoded={}
            for view in encoder.rgb_keys:
                side=256 if variant=='high' and view=='overhead' else 128
                paths=images(r['folder'],view)[r['first']+t-7:r['first']+t+1]
                pixels=np.stack([image_rgb(cv2.imread(str(f)),side) for f in paths])
                obs[view]=torch.from_numpy(pixels[None]).float()/255
                encoded[view]=encode_view(encoder,view,obs[view])
            hist[r['episode']]=(obs,encoded,torch.from_numpy(r['cmd'][None,t-1]))
        def conditions(views,q):
            flat=torch.cat([views['overhead']['pooled'],views['wrist']['pooled'],
                            torch.zeros_like(q[0])],dim=1).reshape(1,-1)
            linear=model.cond_net[0](flat)
            return linear,model.cond_net[2](model.cond_net[1](linear))
        def endpoint(obs,anchor):
            # Use the original action API and the original offline deterministic seed mode.
            traj=model.predict_action(obs,a0=anchor,eps=torch.zeros_like(anchor))['action'][0]
            return xy(traj[-1].numpy())
        rows=[]
        for a,b in meta['pairs']:
            for rec,donor in ((a,b),(b,a)):
                obs,enc,anchor=hist[rec]
                other,alt,_=hist[donor]
                lin,cond=conditions(enc,obs['agent_pos'])
                actual,feedback=model._encode(obs)
                torch.testing.assert_close(cond,actual,rtol=2e-5,atol=2e-5)
                orig=endpoint(obs,anchor)
                for view in encoder.rgb_keys:
                    row=dict(pair=[a,b],recipient=rec,view=view)
                    for stage in enc[view]:
                        measure(row,stage,alt[view][stage],enc[view][stage])
                    mixed=dict(enc)
                    mixed[view]=alt[view]
                    swapped=dict(obs)
                    swapped[view]=other[view]
                    new_lin,new_cond=conditions(mixed,obs['agent_pos'])
                    actual,actual_fb=model._encode(swapped)
                    torch.testing.assert_close(new_cond,actual,rtol=2e-5,atol=2e-5)
                    torch.testing.assert_close(feedback,actual_fb,rtol=0,atol=0)
                    measure(row,'linear',new_lin,lin)
                    measure(row,'condition',new_cond,cond)
                    row['cond_per_pooled']=row['condition_abs_l2']/max(1e-20,row['pooled_abs_l2'])
                    row['endpoint_displacement_mm']=float(np.linalg.norm(endpoint(swapped,anchor)-orig))
                    rows.append(row)
        w=model.cond_net[0].weight.detach().reshape(512,8,1031)
        results[variant]=dict(rows=rows,linear_weight_frobenius={v:norm(w[:,:,s]) for v,s in
            [('overhead',slice(0,512)),('wrist',slice(512,1024))]})
        print(variant,'trace complete',flush=True)
    (out/'trace_results.json').write_text(json.dumps(results,indent=1,allow_nan=False)+'\n')


if __name__=='__main__':
    main(*map(Path,sys.argv[1:3]))

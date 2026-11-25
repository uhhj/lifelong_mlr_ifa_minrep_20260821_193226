"""Model backend for the installed PiPER console's B1 ODE profile."""
from collections import deque
import hashlib
import importlib.util
import json

import cv2
import numpy as np
import torch

from catalog import WORK, digest
from ode_live import B1Predictor, prepare_views as ode_pixels


def manifest_id(bundle):
    data=json.dumps(bundle,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
    return hashlib.sha256(data).hexdigest()


class GroundedODE(B1Predictor):
    """Keep the installed ODE history, command acknowledgement and chunk scheduling."""
    @staticmethod
    def prepare_views(images):
        if len(images)!=2 or any(f.shape!=(480,640,3) or f.dtype!=np.uint8 for f in images):
            raise ValueError('B1 纠偏需要两张 640×480 BGR 图像')
        # The console's existing full-resolution path archives these exact inputs losslessly.
        return np.stack([cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) for frame in images])

    def __init__(self,profile):
        bundle=profile['alignment']
        if manifest_id(bundle)!=profile['sha256'] or profile['task']!='B1':
            raise ValueError('B1 纠偏版本校验失败')
        for name,expected in bundle['files'].items():
            if digest(WORK/name)!=expected:
                raise ValueError('B1 纠偏组件校验失败：'+name)
        from pathlib import Path
        if digest(Path(__file__))!=bundle['backend_sha256']:
            raise ValueError('B1 推理入口校验失败')
        super().__init__(dict(profile,sha256=bundle['base_sha256']))
        script=WORK/bundle['algorithm']
        spec=importlib.util.spec_from_file_location('piper_grounded_b1',script)
        alg=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(alg)
        self.aligner=alg.B1OfflineAlignment(WORK,WORK/bundle['weights'])
        if self.aligner.steady is None:
            raise ValueError('缺少已验收的末段纠偏参数')
        self.raw=deque(maxlen=8)
        self.heat=deque(maxlen=8)
        self.features=deque(maxlen=8)
        self.wrist=None
        self.closed=False
        self.details=None
        self.base_plan=None
        self.close_mm=bundle['close_mm']
        self.radius=bundle['radius_mm']
        # Initialize the heatmap before the station opens its camera streams.
        self.aligner.det.net.cuda().eval()
        self.aligner.det.device='cuda'
        self.aligner.det.estimate([np.zeros((480,640,3),np.uint8)]*8)
        self.capture_ode()

    @torch.inference_mode()
    def capture_ode(self):
        # Replay the identical trained ODE operations; only inputs change between queries.
        example={v:torch.zeros((1,8,3,128,128),device='cuda') for v in ('overhead','wrist')}
        example['agent_pos']=torch.zeros((1,8,7),device='cuda')
        self.condition,self.field_state=self.model._encode(example)
        self.initial=torch.zeros((1,7),device='cuda')
        self.action_scale=self.model.normalizer['action']
        def integrate():
            return self.model.integrate(self.initial,self.field_state,self.condition,self.model.n_action_steps)
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):integrate()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output=integrate()

    @torch.inference_mode()
    def encode_frame(self,pixels,q):
        rgb=torch.as_tensor(pixels,device='cuda').permute(0,3,1,2).contiguous().float()/255
        s=self.model.normalizer['agent_pos'].normalize(torch.as_tensor(q,dtype=torch.float32,device='cuda')[None])
        feat=self.model.obs_encoder(dict(overhead=rgb[:1],wrist=rgb[1:],agent_pos=s if self.model.cond_use_s else torch.zeros_like(s)))
        return feat,self.model._field_s(s)

    @torch.inference_mode()
    def heatmap(self,image):
        crop=self.aligner.det.crop(image)
        x=torch.as_tensor(np.ascontiguousarray(crop),device='cuda').permute(2,0,1)[None].float()/255
        return self.aligner.det.net(x)

    @torch.inference_mode()
    def plan(self,anchor):
        if len(self.raw)!=8 or len(self.history)!=8:
            raise ValueError('B1 原图与反馈历史不完整')
        q=np.stack([joints for _,joints in self.history])
        feat=torch.cat([f for f,_ in self.features],dim=1)
        self.condition.copy_(self.model.cond_net(feat))
        self.field_state.copy_(self.features[-1][1])
        self.initial.copy_(self.action_scale.normalize(torch.as_tensor(anchor[None],device='cuda')))
        self.graph.replay()
        base=self.action_scale.unnormalize(self.output)[0].cpu().numpy()
        if base.shape!=(8,7) or not np.isfinite(base).all():
            raise ValueError('B1 ODE 原始目标无效')
        self.base_plan=base
        self.details={'applied':False,'phase':'after_close' if self.closed else 'closing_chunk'}
        if self.closed or min(float(np.min(base[:,6])),float(q[-1,6]),float(anchor[6]))<=self.close_mm:
            return base
        logits=torch.stack(tuple(self.heat)).mean(0)
        uv=self.aligner.det.positions(logits)[0].cpu().numpy()
        mapped=np.r_[uv,1.]@self.aligner.par['matrix']
        distance=float(np.linalg.norm(mapped-self.aligner.fk(q[-1])))
        if not np.isfinite(distance):raise ValueError('B1 木块位置无效')
        self.details.update(block_uv=uv.tolist(),distance_mm=distance)
        if distance>self.radius:
            self.details['phase']='outside_approach'
            return base
        corrected=self.aligner.plan(uv,q[-1],base,True,wrist=self.wrist,qhistory=q)
        if not np.isfinite(corrected).all() or not np.array_equal(corrected[:,6],base[:,6]):
            raise ValueError('B1 横向纠偏产生无效关节或夹爪目标')
        self.details.update(applied=True,phase='approach')
        return corrected

    def predict_recorded(self,pixels,query,stamp):
        if pixels.shape!=(2,480,640,3) or pixels.dtype!=np.uint8:
            raise ValueError('B1 纠偏缺少全分辨率双相机输入')
        if not self.history or stamp-self.last_stamp>70_000_000:
            self.history.clear()
            self.raw.clear()
            self.heat.clear()
            self.features.clear()
            self.chunk=None
        if int(query['execution'])!=self.execution:
            self.closed=False
        previous=query.get('previous_sent')
        if query['q'][6]<=self.close_mm or (previous is not None and previous[6]<=self.close_mm):
            if not self.closed:self.chunk=None
            self.closed=True
        self.raw.append(cv2.cvtColor(pixels[0],cv2.COLOR_RGB2BGR))
        self.heat.append(self.heatmap(self.raw[-1]))
        self.wrist=cv2.cvtColor(pixels[1],cv2.COLOR_RGB2BGR)
        small=ode_pixels((self.raw[-1],self.wrist))
        self.features.append(self.encode_frame(small,query['q']))
        target=super().predict_recorded(small,query,stamp)
        if self.trace['ready']:
            self.trace['alignment']=dict(self.details)
        return target

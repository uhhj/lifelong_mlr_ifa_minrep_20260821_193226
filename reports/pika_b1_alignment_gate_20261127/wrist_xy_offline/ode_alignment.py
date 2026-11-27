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
from b1_alignment_gate import OVERHEAD_TOLERANCE_MM, wrist_block, wrist_centered
from wrist_cue import ConfirmedCue, modified_step, signed_cue


def manifest_id(bundle):
    data=json.dumps(bundle,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
    return hashlib.sha256(data).hexdigest()


class GroundedODE(B1Predictor):
    """Keep the installed ODE history, command acknowledgement and chunk scheduling."""
    acknowledged_axes = 6
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
        for name, expected in bundle['station_files'].items():
            if digest(Path(__file__).with_name(name)) != expected:
                raise ValueError('B1 门控组件校验失败：' + name)
        super().__init__(dict(profile,sha256=bundle['base_sha256']))
        script=WORK/bundle['algorithm']
        spec=importlib.util.spec_from_file_location('piper_grounded_b1',script)
        alg=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(alg)
        self.aligner=alg.B1OfflineAlignment(WORK,WORK/bundle['weights'])
        self.algorithm=alg
        self.camera=json.loads(Path(__file__).with_name('wrist_camera.json').read_bytes())
        self.cue=ConfirmedCue()
        self.mode=None
        self.plan_mode=None
        if self.aligner.steady is None:
            raise ValueError('缺少已验收的末段纠偏参数')
        self.raw=deque(maxlen=8)
        self.heat=deque(maxlen=8)
        self.features=deque(maxlen=8)
        self.wrist=None
        self.closed=False
        self.details=None
        self.base_plan=None
        self.radius=bundle['radius_mm']
        self.guiding=False
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
    def base_forecast(self,anchor):
        if len(self.raw)!=8 or len(self.history)!=8:
            raise ValueError('B1 原图与反馈历史不完整')
        feat=torch.cat([f for f,_ in self.features],dim=1)
        self.condition.copy_(self.model.cond_net(feat))
        self.field_state.copy_(self.features[-1][1])
        self.initial.copy_(self.action_scale.normalize(torch.as_tensor(anchor[None],device='cuda')))
        self.graph.replay()
        base=self.action_scale.unnormalize(self.output)[0].cpu().numpy()
        if base.shape!=(8,7) or not np.isfinite(base).all():
            raise ValueError('B1 ODE 原始目标无效')
        return base

    @torch.inference_mode()
    def plan(self,anchor):
        q=np.stack([joints for _,joints in self.history])
        base=self.base_forecast(anchor)
        self.base_plan=base
        self.plan_mode=self.mode
        self.details={'applied':False,'phase':'after_close'}
        if self.closed:
            return base
        # Enter in the original calibrated approach region. Once acquired, keep
        # the bounded planar correction until the actuator accepts the close.
        self.guiding = self.guiding or self.visual['distance_mm'] <= self.radius
        if not self.guiding:
            self.details['phase']='outside_approach'
            return base
        uv=np.asarray(self.visual['block_uv'])
        aligner=self.aligner
        if self.mode is None:
            corrected=aligner.plan(uv,q[-1],base,True,wrist=self.wrist,qhistory=q)
        else:
            goal=self.algorithm.settle_goal(uv,q,self.wrist,aligner.par,aligner.steady,aligner.flange)
            start=aligner.fk(q[-1])
            original=self.algorithm.move(aligner.par,start,goal,'feedback','heatmap')-start
            revised=modified_step(original,self.mode,self.pose[:3,:3],self.camera,float(aligner.par['cap']))
            path=start+np.arange(1,9)[:,None]/8*revised
            corrected,errors=aligner.xy.joint_commands(base,path,aligner.par['tip'])
            if max(errors[k] for k in ('xy_max_mm','height_max_mm','angle_max_deg'))>1e-4:
                raise ValueError('B1 腕部末段纠偏位姿重建误差超限')
            self.details.update(baseline_xy_mm=original.tolist(),revised_xy_mm=revised.tolist())
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
            self.guiding=False
        if int(query['execution'])!=self.execution:
            self.guiding=False
            self.cue=ConfirmedCue()
        phase=query['gripper_phase']
        if phase not in ('approach','closing','carrying','released'):
            raise ValueError('B1 夹爪执行阶段无效')
        self.closed=phase!='approach'
        self.raw.append(cv2.cvtColor(pixels[0],cv2.COLOR_RGB2BGR))
        self.heat.append(self.heatmap(self.raw[-1]))
        self.wrist=cv2.cvtColor(pixels[1],cv2.COLOR_RGB2BGR)
        self.visual={'stamp':stamp,'distance_mm':None,'block_uv':None,'wrist_box':None,'aligned':False}
        side=None
        if not self.closed and len(self.heat)==8:
            logits=torch.stack(tuple(self.heat)).mean(0)
            uv=self.aligner.det.positions(logits)[0].cpu().numpy()
            mapped=np.r_[uv,1.]@self.aligner.par['matrix']
            distance=float(np.linalg.norm(mapped-self.aligner.fk(query['q'])))
            if not np.isfinite(distance):raise ValueError('B1 木块位置无效')
            box=wrist_block(self.wrist)
            self.visual.update(block_uv=uv.tolist(),distance_mm=distance,wrist_box=box,
                               aligned=distance<=OVERHEAD_TOLERANCE_MM and wrist_centered(box,query['q'][6]))
            self.pose=self.aligner.flange(np.rad2deg(query['q'][:6]))
            tip=self.pose[:3,3]+self.pose[:3,:3]@self.aligner.par['tip']
            z0,ramp,_,_=self.aligner.steady['guard']
            if (self.guiding or distance<=self.radius) and tip[2]<=self.aligner.steady['contact_z']+z0+ramp:
                if wrist_centered(box,query['q'][6]):
                    side='centered'
                else:
                    side=signed_cue(self.wrist)
        self.mode=self.cue.observe(stamp,side)
        if self.chunk is not None and self.plan_mode!=self.mode:
            self.chunk=None
        self.visual.update(wrist_side=side,confirmed_side=self.mode)
        small=ode_pixels((self.raw[-1],self.wrist))
        self.features.append(self.encode_frame(small,query['q']))
        target=super().predict_recorded(small,query,stamp)
        if self.trace['ready']:
            self.trace['alignment']=dict(self.details)
            self.trace['alignment'].update(self.visual)
            if not self.closed:
                # Keep the three independent width forecasts even when the arm
                # legitimately consumes steps 2–8 from an earlier plan.
                anchor=query['previous_sent'] if query['execution'] else query['q']
                forecast=(self.base_plan if self.trace['chunk_stamp']==stamp else
                          self.base_forecast(np.asarray(anchor,dtype=np.float32)))
                self.intent_widths=forecast[:,6]
            else:
                self.intent_widths=()
        else:
            self.intent_widths=()
        return target

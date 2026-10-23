"""Frozen official CLIP patch export with global reconstruction parity, no policy updates."""
import h5py
from .common import *

def main():
 verify();capacity(int(3.3*1024**3));assert not (OUT/'PATCH_MANIFEST.json').exists()
 from physical_effect_probe.export_native_view import native_batch
 import robomimic.utils.obs_utils as ObsUtils
 from libero.lifelong.utils import get_task_embs
 c=config();ObsUtils.initialize_obs_utils_with_obs_specs({'obs':c.data.obs.modality});seed(41);p=full_policy();enc=p.image_encoders.get_base_model();clip=enc.clip
 assert not enc.linear,'This protocol expects actual official768 pooler and separate image_proj'
 captured=[]
 def hook(module,args,result):captured.append((result.last_hidden_state.detach(),result.pooler_output.detach()))
 handle=clip.register_forward_hook(hook);manifest=[];forward_batches=0
 for task in (6,7):
  d,native=data(task);meta=d['metadata'];b=js(PARENT/'bindings'/('task%d.json'%task));assert sha(b['hdf5'])==b['hdf5_sha256'];frames=[];globalframes=[];offsets={};maxdiff=0.;nframes=0
  with h5py.File(b['hdf5'],'r') as h:
   lang=json.loads(h['data'].attrs['problem_info'])['language_instruction'];tok=get_task_embs(c,[lang]);instr=tok['input_ids'][0];mask=tok['attention_mask'][0]
   for demo in sorted({r['demo_id'] for r in meta}):
    g=h['data'][demo];n=len(g['actions']);offsets[demo]=(nframes,n)
    for st in range(0,n,16):
     obs=[{c.data.obs_key_mapping[k]:g['obs'][k][i] for k in c.data.obs_key_mapping} for i in range(st,min(n,st+16))];captured.clear()
     with torch.no_grad():
      y=native_batch(p,obs,instr,mask);assert len(captured)==2
      ls=y[:,0,0];sc=p.film_scale(ls);sh=p.film_shift(ls);cam=[]
      for camera,(hidden,pool) in enumerate(captured):
       assert hidden.shape[1:]==(197,768)
       reconstructed=(1+sc)*p.image_proj(pool)+sh
       diff=float((reconstructed-y[:,0,1+camera]).abs().max());maxdiff=max(diff,maxdiff);assert torch.allclose(reconstructed,y[:,0,1+camera],atol=1e-4,rtol=1e-5)
       z=clip.vision_model.post_layernorm(hidden[:,1:]);z=(1+sc[:,None])*p.image_proj(z)+sh[:,None];cam.append(z.cpu())
      frames.append(torch.stack(cam,1));globalframes.append(y[:,0].cpu());forward_batches+=1
    nframes+=n
  patch=torch.cat(frames);assert patch.shape==(nframes,2,196,512);indices=[]
  for r in meta:
   off,n=offsets[r['demo_id']];indices.append(torch.arange(r['window_start'],r['window_start']+8).clamp(max=n-1)+off)
  indices=torch.stack(indices);globalwindows=torch.cat(globalframes)[indices];native_diff=float((globalwindows-native).abs().max());assert torch.allclose(globalwindows,native,atol=1e-4,rtol=1e-5),'New extraction differs from original native path'
  f=OUT/'private'/('patch%d.pth'%task);capacity(patch.numel()*4+50*1024**2);save(f,dict(patch=patch,window_indices=indices,frame_offsets=offsets))
  manifest.append(dict(task=task,frames=nframes,windows=len(meta),shape=list(patch.shape),dtype=str(patch.dtype),bytes=f.stat().st_size,sha256=sha(f),global_reconstruction_max_abs=maxdiff,native_all_window_max_abs=native_diff,hdf5_sha256=b['hdf5_sha256'],labels_sha256=sha(OLD/'output/private'/('labels%d.pt'%task)),native_sha256=sha(OLD/'output/private'/('native%d.pt'%task))))
  log('patch_export_task',**manifest[-1]);del patch,frames,globalframes,globalwindows,native,d
 handle.remove();verify();write(OUT/'PATCH_MANIFEST.json',dict(status='PASS_FROZEN_PATCH_GLOBAL_PARITY',tasks=manifest,native_batch_calls=forward_batches,camera_clip_calls=2*forward_batches,optimizer_calls=0,environment_episodes=0))
if __name__=='__main__':main()

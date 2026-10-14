"""Two process modes keep CLIP text / vision model memory separate on CPU."""
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time
from types import SimpleNamespace
import h5py
import numpy as np
import torch
import yaml
from peft import LoraConfig,get_peft_model
from transformers import CLIPTokenizer,logging
from .common import sha,write,npz

def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def seq(in_dim,hidden,out_dim,act):
    return torch.nn.Sequential(torch.nn.Linear(in_dim,hidden),act,torch.nn.Linear(hidden,out_dim))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--mode',choices=['language','vision'],required=True)
    a=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(42);logging.set_verbosity_error()
    assert not torch.cuda.is_initialized()
    manifest=json.loads((a.out/'DATA_MANIFEST.json').read_text());protocol=json.loads((a.out/'PROTOCOL.json').read_text())
    binding=dict(manifest=sha(a.out/'DATA_MANIFEST.json'),protocol=sha(a.out/'PROTOCOL.json'),
                 encoder_source=sha(__file__),common_source=sha(Path(__file__).with_name('common.py')),
                 rgb_source=sha(a.repo/'libero/lifelong/models/modules/rgb_modules.py'),
                 lang_source=sha(a.repo/'libero/lifelong/models/modules/language_modules.py'),
                 policy_source=sha(a.repo/'libero/lifelong/models/complete_policy.py'),
                 actual_feature_config=sha(a.repo/'libero/configs/policy/complete_policy.yaml'))
    write(a.out/'ENCODER_BINDING.json',binding,immutable=True)
    cp_path=a.repo/'checkpoints'/protocol['checkpoint']
    assert sha(cp_path)==protocol['checkpoint_sha256']
    start=time.perf_counter();cp=torch.load(cp_path,map_location='cpu');cfg=cp['cfg'];sd=cp['state_dict']
    rank,alpha=cfg.policy.rank,cfg.policy.alpha
    actual_cfg=yaml.safe_load((a.repo/'libero/configs/policy/complete_policy.yaml').read_text())
    assert actual_cfg['use_language'] is True and actual_cfg['use_state'] is False
    kwargs=LoraConfig(r=rank,lora_alpha=alpha,target_modules=['q_proj','v_proj'])
    def portion(prefix): return {k[len('module.'+prefix+'.'):]:v for k,v in sd.items() if k.startswith('module.'+prefix+'.')}
    if a.mode=='language':
        weights=portion('language_encoder')
        heads={k:portion(k) for k in ['film_scale','film_shift','state_encoder','state_scale','state_shift']}
        max_word_len=cfg.data.max_word_len
        del cp,sd,cfg;gc.collect()
        native=module(a.repo/'libero/lifelong/models/modules/language_modules.py','pilot_language')
        encoder=get_peft_model(native.CLIPTextEncoder(),kwargs);encoder.load_state_dict(weights,strict=True);encoder.eval();encoder.requires_grad_(False)
        descriptions=[next(r['language'] for r in manifest['demos'] if r['task']==task) for task in range(10)]
        tokenizer=CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16',local_files_only=True)
        tokens=tokenizer(text=descriptions,max_length=max_word_len,padding='max_length',return_attention_mask=True,return_tensors='pt')
        with torch.inference_mode(): values=encoder(**tokens).numpy().copy()
        assert values.shape==(10,512) and np.isfinite(values).all()
        npz(a.out/'LANGUAGE.npz',embeddings=values,input_ids=tokens['input_ids'].numpy(),attention_mask=tokens['attention_mask'].numpy())
        temp=a.out/'spatial_heads.tmp.pt';torch.save(heads,temp);temp.replace(a.out/'spatial_heads.pt')
        write(a.out/'LANGUAGE_COMPLETE.json',dict(binding=binding,language_sha=sha(a.out/'LANGUAGE.npz'),heads_sha=sha(a.out/'spatial_heads.pt'),
            descriptions=descriptions,max_word_len=max_word_len,strict_load=True,seconds=time.perf_counter()-start,cuda_initialized=torch.cuda.is_initialized()),immutable=True)
        print('LANGUAGE_COMPLETE',flush=True);return
    weights=portion('image_encoders');projection_weights=portion('image_proj')
    del cp,sd,cfg;gc.collect()
    native=module(a.repo/'libero/lifelong/models/modules/rgb_modules.py','pilot_rgb')
    encoder=get_peft_model(native.CLIPVisionEncoder(),kwargs);encoder.load_state_dict(weights,strict=True)
    projection=torch.nn.Linear(768,512);projection.load_state_dict(projection_weights,strict=True)
    encoder.eval();projection.eval();encoder.requires_grad_(False);projection.requires_grad_(False)
    del weights,projection_weights;gc.collect()
    lcomplete=json.loads((a.out/'LANGUAGE_COMPLETE.json').read_text())
    assert lcomplete['binding']==binding and lcomplete['language_sha']==sha(a.out/'LANGUAGE.npz') and lcomplete['heads_sha']==sha(a.out/'spatial_heads.pt')
    with np.load(a.out/'LANGUAGE.npz') as z: lang=torch.from_numpy(z['embeddings'].copy())
    hs=torch.load(a.out/'spatial_heads.pt',map_location='cpu');heads={}
    for name,w in hs.items():
        heads[name]=seq(w['0.weight'].shape[1],w['0.weight'].shape[0],w['2.weight'].shape[0],torch.nn.ReLU(inplace=True) if name=='state_encoder' else torch.nn.GELU())
        heads[name].load_state_dict(w,strict=True);heads[name].eval();heads[name].requires_grad_(False)
    # Execute only the actual native spatial_encode method for a runtime parity check.
    tree=ast.parse((a.repo/'libero/lifelong/models/complete_policy.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CompletePolicy')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='spatial_encode')
    scope={'torch':torch};exec(compile(ast.Module(body=[method],type_ignores=[]),str(a.repo/'libero/lifelong/models/complete_policy.py'),'exec'),scope)
    completed=0;parity_done=False
    for rec in manifest['demos']:
        target=a.out/(rec['id']+'_features.npz');complete=a.out/(rec['id']+'_FEATURE_COMPLETE.json')
        if complete.exists():
            old=json.loads(complete.read_text());assert old['binding']==binding and old['sha256']==sha(target)
            completed+=rec['n'];continue
        with np.load(a.out/(rec['id']+'_raw.npz')) as z: state=torch.tensor(z['state'],dtype=torch.float32)
        visual=[];full=[];raw_hashes=[hashlib.sha256(),hashlib.sha256()]
        with h5py.File(rec['hdf5'],'r') as h:
            d=h['data']['demo_'+str(rec['demo'])]
            for begin in range(0,rec['n'],8):
                end=min(begin+8,rec['n']);length=end-begin
                raw=[d['obs'][key][begin:end] for key in ['agentview_rgb','eye_in_hand_rgb']]
                for digest,rgb in zip(raw_hashes,raw): digest.update(rgb.tobytes())
                # Native image preprocessing, no display flip / augmentation.
                image=torch.from_numpy(np.concatenate(raw,axis=0)).float().div(255).clamp(0,1).permute(0,3,1,2)
                with torch.inference_mode():
                    v=projection(encoder(image,langs=None));v=torch.stack([v[:length],v[length:]],dim=1)
                    le=lang[rec['task']].expand(length,-1)
                    vm=(1+heads['film_scale'](le))[:,None,:]*v+heads['film_shift'](le)[:,None,:]
                    # Match actual native use_state=False branch, NOT a reconstructed state token.
                    sm=(1+heads['state_scale'](le))*v[:,1]+heads['state_shift'](le)
                    H=torch.cat([le[:,None,:],vm,sm[:,None,:]],dim=1)
                    if not parity_done:
                        from robomimic.utils.obs_utils import ImageModality
                        official=ImageModality._default_obs_processor(torch.from_numpy(np.concatenate(raw,axis=0)))
                        assert torch.equal(image,official)
                        adapter=SimpleNamespace(image_encoders=encoder,image_proj=projection,rgbkey={'agentview_rgb':0,'eye_in_hand_rgb':0},
                            language_encoder=lambda **kw:le,cfg=SimpleNamespace(aug=False,policy=SimpleNamespace(use_language=True,use_state=False)),**heads)
                        data=dict(instr_token={},obs=dict(agentview_rgb=image[:length,None],eye_in_hand_rgb=image[length:,None],
                            joint_states=state[begin:end,None,:7],gripper_states=state[begin:end,None,7:]))
                        check=scope['spatial_encode'](adapter,data)[:,0]
                        error=float((check-H).abs().max());assert torch.allclose(check,H,rtol=1e-5,atol=1e-5),error
                        write(a.out/'NATIVE_FEATURE_PARITY.json',dict(status='PASS',max_abs=error,native_method_source=binding['policy_source'],
                            image_preprocess_bitwise_equal=True,language='native CLIPTextEncoder strict checkpoint weights',
                            full_H_shape=[4,512],native_use_state=False,fourth_token='state-FiLM(eih visual)',
                            feature_location='image_proj BEFORE FiLM',cuda_initialized=torch.cuda.is_initialized()),immutable=True)
                        parity_done=True
                visual.append(v.numpy().copy());full.append(H.numpy().copy())
        V=np.concatenate(visual);H=np.concatenate(full)
        assert V.shape==(rec['n'],2,512) and H.shape==(rec['n'],4,512)
        assert np.isfinite(V).all() and np.isfinite(H).all()
        npz(target,visual=V,full_H=H)
        write(complete,dict(binding=binding,sha256=sha(target),raw_image_sha256=[s.hexdigest() for s in raw_hashes],frames=rec['n']),immutable=True)
        completed+=rec['n']
        status=dict(stage='FEATURE_EXPORT',frames_completed=completed,frames_total=manifest['frames'],demo=rec['id'],
                    seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,formal_training_resumed=False)
        write(a.out/'STATUS.json',status);print(json.dumps(status),flush=True)
    assert not torch.cuda.is_initialized()
    write(a.out/'FEATURES_COMPLETE.json',dict(status='PASS',binding=binding,frames=completed,seconds=time.perf_counter()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,formal_training_resumed=False),immutable=True)

if __name__=='__main__': main()

from .common import *
def main():
    assert not (O/'CONTRACT.json').exists();assert js(O/'INTEGRATION.json')['status']=='PASS';assert js(O/'COST_ISOLATION.json')['status']=='PASS'
    files=js(O/'INPUT_SHA256.json')
    for p in list((W/'source').rglob('*.py'))+list((W/'attachment').glob('*'))+[W/'admission.py',W/'language_and_gates.npz',W/'pipeline.py',W/'env.sh']:
        if p.is_file():files[str(p)]=sha(p)
    for name in ['ADMISSION.json','PARENT_BINDING.json','INTEGRATION.json','COST_ISOLATION.json']:files[str(O/name)]=sha(O/name)
    write(O/'CONTRACT.json',dict(name='ZR_v2_OBJECT_seed43_repeat1_parent',seed=43,files=files,epochs=40,optimizer_steps=10550,sample_exposures=105240,formal_episodes=200,conditional_extension='all four exact metric criteria and all engineering acceptance; user 2026-10-18 authorization'))
    write(O/'resolved_calibration.json',dict(parent_cfg=cfg('coreset'),spec=__import__('yaml').safe_load((W/'attachment/spec.yaml').read_text()),actual=dict(seed=43,batch=10,epochs_per_stage=10,workers=0,drop_last=False,sampler_seed=0,optimizer='AdamW_UV_only',lr=1e-4,weight_decay=.1,warmup500=True,common_eval_frozen=True,T0_eval_frozen=True,output=str(O))))
    status('READY')
if __name__=='__main__':main()

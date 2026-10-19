from .common import *
def main():
    assert not (O/'CONTRACT.json').exists();assert js(O/'INTEGRATION.json')['status']=='PASS';assert js(O/'COST_ISOLATION.json')['status']=='PASS'
    files=js(O/'INPUT_SHA256.json')
    for p in list((W/'source').rglob('*.py'))+list((W/'attachment').glob('*'))+[W/'admission.py',W/'language_and_gates.npz',W/'pipeline.py',W/'env.sh',W/'RUN_SPEC.json',Path(SPEC['controller_contract'])]:
        if p.is_file():files[str(p)]=sha(p)
    for name in ['ADMISSION.json','PARENT_BINDING.json','INTEGRATION.json','COST_ISOLATION.json']:files[str(O/name)]=sha(O/name)
    write(O/'CONTRACT.json',dict(name=ARM+'_'+SUITE+'_seed'+str(SEED),seed=SEED,files=files,epochs=40,optimizer_steps=STEPS,sample_exposures=EXPOSURES,formal_episodes=200,conditional_extension='User 2026-10-19 whole-three-seed mean AUC positive and at least 2/3 positive; all engineering PASS'))
    write(O/'resolved_calibration.json',dict(parent_cfg=cfg(PARM),spec=SPEC,actual=dict(seed=SEED,batch=10,epochs_per_stage=10,workers=0,drop_last=False,sampler_seed=0,optimizer='AdamW_UV_only',lr=1e-4,weight_decay=.1,warmup500=True,common_eval_frozen=True,T0_eval_frozen=True,output=str(O))))
    status('READY')
if __name__=='__main__':main()

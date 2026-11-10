"""Independent process-exit and actual final-tensor load acceptance. No rollouts."""
import json,os,subprocess,time
import torch
from .common import O,write,sha,verify
from .model import cfg,strict_downstream
from libero.lifelong.models.brief import BriefPolicy
def main():
    verify();assert (O/'pipeline.exit').read_text().strip()=='0'
    lines=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
    assert not [x for x in lines if any(' -m coreset_exp.'+m in x for m in ('export','train','evaluate','summarize')) and 'ps -eo' not in x]
    loaded={}
    for arm in ('random','coreset'):
        p=BriefPolicy(cfg(arm),None).cpu()
        for t in range(6,10):
            file=O/arm/'models'/f'mlrifa_{t}_model_ep99.pth'
            raw=torch.load(file,map_location='cpu');assert raw['epoch_index']==99 and raw['task']==t
            assert raw['contract_sha256']==sha(O/'CONTRACT.json') and raw['modal_sha256']==sha(O/'MODAL_CONTRACT.json')
            loaded[f'{arm}_{t}']=strict_downstream(p,file)
    summary=json.loads((O/'SUMMARY.json').read_text());assert summary['formal_records']==400 and summary['old_inputs_unchanged']
    write(O/'INDEPENDENT_ACCEPTANCE.json',dict(status='PASS',pipeline_exit=0,loaded_downstream_keys=loaded,
        final_checkpoints=8,formal_records=400,source_verified=True,updated_unix=time.time()))
    with (O/'REPORT.md').open('a') as f:f.write('\n最终验收：COMPLETED；pipeline.exit=0，无训练/评估残留进程，8个最终checkpoint实际张量完整加载通过。\n')
    write(O/'RUN_STATUS.json',dict(status='COMPLETED',seed=42,arms=['random','coreset'],epochs=800,formal_records=400,updated_unix=time.time()))
if __name__=='__main__':main()

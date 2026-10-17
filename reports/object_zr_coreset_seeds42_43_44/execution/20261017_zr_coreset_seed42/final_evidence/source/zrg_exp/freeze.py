from .common import *
def main():
    verify(False);assert not(O/'CONTRACT.json').exists();assert json.loads((O/'INTEGRATION.json').read_text())['status']=='PASS'
    configs={str(O/n):sha(O/n) for n in ('spec.yaml','resolved_train.json','resolved_eval.json','MODEL_CONTRACT.json','MODAL_CONTRACT.json','INPUT_SHA256.json','PROTOTYPE_BINDING.json','PARENT_BINDING.json','INTEGRATION.json','NATIVE_ACTION_CHECK.json','NATIVE_PATH_PARITY.json')}
    configs.update({str(S.parent/n):sha(S.parent/n) for n in ('env.sh','pipeline.sh','supervise.sh','final_process_audit.py','admission.py','NUMERIC_ADMISSION.json')})
    write(O/'CONTRACT.json',dict(source_sha256=sources(),configs=configs,formal_epochs=400,formal_steps=105500,episodes=200,integration_calls_max=4,parent_sha256=sha(P/'CONTRACT.json')))
    print('FROZEN',sha(O/'CONTRACT.json'))
if __name__=='__main__':main()

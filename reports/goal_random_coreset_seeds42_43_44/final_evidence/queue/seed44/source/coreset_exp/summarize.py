from .common import SEED,SELECTOR_SEED
"""Read-only numerical audit. No environment execution or optimizer creation."""
import json,math,time,csv
from fractions import Fraction as F
from .common import O,sha,write,verify,hashes
def metrics(r):
    fwt=sum(r[j,j] for j in range(6,10))/4
    nbt=sum(sum(r[j,j]-r[i,j] for i in range(j+1,10))/(9-j) for j in range(6,9))/3
    auc=sum(sum(r[i,j] for i in range(j,10))/(10-j) for j in range(6,10))/4
    return {n:dict(fraction=str(v),percent=float(v*100)) for n,v in dict(AUC=auc,FWT=fwt,NBT=nbt).items()}
def readlines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def main():
    c=verify();features=json.loads((O/'FEATURES.json').read_text());result={};allrows=[];attempt_counts={}
    assert json.loads((O/'GPU_PREFLIGHT.json').read_text())['status']=='PASS'
    assert json.loads((O/'TASK6_TENSOR_PARITY.json').read_text())['equal']
    assert (O/'random/INITIAL_STATE.json').read_bytes()==(O/'coreset/INITIAL_STATE.json').read_bytes()
    expected_counts={int(t):n for t,n in features['counts'].items()}
    totalsteps=100*sum(math.ceil((expected_counts[t]+sum(math.ceil(.5*expected_counts[j]) for j in range(6,t)))/10) for t in range(6,10))
    assert totalsteps==76100
    for arm in ('random','coreset'):
        root=O/arm;epochs=json.loads((root/'epochs.json').read_text());assert len(epochs)==400
        assert [(x['task'],x['epoch']) for x in epochs]==[(t,e) for t in range(6,10) for e in range(100)]
        assert epochs[-1]['steps']==totalsteps
        prior=0
        for row in epochs:
            assert row['steps']==prior+row['batches'];prior=row['steps']
            assert row['batches']==math.ceil((expected_counts[row['task']]+sum(math.ceil(.5*expected_counts[j]) for j in range(6,row['task'])))/10)
        training=json.loads((root/'TRAINING_COMPLETE.json').read_text());assert training['steps']==totalsteps
        assert training['checkpoints']==hashes(root/'models','*.pth') and len(training['checkpoints'])==4
        r={};trialkeys=set();events=[];coverage=[]
        for t in range(6,10):
            m=json.loads((root/'storage'/f'task{t}.json').read_text());assert m['K']==math.ceil(expected_counts[t]*.5)
            assert len({x['destination'] for x in m['selected']})==m['K']
            assert m['contract_sha256']==sha(O/'CONTRACT.json') and m['modal_sha256']==c['modal_sha256']
            for x in m['selected']:assert sha(x['source'])==x['sha256']==sha(root/'buffer'/x['destination'])
            coverage.append(dict(task=t,K=m['K'],selection=m['selection']))
            for j in range(6,t+1):
                p=root/'evaluation'/f'trials_stage{t}_task{j}.jsonl';rows=readlines(p)
                assert len(rows)==20 and {x['trial'] for x in rows}==set(range(20))
                for row in rows:
                    assert row['arm']==arm and row['stage']==t and row['task']==j and row['seed']==SEED
                    assert row['epoch']==99 and row['initial_state_index']==row['trial']
                    assert isinstance(row['success'],bool) and 1<=row['vector_group_steps']<=600
                    assert row['checkpoint_sha256']==sha(root/'models'/f'mlrifa_{t}_model_ep99.pth')
                    assert row['modal_sha256']==c['modal_sha256']
                    key=(t,j,row['trial']);assert key not in trialkeys;trialkeys.add(key)
                r[t,j]=F(sum(x['success'] for x in rows),20);allrows.extend(rows)
                cell=json.loads((root/'evaluation'/f'cell_stage{t}_task{j}.json').read_text());assert cell['success_rate']==float(r[t,j])
                ev=readlines(root/'evaluation'/f'attempts_stage{t}_task{j}.jsonl');events.extend(ev)
                assert [x['event'] for x in ev]==['environment_construct_start','environment_construct_complete','vector_attempt_start','vector_attempt_complete','vector_attempt_start','vector_attempt_complete','environment_closed']
                assert [x['env_seed'] for x in ev if x['event']=='vector_attempt_start']==[SEED,SEED+1]
        assert len(trialkeys)==200
        attempt_counts[arm]=dict(vector_attempts=20,environment_lanes=200,technical_exceptions=0,formal_records=200)
        result[arm]=dict(metrics=metrics(r),matrix=[[float(r[i,j]*100) if (i,j) in r else None for j in range(6,10)] for i in range(6,10)],
            epochs=400,steps=totalsteps,training_seconds=training['summed_epoch_seconds'],coverage=coverage)
    left={(x['stage'],x['task'],x['trial']):x for x in allrows if x['arm']=='random'}
    for x in allrows:
        if x['arm']=='coreset':assert x['initial_state_sha256']==left[x['stage'],x['task'],x['trial']]['initial_state_sha256']
    for p,h in json.loads((O/'OLD_INPUTS_READONLY.json').read_text()).items():assert sha(p)==h,p
    delta={n:result['coreset']['metrics'][n]['percent']-result['random']['metrics'][n]['percent'] for n in ('AUC','FWT','NBT')}
    write(O/'SUMMARY.json',dict(arms=result,delta_coreset_minus_random_percentage_points=delta,attempts=attempt_counts,
        formal_records=400,old_inputs_unchanged=True,seed=SEED,extensions=[42,43,44],export_seconds=features['seconds']))
    report=[f'# Seed{SEED} Random vs Coreset','', '仅一个训练seed；按论文描述启用状态模态，官方checkpoint原始模态配置缺失时仍记未知。', '',
        '| 方法 | AUC (%) | FWT (%) | NBT (%) |','|---|---:|---:|---:|']
    for arm in result:report.append('| '+arm+' | '+' | '.join(f"{result[arm]['metrics'][n]['percent']:.6f}" for n in ('AUC','FWT','NBT'))+' |')
    report.extend(['','400正式闭环；每组400epoch、'+str(totalsteps)+'步、4个ep99最终checkpoint。',
        '状态：数值与输入审计通过，仍需独立进程退出和权重加载验收。',
        '限制：单seed不能判断统计显著性；RTX3090与论文A100不同；预训练模态元数据、warmup/调度及最终checkpoint选择细节未公开；IFA随buffer变化，不能独立归因于几何覆盖。',
        '仅执行用户授权的GOAL三seed42/43/44；不更新旧实验指标，不执行后续调参。'])
    (O/'REPORT.md').write_text('\n'.join(report)+'\n')
    write(O/'RUN_STATUS.json',dict(status='AWAITING_INDEPENDENT_EXIT_AUDIT',updated_unix=time.time()))
if __name__=='__main__':main()

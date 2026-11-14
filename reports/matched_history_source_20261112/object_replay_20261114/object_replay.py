"""Run the frozen OBJECT evaluator against the accepted Task9 checkpoints."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


BASE = Path('/data/mlr_ifa/experiment_workspaces')
SOURCE = BASE / 'MATCHED_HISTORY_SOURCE_GOAL_OBJECT_20261112_v1_engineering02'
ROOT = BASE / 'MATCHED_HISTORY_SOURCE_OBJECT_EVAL_REPLAY_20261114_v1'
CONTRACT = 'bc3db9e6f252cc5167f7bb043ae420a7daeb88b7b6e4959d5c9a8cfa6281b397'
CASES = ('OBJECT_41', 'OBJECT_42', 'OBJECT_43')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1048576), b''):
            h.update(data)
    return h.hexdigest()


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temp, path)


def gpu():
    return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip()


def check_source():
    assert (SOURCE/'controller.exit').read_text().strip()=='0'
    contract=SOURCE/'PAIR_CONTRACT.json'
    assert sha(contract)==CONTRACT
    frozen=read(contract)['files']
    assert all(sha(Path(p))==h for p,h in frozen.items())
    for case in CASES:
        p=SOURCE/'pairs'/case
        for name in ['evaluate.exit','train_E.exit','train_S.exit']:
            assert (p/name).read_text().strip()=='0'
        orig=p/'OUTCOMES.json'
        item=ACCEPTED[case]
        assert sha(orig)==item['outcome_sha256'] and len(read(orig))==160
        for arm in 'ES':
            tr=read(p/('TRAIN_%s.json'%arm))
            assert tr['sha256']==item['training'][arm]['checkpoint_sha256']==sha(Path(tr['checkpoint']))


def evaluate(case):
    p=SOURCE/'pairs'/case
    out=ROOT/case
    sys.path.insert(0,str(p))
    import evaluate as evaluator
    verify=evaluator.verify
    original_dir=evaluator.R
    assert original_dir==p
    def verify_inputs():
        assert sha(Path(__file__))==read(ROOT/'CONTRACT.json')['wrapper_sha256']
        result=verify()
        for arm in 'ES':
            assert sha(evaluator.model_path(arm,9))==ACCEPTED[case]['training'][arm]['checkpoint_sha256']
        return result
    # The original evaluator reads its initial-state ledger through R.
    assert sha(out/'INITIAL_SPLIT.json')==sha(p/'INITIAL_SPLIT.json')
    evaluator.R=out
    evaluator.verify=verify_inputs
    evaluator.main()


def controller():
    with (ROOT/'STARTED').open('x') as f:
        f.write(str(os.getpid()))
    try:
        check_source()
        for case in CASES:
            directory=ROOT/case
            with (directory/'evaluate.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-B',__file__,'evaluate',case],
                    cwd=directory,stdout=log,stderr=subprocess.STDOUT)
                write(ROOT/'STATUS.json',dict(case=case,phase='evaluate',pid=child.pid,controller=os.getpid()))
                rc=child.wait()
            (directory/'evaluate.exit').write_text(str(rc))
            assert rc==0,(case,rc)
            assert len(read(directory/'OUTCOMES.json'))==160
        check_source()
        write(ROOT/'STATUS.json',dict(phase='complete',controller=os.getpid()))
        (ROOT/'controller.exit').write_text('0')
    except BaseException:
        write(ROOT/'FAILURE.json',dict(error=traceback.format_exc()))
        (ROOT/'controller.exit').write_text('1')
        raise


def launch(source):
    assert not ROOT.exists() and not gpu()
    assert shutil.disk_usage('/data').free >= 2*1024**3 + 128*1024**2
    check_source()
    ROOT.mkdir()
    path=ROOT/'object_replay.py'
    path.write_text(source,encoding='utf-8')
    write(ROOT/'CONTRACT.json',dict(wrapper_sha256=sha(path),parent=CONTRACT,
        cases=CASES,formal_episodes=480,optimizer_updates=0))
    for case in CASES:
        dest=ROOT/case
        dest.mkdir()
        shutil.copyfile(SOURCE/'pairs'/case/'INITIAL_SPLIT.json',dest/'INITIAL_SPLIT.json')
    with (ROOT/'controller.log').open('x') as log:
        proc=subprocess.Popen([sys.executable,'-B',str(path),'controller'],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    return dict(root=str(ROOT),controller=proc.pid,contract_sha256=sha(ROOT/'CONTRACT.json'))


def status():
    out=dict(observed=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
             state=read(ROOT/'STATUS.json') if (ROOT/'STATUS.json').exists() else None,
             exit=(ROOT/'controller.exit').read_text().strip() if (ROOT/'controller.exit').exists() else None,
             gpu=gpu(), cases={})
    for case in CASES:
        p=ROOT/case
        ep=p/'OUTCOMES.json'
        out['cases'][case]=len(read(ep)) if ep.exists() else 0
    if (ROOT/'FAILURE.json').exists():
        out['failure']=read(ROOT/'FAILURE.json')
        for case in CASES:
            p=ROOT/case
            if (p/'evaluate.exit').exists() and (p/'evaluate.exit').read_text()!='0':
                out['failure'][case]=(p/'evaluate.log').read_text(errors='replace').splitlines()[-16:]
    return out


if __name__=='__main__' and len(sys.argv)>1:
    if sys.argv[1]=='controller':
        controller()
    else:
        assert sys.argv[1]=='evaluate' and sys.argv[2] in CASES
        evaluate(sys.argv[2])

"""Materialize versioned runtime only after candidate validation; never launches training."""
import json,shutil,time,subprocess
from pathlib import Path
from .io import write_json
from qp_lbfgsb_kkt_driven_v1.offline import OUT
from sample_qp_lifelong_v1_continuation_v3 import common as c

OPT=c.ROOT/'sample_qp_optimized_v1_continuation_v4';TRAIN=c.ROOT/'sample_qp_lifelong_v1_continuation_v4';REPORT=c.ROOT/'reports/sample_qp_lifelong_v1_continuation_v4'

def main():
    for name in ['HARD_QP_REGRESSION_V3.json','GPU_CHECK.json','NORMAL_NONZERO_GPU_PARITY.json']:
        assert c.read(OUT/name)['status']=='PASS',name
    assert 'Ran 16 tests' in (OUT/'candidate_tests_final.log').read_text() and (OUT/'candidate_tests_final.log').read_text().rstrip().endswith('OK')
    assert not OPT.exists() and not TRAIN.exists() and not REPORT.exists(),'Do not overwrite runtime'
    OPT.mkdir();TRAIN.mkdir();REPORT.mkdir()
    candidate=c.ROOT/'qp_trust_continuation_candidate_v1'
    for name in ['__init__.py','solver.py','core.py','trust.py','numeric.py','io.py','tests.py']:
        (OPT/name).write_text((candidate/name).read_text())
    (OPT/'__init__.py').write_text('"""Version 4: unchanged v3 first fallback, equivalent trust-constr second fallback."""\n')
    (TRAIN/'__init__.py').write_text('"""Frozen lifelong protocol with v4 numerical fallback and failure capture."""\n')
    s=(candidate/'step.py').read_text().replace('from .core import','from sample_qp_optimized_v1_continuation_v4.core import');(TRAIN/'step.py').write_text(s)
    for name in ['common.py','train.py','evaluate.py','summarize.py']:
        s=(c.ROOT/'sample_qp_lifelong_v1_continuation_v3'/name).read_text().replace('sample_qp_lifelong_v1_continuation_v3','sample_qp_lifelong_v1_continuation_v4').replace('sample_qp_optimized_v1_continuation_v3','sample_qp_optimized_v1_continuation_v4')
        if name=='train.py':
            s=s.replace('refs=reference_batches(algo,task);w=Workspace(g.params(algo));first_epoch=0',"refs=reference_batches(algo,task);w=Workspace(g.params(algo));first_epoch=0\n            reference_manifest_sha=sha(OUT/('plans/references_task%d.json'%task))")
            old='rec=update(algo,data,task,epoch,refs,w,guard)'
            new="rec=update(algo,data,task,epoch,refs,w,guard,\n                            diag_dir=OUT/'qp_failures'/('attempt%03d_t%d_e%03d_b%04d'%(attempt,task,epoch,batch_index)),\n                            context=dict(seed=SEED,task=task,epoch=epoch,batch=batch_index,pre_global_step=finished,record_indices=ix,source_ids=ids,reference_sha256=reference_manifest_sha))"
            assert old in s;s=s.replace(old,new)
        (TRAIN/name).write_text(s)
    # Plans/config/evaluation settings remain those of the frozen parent. No checkpoint migration.
    (REPORT/'plans').mkdir()
    for p in sorted((c.OUT/'plans').glob('*.json')):shutil.copyfile(p,REPORT/'plans'/p.name)
    if (c.OUT/'INITIALIZATION_EXPECTED.json').exists():shutil.copyfile(c.OUT/'INITIALIZATION_EXPECTED.json',REPORT/'INITIALIZATION_EXPECTED.json')
    parent=c.read(c.OUT/'PROTOCOL_FROZEN.json');protocol=json.loads(json.dumps(parent))
    protocol.update(parent_protocol=str(c.OUT/'PROTOCOL_FROZEN.json'),parent_protocol_sha256=c.sha(c.OUT/'PROTOCOL_FROZEN.json'),protocol_revision='v4_trust_constr_second_equivalent_fallback',freeze_time=time.time())
    protocol['algorithm']+=' If the unchanged first L-BFGS-B fallback also fails, run the frozen trust-constr second fallback from x_standard and use its first project-KKT-passing iterate, else fail-stop after complete capture.'
    protocol['numerical_revision']=dict(standard='unchanged SLSQP+10k+20k',first_fallback='unchanged v3 LBFGSB options',second_fallback=c.read(OUT/'TRUST_PROTOCOL.json'),failure_capture='explicit train->step->core->solver path; atomic arrays + trajectories + pre-update metadata',formal_training_resumed=False)
    for folder in [OPT,TRAIN]:
        for p in folder.glob('*.py'):protocol['files'][str(p.relative_to(c.ROOT))]=c.sha(p)
    for p in (REPORT/'plans').glob('*.json'):protocol['files'][str(p.relative_to(c.ROOT))]=c.sha(p)
    write_json(REPORT/'PROTOCOL_FROZEN.json',protocol)
    checks={}
    for name in ['solver.py','core.py','trust.py','numeric.py','io.py']:
        checks[name]=(OPT/name).read_bytes()==(candidate/name).read_bytes()
    checks['step.py']=(TRAIN/'step.py').read_text().replace('from sample_qp_optimized_v1_continuation_v4.core import','from .core import')==(candidate/'step.py').read_text()
    assert all(checks.values())
    write_json(REPORT/'CANDIDATE_CODE_TRANSFER.json',dict(status='PASS',checks=checks,candidate_sources={p.name:c.sha(p) for p in candidate.glob('*.py')},formal_sources={str(p.relative_to(c.ROOT)):c.sha(p) for d in [OPT,TRAIN] for p in d.glob('*.py')},parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),formal_training_resumed=False))
    write_json(REPORT/'STATUS.json',dict(status='PACKAGED_PENDING_FINAL_PREFLIGHT',formal_training_resumed=False))
    (TRAIN/'RUNBOOK.md').write_text('''# v4 numerical continuation runtime\n\nTraining remains stopped. Standard QP and v3 L-BFGS-B are unchanged. Only a failed first fallback invokes trust-constr from the original 30k solution, using the first KKT-certified iterate.\n\nThe frozen parent seed/task/data/optimizer/evaluation protocol is inherited. Failure dumps include both fallback trajectories and pre-update metadata. No /tmp or sitecustomize dependency.\n\nCandidate GPU checks transfer by byte-identical solver/core/trust implementation and import-only step normalization. Formal preflight records the transfer evidence. This round does not launch train/evaluate or create a formal recovery slot. Diagnostic POST_STEP_STATE files support a direct single-step restore; they are not canonical epoch checkpoints. Before a later authorized continuation, use the existing parent-child canonical recovery migration; do not continue a failed process or label a mid-epoch diagnostic state an epoch recovery slot.\n''')
    print('V4_PACKAGED_NOT_STARTED',flush=True)
if __name__=='__main__':main()

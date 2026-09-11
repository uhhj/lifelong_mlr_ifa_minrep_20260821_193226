from pathlib import Path
from confirmation_experiment.common import ROOT,RUNS,REPORTS,read,write,sha

WORK=RUNS/'recovery_v1'
AUDIT=REPORTS/'recovery_v1'
CELLS=[(stage,task) for stage in range(6,10) for task in range(6,stage+1)]

def install_abort_routing():
    # Keep the original FAILED.json as evidence, rather than deleting it to
    # evade the old guard. This recovery has its own failure domain.
    import confirmation_experiment.common as common
    import confirmation_experiment.instrumentation as instrument
    original=common.wait_for
    def routed(path,abort=None,timeout=24*3600):
        if abort is not None and Path(abort)==RUNS/'FAILED.json':abort=WORK/'FAILED.json'
        return original(path,abort,timeout)
    common.wait_for=routed;instrument.wait_for=routed

def verify_original_code():
    pre=read(REPORTS/'PREFLIGHT_COMPLETE.json')
    for name,h in pre['package_sha256'].items():assert sha(ROOT/name)==h,('Frozen implementation changed',name)
    frozen=read(REPORTS/'FROZEN_INPUTS.json')
    for name,h in frozen['code_sha256'].items():assert sha(ROOT/name)==h,name
    for name,h in pre['historical_asset_sha256'].items():assert sha(ROOT/name)==h,name

def verify_preserved_before_resume():
    for name,m in read(AUDIT/'PRESERVED.json')['files'].items():
        p=ROOT/name
        assert p.stat().st_size==m['bytes'] and sha(p)==m['sha256'],('Original evidence changed',name)

def verify_frozen_inputs():
    frozen=read(REPORTS/'FROZEN_INPUTS.json')
    for name,h in frozen['input_sha256'].items():assert sha(ROOT/name)==h,name
    for files in frozen['cache_sha256'].values():
        for name,h in files.items():assert sha(ROOT/name)==h,name

def package_hashes():
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'confirmation_experiment/recovery').rglob('*'))
            if p.is_file() and p.suffix in ('.py','.md','.ps1') and '__pycache__' not in p.parts}

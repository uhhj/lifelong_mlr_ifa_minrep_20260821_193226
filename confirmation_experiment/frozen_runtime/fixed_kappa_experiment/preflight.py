"""Audit all three endpoint pairs and inputs before spending time on new training."""
import argparse
import json
from pathlib import Path

from install import plan
from protocol import audit_references, require, source_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(); repo=a.repo.resolve()
    require(not plan(repo),'Install experiment files before preflight')
    result={'status':'PASS','source_sha256':source_hashes(repo),'seeds':{}}
    for seed in (42,43,44):
        feature=repo/'features'/('object_seed%d'%seed)
        audit=audit_references(repo,seed,feature,repo/'checkpoints/object_multitask_model_ep99.pth')
        audit['feature_file_counts']={}
        for task in range(6,10):
            count=sum(1 for _ in (feature/str(task)).glob('*.pt'))
            require(count>0,'Missing feature cache seed%d task%d'%(seed,task))
            audit['feature_file_counts'][str(task)]=count
        result['seeds'][str(seed)]=audit
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('PASS: all three endpoint pairs, source files and feature directories')


if __name__=='__main__': main()

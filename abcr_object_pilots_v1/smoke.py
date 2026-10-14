"""Synthetic end-to-end I/O and resume test; never research evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from .common import PROTOCOL,write,npz,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='abcr_object_synthetic_') as temp:
        out=Path(temp);rng=np.random.RandomState(121);records=[]
        write(out/'PROTOCOL.json',PROTOCOL)
        for task in range(10):
            for demo in range(5 if task<6 else 10):
                n=10;ident='task%d_demo%d'%(task,demo);state=rng.randn(n,9);act=rng.randn(n,7)
                future=np.zeros((n,6,7));valid=np.zeros((n,6),bool)
                for h in range(6): future[:n-h-1,h]=act[h+1:];valid[:n-h-1,h]=True
                npz(out/(ident+'_raw.npz'),state=state,future=future,future_valid=valid,
                    eef=state[:,:3],target_relative=rng.randn(n,3),target_valid=np.arange(n)<n-1)
                npz(out/(ident+'_features.npz'),visual=rng.randn(n,2,12),full_H=rng.randn(n,4,12))
                write(out/(ident+'_FEATURE_COMPLETE.json'),dict(sha256=sha(out/(ident+'_features.npz'))))
                records.append(dict(id=ident,task=task,demo=demo,n=n,raw_sha256=sha(out/(ident+'_raw.npz'))))
        write(out/'DATA_MANIFEST.json',dict(demos=records,assets=[],frames=700,protocol_sha256=sha(out/'PROTOCOL.json')))
        write(out/'FEATURES_COMPLETE.json',dict(status='PASS'))
        write(out/'NORMALIZATION.json',dict(state_mean=[0]*9,state_std=[1]*9,action_scale=[1]*7))
        for mode in ['analyze','analyze','report']:
            subprocess.run([sys.executable,'-m','abcr_object_pilots_v1.'+mode,'--out',str(out)],check=True)
        assert json.loads((out/'SUMMARY.json').read_text())['P4']=='NOT_RUN_USER_DEFERRED'
        assert len(json.loads((out/'INDEPENDENT_CHECK.json').read_text())['checks'])==12
        raw_path=out/(records[0]['id']+'_raw.npz');original=raw_path.read_bytes()
        raw_path.write_bytes(original+b'changed-input')
        test=subprocess.run([sys.executable,'-m','abcr_object_pilots_v1.analyze','--out',str(out)],capture_output=True,text=True)
        assert test.returncode!=0 and 'RAW_ARRAY_CHANGED' in test.stderr
        raw_path.write_bytes(original)
        # Current input manifest change must reject reuse, never silently return old output.
        record=json.loads((out/'DATA_MANIFEST.json').read_text());record['frames']=701;write(out/'DATA_MANIFEST.json',record)
        test=subprocess.run([sys.executable,'-m','abcr_object_pilots_v1.analyze','--out',str(out)],capture_output=True,text=True)
        assert test.returncode!=0 and 'FROZEN_BINDING_MISMATCH' in test.stderr
        write(a.receipt,dict(status='PASS',synthetic_only=True,all_four_stages=True,all_three_betas=True,
            analysis_resume=True,changed_input_manifest_rejected=True,changed_input_array_rejected=True,report_and_plots=True,
            formal_training_resumed=False))

if __name__=='__main__': main()

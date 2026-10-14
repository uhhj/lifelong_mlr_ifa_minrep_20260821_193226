"""Recalculate from exported binary feature arrays; no original B numerical code."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    out=args.out
    fr=json.loads((out/'frozen_original_rules.json').read_text())
    export=json.loads((out/'FEATURE_EXPORT_COMPLETE.json').read_text())
    for name,sha in export['outputs'].items():
        assert digest(out/name)==sha
    with open(out/'BASE_PAIR_SCORES.csv',encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    final=json.loads((out/'BASE_VALIDATION_REPORT.json').read_text())
    results={}
    max_difference=0.
    for role in ('calibration','validation'):
        with np.load(out/(role+'_records.npz'),allow_pickle=False) as z:
            d={k:z[k] for k in z.files}
        lookup={x:i for i,x in enumerate(d['ids'])}
        counts=dict(tp=0,fp=0,tn=0,fn=0)
        scores_negative=[]
        for r in rows:
            if r['role']!=role:
                continue
            i,j=lookup[r['id_a']],lookup[r['id_b']]
            values=[]
            for key in ('agent_unmod','eih_unmod'):
                a,b=d[key][i].astype(float),d[key][j].astype(float)
                value=sum(float(x)*float(y) for x,y in zip(a,b))/math.sqrt(sum(float(x)**2 for x in a)*sum(float(x)**2 for x in b))
                values.append(value)
            sv=min(values)
            max_difference=max(max_difference,abs(sv-float(r['min_cos'])))
            qa,qb=d['eef_quat_xyzw'][i],d['eef_quat_xyzw'][j]
            dot=abs(sum(float(a)*float(b) for a,b in zip(qa,qb)))/math.sqrt(sum(float(a)**2 for a in qa)*sum(float(b)**2 for b in qb))
            ds=max(math.sqrt(sum(float(a-b)**2 for a,b in zip(d['eef_pos_m'][i],d['eef_pos_m'][j])))/.03,
                   2*math.acos(min(1.,dot))/math.radians(15.),abs(float(d['gripper_open_fraction'][i]-d['gripper_open_fraction'][j]))/.10)
            max_difference=max(max_difference,abs(ds-float(r['physical_distance'])))
            pred=ds<=1 and sv>=fr['visual']['tau_v']
            assert pred==(r['predicted_match']=='True')
            if r['label']=='0' and ds<=1:
                scores_negative.append(sv)
            if r['label'] in ('0','1'):
                key=('tp' if pred else 'fn') if r['label']=='1' else ('fp' if pred else 'tn')
                counts[key]+=1
        expected=final['roles'][role]['metrics_relative_to_ai']
        assert all(counts[k]==expected[k] for k in counts)
        results[role]=counts
        if role=='calibration':
            q=sorted(scores_negative)[math.ceil(.95*(len(scores_negative)-1))]
            assert abs(min(1.,q+1e-7)-fr['visual']['tau_v'])<1e-12
    assert max_difference<1e-10
    record=dict(status='PASS',python=sys.version,numpy=np.__version__,pairs=len(rows),
                role_confusion_matrices=results,max_abs_difference=max_difference,
                original_B_numeric_code_imported=False,
                scope='Independent array arithmetic and threshold reconstruction, not independent human labels or independent raw-image encoder execution',
                formal_training_resumed=False)
    path=out/'CROSS_ENVIRONMENT_RECHECK.json'
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)
    print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':
    main()

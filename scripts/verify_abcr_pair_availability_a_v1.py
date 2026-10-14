"""Read-only independent result verification; run only after all stage commits."""
import argparse
import collections
import json
import math
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from abcr_pair_availability_a_v1.io import sha, write, read_table


def decode(path):
    d=torch.load(path,map_location='cpu',weights_only=True)
    h=d['feature'].numpy()[0,-1].astype(np.float64)
    a=d['actions'].numpy()[0,-1].astype(np.float64)
    return h[[1,2]],a


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=ap.parse_args()
    root=a.root.resolve();out=root/'reports/abcr_pair_availability_a_v1'
    assert json.loads((out/'STATUS.json').read_text())['status']=='COMPLETE_AVAILABLE_ASSETS'
    checks=[];details=[];max_cos=0.;max_dist=0.;tested=0
    ctx=json.loads((out/'EXPERIMENT_CONTEXT.json').read_text())
    assert all(sha(root/k)==v for k,v in ctx['source_sha256'].items())
    checks.append('formal source hashes unchanged')
    for seed in (42,43,44):
        for task in (7,8,9):
            d=out/('object_seed%d_task%d'%(seed,task))
            complete=json.loads((d/'COMPLETE.json').read_text())
            assert all(sha(d/k)==v for k,v in complete['outputs_sha256'].items())
            m=json.loads((d/'manifest.json').read_text())
            assert m['dropped_current']==m['dropped_old']==0
            assert m['execution']['cpu_only'] and not m['execution']['cuda_initialized']
            sums=read_table(d/'summary_with_uncertain.csv');old=read_table(d/'by_old_task.csv')
            candidates=read_table(d/'retrieved_candidates.csv');trials=read_table(d/'batch_trials.csv')
            batch=read_table(d/'batch_comparison.csv')
            for r in sums:
                threshold=r['visual_threshold']
                assert int(r['all_possible_cross_pairs'])==m['n_current_used']*m['n_old_used']
                assert int(r['share_pairs'])+int(r['bif_pairs'])+int(r['uncertain_pairs'])==int(r['visual_candidates'])
                by=[x for x in old if x['visual_threshold']==threshold]
                for fld,total in [('possible','all_possible_cross_pairs'),('matched','visual_candidates'),('share','share_pairs'),('bif','bif_pairs')]:
                    assert sum(int(x[fld]) for x in by)==int(r[total])
                cr=[x for x in candidates if x['visual_threshold']==threshold]
                assert len(cr)==int(r['retrieved_topk_matched'])
                assert all(int(x['old_task'])<task for x in cr)
                for label,field,coverage in [('share','retrieved_topk_share','anchor_share_coverage_retrieved'),('bif','retrieved_topk_bif','anchor_bif_coverage_retrieved')]:
                    group=[x for x in cr if x['class']==label]
                    assert len(group)==int(r[field])
                    assert abs(len({x['new_file'] for x in group})/m['n_current_used']-float(r[coverage]))<=1e-6
            for b in batch:
                group=[r for r in trials if r['strategy']==b['strategy'] and r['visual_threshold']==b['visual_threshold']]
                assert len(group)==500
                assert abs(sum(r['has_both']=='True' for r in group)/500-float(b['fraction_batches_with_both']))<1e-12
            # Independent NumPy float64 formulas on stratified rows. CSV is rounded
            # to six decimals by supplied code, so numeric parity uses <=1e-6.
            # Semantic class checks are strict away from a rounding boundary.
            boundary=0
            for r in read_table(d/'audit_sample.csv'):
                h,act=decode(r['new_file']);k,oldact=decode(r['old_file'])
                cos=float(np.mean(np.sum(h*k,axis=1)/(np.linalg.norm(h,axis=1)*np.linalg.norm(k,axis=1))))
                dist=float(np.sqrt(np.mean(((act[:6]-oldact[:6])/np.array(m['scale_motion_iqr_floor_0p1']))**2)))
                conflict=(act[6]>0)!=(oldact[6]>0)
                idle=np.sqrt(np.mean(act[:6]**2))<.05 and np.sqrt(np.mean(oldact[:6]**2))<.05
                max_cos=max(max_cos,abs(cos-float(r['visual_cosine'])))
                max_dist=max(max_dist,abs(dist-float(r['action_distance'])))
                assert abs(cos-float(r['visual_cosine']))<1e-6
                assert abs(dist-float(r['action_distance']))<2e-6
                if min(abs(cos-float(r['visual_threshold'])),abs(dist-.5),abs(dist-1.5))<2e-6:
                    boundary+=1
                else:
                    assert cos>=float(r['visual_threshold'])
                    label='share' if dist<=.5 and not conflict else 'bif' if dist>=1.5 or conflict else 'uncertain'
                    assert r['class']==label
                assert int(r['both_idle'])==int(idle)
                tested+=1
            details.append(dict(seed=seed,task=task,status='PASS',stratified_boundary_rows=boundary))
    assert not torch.cuda.is_initialized()
    write(out/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',stage_units=details,
        checks=checks+['stage output SHA256 integrity','count partitions','by-old sums','retrieved counts and anchor coverages',
            '500 paired trials per threshold/strategy','no future old-task rows','independent NumPy formulas on stratified candidates'],
        numeric_rows=tested,max_csv_cosine_absolute_difference=max_cos,max_csv_distance_absolute_difference=max_dist,
        numeric_precision_note='Supplied CSV rounds six decimals; independent recomputation uses binary64 from original binary32 tensors. Boundary rows are reported, never relabeled.',
        human_or_physical_audit=False,formal_training_resumed=False))
    print('INDEPENDENT_VERIFICATION PASS',tested,'rows',flush=True)


if __name__=='__main__':main()

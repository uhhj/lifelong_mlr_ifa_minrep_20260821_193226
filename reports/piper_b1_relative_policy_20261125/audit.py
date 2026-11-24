"""Recompute pair-level scores from scalar episode outcomes and check the previous baseline."""
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_bytes())


def summary(out):
    previous=out.parent/'piper_b1_vision_contrasts_20261124'
    archive=load(previous/'final.json')
    provenance=load(out/'data_audit.json')
    fitted=load(out/'fit.json')
    results=load(out/'outcomes.json')
    pairs=archive['geometry']['pairs']
    assert archive['checkpoint_ids']['baseline']==provenance['checkpoint_sha256']
    assert hashlib.sha256((out/'protocol.md').read_bytes()).hexdigest()==provenance['protocol_sha256']
    assert pairs==provenance['prior_pairs'] and len(pairs)==16
    training=set(fitted['train_episodes'])
    held=set(archive['geometry']['test_episodes'])
    assert len(held)==33 and training.isdisjoint(held)
    assert fitted['coordinate_training']=={'wood':50,'jaw':33}
    replay=fitted['verification']
    assert replay['natural_images_to_joints']==33 and replay['paired_image_swaps']==32
    assert replay['maximum_metric_difference_mm']<1e-6
    assert len(training)==provenance['train_episode_count']
    assert sum(fitted['train_window_counts'].values())+len(held)==provenance['windows']
    assert provenance['paired_directions']==32
    metrics={}
    table={}
    keys=['signed_mm','orthogonal_mm','amplitude_mm','contact_signed_mm','donor_distance_reduction_mm','separation_mm']
    boot=np.random.default_rng(1125).integers(0,16,(20000,16))
    for mode in ['base','fixed','permuted','relative','wood_only','jaw_only']:
        rows=[r for r in results['directions'] if r['mode']==mode]
        assert len(rows)==32
        values=[]
        for a,b in pairs:
            duo=[r for r in rows if (r['recipient'],r['donor']) in [(a,b),(b,a)]]
            assert len(duo)==2 and {r['recipient'] for r in duo}=={a,b}
            for r in duo:
                np.testing.assert_allclose(np.hypot(r['signed_mm'],r['orthogonal_mm']),r['amplitude_mm'],atol=1e-7)
            values.append([np.mean([r[k] for r in duo]) for k in keys])
        v=np.array(values)
        signed=v[:,0]
        table[mode]=v.tolist()
        difference=signed-np.array(table['base'])[:,0]
        if mode=='base':
            np.testing.assert_allclose(signed,np.array(archive['approach_pairs_zero']['baseline']['overhead'])[:,0],atol=1e-7)
            np.testing.assert_allclose(v[:,-1],archive['geometry']['pair_separations_mm'],atol=1e-7)
        metrics[mode]=dict(positive=int(sum(signed>0)),positive_over_1mm=int(sum(signed>1)),
            negative_over_1mm=int(sum(signed< -1)),median_signed_mm=float(np.median(signed)),
            mean_signed_mm=float(np.mean(signed)),signed_range_mm=[float(signed.min()),float(signed.max())],
            median_gain=float(np.median(signed/v[:,-1])),
            median_absolute_orthogonal_directed_mm=float(np.median([abs(r['orthogonal_mm']) for r in rows])),
            median_diff_vs_base=float(np.median(difference)),
            median_diff_vs_base_ci95=np.percentile(np.median(difference[boot],axis=1),[2.5,97.5]).tolist(),
            median_diff_vs_fixed=float(np.median(signed-np.array(table.get('fixed',table['base']))[:,0])),
            median_contact_signed_mm=float(np.median(v[:,3])),positive_contact=int(sum(v[:,3]>0)),
            median_donor_distance_reduction_mm=float(np.median(v[:,4])),
            median_total_displacement_mm=float(np.median(v[:,2])))
    metrics_natural={}
    for mode in ['base','fixed','permuted','relative','expert']:
        rows=[r for r in results['natural'] if r['mode']==mode]
        assert len(rows)==len(held) and {r['episode'] for r in rows}==held
        advances=[r['initial_distance_mm']-r['end_distance_mm'] for r in rows]
        metrics_natural[mode]=dict(n=len(rows),closer=int(sum(a>0 for a in advances)),
            median_advance_mm=float(np.median(advances)),
            median_endpoint_vs_expert_mm=float(np.median([r['endpoint_error_vs_expert_mm'] for r in rows])),
            median_distance_to_block_mm=float(np.median([r['end_distance_mm'] for r in rows])),
            median_arm_mae_deg=float(np.median([r['arm_mae_deg'] for r in rows])))
    diff=metrics_natural['base']['median_advance_mm']-archive['metrics']['baseline']['approach/zero/natural']['median_advance_mm']
    assert abs(diff)<1e-6
    return dict(pairs=pairs,pair_averages_columns=keys,pair_averages=table,metrics=metrics,
                natural=metrics_natural,boot_unit='disjoint pair',bootstrap_repetitions=len(boot))


if __name__=='__main__':
    out=Path(__file__).resolve().parent
    (out/'summary.json').write_bytes((json.dumps(summary(out),indent=1,allow_nan=False)+'\n').encode())

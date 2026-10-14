"""Read-only audit of the finished R5 evidence, independent scalar recomputation."""
import argparse, csv, io, time, unittest
from pathlib import Path
import h5py
import numpy as np
from scipy.stats import spearmanr
from .common import read, write, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); out = a.out; begin = time.monotonic()
    manifest = read(out/'START_MANIFEST.json')['starts']
    binding = read(out/'ENGINE_BINDING.json')
    assert sha(out/'PROTOCOL.json') == binding['protocol']
    for path, expected in binding['files'].items():
        assert sha(path) == expected, path
    checked = {}; expected_B = set(); expected_C = set(); batch_pairs = 0
    for e in manifest:
        s = e['start']; sid = s['start_id']
        assert s['anchor']['stage_step'] == s['stage_step']
        for p, h in [(Path(s['anchor']['anchor']['path']), s['anchor']['anchor']['sha256']),
                     (a.repo/s['plan']['path'], s['plan']['sha256'])]:
            assert sha(p) == h; checked[str(p)] = h
        plan = read(a.repo/s['plan']['path'])
        assert len(e['batch_records']) == len(s['record_indices']) == 10
        assert not e['batch_unmapped_slots']
        for i, r in zip(s['record_indices'], e['batch_records']):
            assert plan['immutable_feature_origins'][i] == r['origin']
        batch_traj = {(r['task'], r['demo']) for r in e['batch_records']}
        for group in e['probes'].values():
            assert len(group) == 12
            assert all((r['task'], r['demo']) not in batch_traj for r in group)
        records = e['batch_records'] + e['probes']['old'] + e['probes']['new']
        for pair in e['pairs']:
            records += [pair['new'], pair['old']]
            expected_B.add((sid, pair['pair_id']))
        for r in records:
            p = a.repo/r['origin']
            if str(p) not in checked:
                assert sha(p) == r['sha256']; checked[str(p)] = r['sha256']
        expected_C.update((sid, arm) for arm in ['MLR', 'SHARE_ONLY', 'BIF_ONLY', 'ABCR'])
        batch_pairs += len(e['batch_pairs'])
    Bs = [read(p) for p in sorted((out/'B').glob('*/*.json'))]
    Cs = [read(p) for p in sorted((out/'C').glob('*/*.json'))]
    assert {(v['row']['start_id'], v['row']['pair_id']) for v in Bs} == expected_B
    assert {(v['start_id'], v['arm']) for v in Cs} == expected_C
    for r in Bs:
        assert r['binding'] == binding and all(r['restored'].values())
        assert r['row']['optimizer_delta'] == r['row']['scheduler_delta'] == 1
    native = {r['start_id']: r for r in Cs if r['arm'] == 'MLR'}
    no_op = 0
    for r in Cs:
        assert r['binding'] == binding and all(r['restored'].values())
        assert r['counts']['optimizer_delta'] == r['counts']['scheduler_delta'] == 1
        if r['arm'] == 'MLR': assert all(r['native_parity'].values())
        if r['no_active_pair']:
            n = native[r['start_id']]
            assert r['candidate_evidence'] == n['candidate_evidence']
            assert r['observations'] == n['observations'] and r['rng_candidate'] == n['rng_candidate']
            no_op += 1
        for v in r['observations']:
            assert v['updated'] - v['base'] == v['delta']
    # All physical input metrics independently recomputed from original raw arrays.
    pilot = a.repo/'reports/abcr_object_pilots_v1'
    dm = {(r['task'], r['demo']): r for r in read(pilot/'DATA_MANIFEST.json')['demos']}
    norm = read(pilot/'NORMALIZATION.json'); raw = {}; metric_max_error = 0.
    for e in manifest:
        for p in e['pairs']:
            rr = []
            for side in ['new', 'old']:
                r = p[side]; key = (r['task'], r['demo'])
                if key not in raw:
                    m = dm[key]; path = pilot/(m['id']+'_raw.npz')
                    assert sha(path) == m['raw_sha256']; raw[key] = dict(np.load(path))
                rr.append((raw[key], r['t']))
            (n, i), (o, j) = rr
            delta = (n['actions'][i+1:i+5].astype(float)-o['actions'][j+1:j+5].astype(float))/norm['action_scale']
            values = {'action_distance': np.sqrt(np.mean(delta**2)),
                      'state_distance': np.linalg.norm((n['state'][i]-o['state'][j])/norm['state_std']),
                      'eef_m': np.linalg.norm(n['eef'][i]-o['eef'][j]),
                      'object_relative_m': np.linalg.norm(n['target_relative'][i]-o['target_relative'][j])}
            for key, v in values.items():
                err = abs(float(v)-p[key]); metric_max_error = max(metric_max_error, err)
                assert err <= 1e-7, (key, err)
    # Recompute each test fold without using the analysis module's split or regression code.
    audit = read(out/'AUDIT_SELECTION_PRIVATE.json')['pairs']
    labels = {r['audit_id']: r for r in csv.DictReader((out/'audit/AI_LABELS.csv').open())}
    rows = [v['row'] for v in Bs]; model_features = read(out/'PROTOCOL.json')['B']['models']
    n_folds = 0; pred_error = 0.; nonleak = 0
    for saved in csv.DictReader((out/'R5_B_CROSSFIT.csv').open()):
        subset = saved['subset']
        pool = [r for r in rows if subset == 'all_resolved_diagnostic' or (r['physical_pass'] and (subset == 'physical_pass' or labels[r['audit_id']]['label'] == 'COMPARABLE'))]
        tr = [r for r in pool if r['start_id'] != saved['start_id'] and r['new_trajectory'] not in (saved['new_trajectory'],saved['old_trajectory']) and r['old_trajectory'] not in (saved['new_trajectory'],saved['old_trajectory'])]
        assert len(tr) == int(saved['train_rows']) >= 20
        assert len({r['start_id'] for r in tr}) == int(saved['train_checkpoints']) >= 4
        nonleak += len(tr)
        for name, features in model_features.items():
            X = np.array([[r[k] for k in features] for r in tr]); y = np.array([r['old_delta'] for r in tr])
            mu, sd = X.mean(0), np.maximum(X.std(0), 1e-12); Z = (X-mu)/sd
            # Augmented least squares is independent of the production normal-equation implementation.
            coef = np.linalg.lstsq(np.vstack([Z,np.eye(len(features))]), np.r_[y-y.mean(),np.zeros(len(features))], rcond=None)[0]
            pred = ((np.array([float(saved[k]) for k in features])-mu)/sd)@coef+y.mean()
            err = abs(pred-float(saved['pred_'+name])); pred_error = max(pred_error, float(err)); assert err < 1e-9
        n_folds += 1
    # Verify every saved physical-state frame against its original HDF5 state.
    sim = read(out/'SIMULATOR_RECONSTRUCTION.json')['records']; source = {r['audit_id']: r for r in audit}; frames = 0
    import hashlib
    for r in sim:
        p = source[r['audit_id']]; assert sha(out/'audit'/(r['audit_id']+'.png')) == r['image_sha256'] == labels[r['audit_id']]['image_sha256']
        for side, task in [('new',p['task']),('old',p['old_task'])]:
            m = dm[task,p[side+'_demo']]
            with h5py.File(p[side+'_hdf5'],'r') as f:
                d = f['data/demo_%d'%p[side+'_demo']]
                for v in r['sides'][side]['timeline']:
                    if v['status'] != 'RECONSTRUCTED_FORWARD_ONLY': continue
                    t = v['frame']; state = d['states'][t+1]
                    assert hashlib.sha256(state.tobytes()).hexdigest() == v['state_sha']
                    xyz = state[1+m['target_qpos_adr']:1+m['target_qpos_adr']+3]; eef = d['obs/ee_pos'][t]
                    assert np.array_equal(xyz, v['target_xyz']) and np.array_equal(eef, v['eef_xyz'])
                    assert np.array_equal(xyz-eef, v['target_relative']); frames += 1
    # Structured record from an actual rerun of the existing tests.
    from .tests import TestR5
    stream = io.StringIO(); result = unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestR5))
    write(out/'UNIT_TEST_REPORT.json',dict(passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),log=stream.getvalue()))
    assert result.wasSuccessful()
    write(out/'INPUT_AND_RESULT_INTEGRITY.json',dict(status='PASS',anchors=len(manifest),input_files_verified=len(checked),inputs=checked,engine_binding=binding,
        B_records=len(Bs),C_branches=len(Cs),native_parity_checkpoints=len(native),B_and_C_restores=len(Bs)+len(Cs),exact_no_op_branches=no_op,
        input_metric_max_abs_error=metric_max_error,crossfit_test_rows=n_folds,crossfit_train_test_comparisons=nonleak,crossfit_prediction_max_abs_error=pred_error,
        audit_image_hashes=len(sim),original_physical_state_frames=frames,notes='Independent scalar/split/least-squares recomputation and direct original state checks; same codebase audit, not external replication. Contacts were produced by simulator reconstruction, not independently re-simulated here.',seconds=time.monotonic()-begin,formal_training_resumed=False))
    print('INTEGRITY PASS',len(Bs),len(Cs),n_folds,frames,flush=True)


if __name__ == '__main__': main()

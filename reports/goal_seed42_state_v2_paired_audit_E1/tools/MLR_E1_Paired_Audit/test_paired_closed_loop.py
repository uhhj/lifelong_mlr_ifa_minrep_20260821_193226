"""Synthetic software tests only. These are NOT the user's robot results."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import analyze_paired_closed_loop as audit


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def fixture(root):
    root.mkdir(parents=True)
    (root / 'evaluation').mkdir()
    patch = b'SYNTHETIC SOFTWARE FIXTURE. NOT AN EXPERIMENT PATCH.\n'
    (root / 'source.patch').write_bytes(patch)
    patch_hash = hashlib.sha256(patch).hexdigest()
    contract = {'use_vision': True, 'use_language': True, 'use_state': True,
                'token_order': ['language', 'agentview', 'eye_in_hand', 'state'],
                'code_patch_sha256': patch_hash, 'synthetic_test_only': True}
    contract['fingerprint'] = hashlib.sha256(json.dumps(contract, sort_keys=True,
                                  separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
    dump(root / 'modal_contract.json', contract)
    cp = {str(t): {'task': t, 'epoch_index': 99, 'sha256': '%064x' % t} for t in audit.TASKS}
    dump(root / 'checkpoints/manifest.json', cp)
    choices = {
        (6, 6): set(range(10)),
        (7, 6): set(range(5)) | set(range(10, 15)),
        (8, 6): set(range(4)) | set(range(10, 16)),
        (9, 6): set(range(3)) | set(range(10, 17)),
        (7, 7): set(range(20)), (8, 7): set(range(20)), (9, 7): set(range(20)),
        (8, 8): set(range(8)), (9, 8): set(range(6)), (9, 9): set(range(12)),
    }
    records = []
    for (stage, task), successes in sorted(choices.items()):
        for i in range(20):
            success = i in successes
            records.append({'kind': 'FORMAL', 'run_id': audit.RUN_ID, 'training_seed': 42,
                'stage': stage, 'task_id': task, 'task_name': 'SYNTHETIC_TASK_%d' % task,
                'checkpoint_sha': cp[str(stage)]['sha256'], 'eval_seed': 42,
                'env_batch_seed': 42 + i // 10, 'worker_seed': 42 + i // 10 + i % 10,
                'init_state_id': i, 'init_state_hash': hashlib.sha256(('%d:%d' % (task, i)).encode()).hexdigest(),
                'success': success, 'policy_steps': 100 + stage + i if success else 600,
                'termination_reason': 'success' if success else 'max_policy_steps',
                'exception': None, 'modal_fingerprint': contract['fingerprint']})
    store_records(root, records)
    metrics = {'FWT': .625, 'NBT': 1.0 / 30, 'AUC': .6125, 'FinalAvgSR': .6,
               'NBT_per_task': [0., 0., .1], 'AUC_per_task': [.5, 1., .35, .6]}
    dump(root / 'metrics.json', metrics)
    acceptance = {'status': 'PASS', 'source_patch_sha256': patch_hash,
                  'modal_fingerprint': contract['fingerprint'],
                  'stages': {str(t): {'checkpoint_sha256': cp[str(t)]['sha256']} for t in audit.TASKS},
                  'success_counts_out_of_20': {'%d,%d' % k: len(v) for k, v in choices.items()},
                  'metrics': {k: metrics[k] for k in ('FWT', 'NBT', 'AUC', 'FinalAvgSR')}}
    dump(root / 'final_acceptance.json', acceptance)
    return records, acceptance, cp, contract


def store_records(root, records):
    (root / 'evaluation/rollouts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')


class PairedAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='synthetic_paired_audit_test_')
        self.base = Path(self.tmp.name)
        self.root = self.base / 'evidence'
        self.records, self.acceptance, self.cp, self.contract = fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_end_to_end_synthetic_with_offsetting_outcomes(self):
        out = self.base / 'output'
        result = audit.audit(self.root, out)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['pair_rows_total'], 240)
        self.assertEqual(result['unique_stage_pair_initial_states'], 180)
        self.assertEqual(result['initial_state_histories'], 80)
        self.assertEqual(result['input_hashes_before'], result['input_hashes_after'])
        self.assertEqual(result['new_rollouts'], 0)
        d = audit.read_json(out / 'nbt_decomposition.json')
        self.assertAlmostEqual(d['macro_observed_loss_pp'], 40.0 / 3)
        self.assertAlmostEqual(d['macro_observed_gain_pp'], 10.)
        self.assertAlmostEqual(d['signed_NBT_pp'], 10.0 / 3)
        cells = audit.validate_records(self.records, self.acceptance, self.cp, self.contract)
        _, summaries = audit.paired_views(cells)
        chosen = next(r for r in summaries if r['view']=='adjacent' and r['task_id']==6 and r['after_stage']==7)
        self.assertEqual((chosen['SS'], chosen['SF'], chosen['FS'], chosen['FF']), (5, 5, 5, 5))
        self.assertEqual(chosen['net_success_change_pp'], 0.)
        self.assertEqual(chosen['observed_flip_pp'], 50.)
        self.assertEqual(chosen['ss_median_after_minus_before_steps'], 1.)

    def test_zero_prior_success_denominator_and_no_ss(self):
        rows = [{'transition': k, 'before_policy_steps': 600, 'after_policy_steps': 600}
                for k in ['FF'] * 15 + ['FS'] * 5]
        summary = audit.summarize_pairs(rows)
        self.assertIsNone(summary['conditional_retention'])
        self.assertIsNone(summary['conditional_loss'])
        self.assertIsNone(summary['ss_median_after_minus_before_steps'])
        self.assertEqual(summary['net_success_change_pp'], 25.)

    def test_task9_has_no_forgetting_or_recovery_observation(self):
        cells = audit.validate_records(self.records, self.acceptance, self.cp, self.contract)
        for r in audit.state_histories(cells):
            if r['task_id'] == 9:
                self.assertEqual(r['pattern'], 'single_observation')
                self.assertIsNone(r['adjacent_flips'])
                self.assertIsNone(r['initial_success_ever_observed_failure'])
                self.assertIsNone(r['initial_success_lost_then_final_success'])

    def test_duplicate_record_rejected(self):
        records = copy.deepcopy(self.records)
        records[1] = copy.deepcopy(records[0])
        with self.assertRaisesRegex(audit.AuditError, 'duplicate initial'):
            audit.validate_records(records, self.acceptance, self.cp, self.contract)

    def test_missing_record_rejected(self):
        with self.assertRaisesRegex(audit.AuditError, 'exactly 200'):
            audit.validate_records(self.records[:-1], self.acceptance, self.cp, self.contract)

    def test_same_id_different_hash_rejected(self):
        records = copy.deepcopy(self.records)
        next(r for r in records if r['stage']==7 and r['task_id']==6 and r['init_state_id']==0)['init_state_hash'] = 'a'*64
        with self.assertRaisesRegex(audit.AuditError, 'Cross-stage'):
            audit.validate_records(records, self.acceptance, self.cp, self.contract)

    def test_seed_mismatch_rejected(self):
        records = copy.deepcopy(self.records)
        records[0]['worker_seed'] = 999
        with self.assertRaisesRegex(audit.AuditError, 'worker seed'):
            audit.validate_records(records, self.acceptance, self.cp, self.contract)

    def test_checkpoint_mismatch_rejected(self):
        records = copy.deepcopy(self.records)
        records[0]['checkpoint_sha'] = 'a'*64
        with self.assertRaisesRegex(audit.AuditError, 'checkpoint mismatch'):
            audit.validate_records(records, self.acceptance, self.cp, self.contract)

    def test_boolean_string_rejected(self):
        records = copy.deepcopy(self.records)
        records[0]['success'] = 'false'
        with self.assertRaisesRegex(audit.AuditError, 'JSON boolean'):
            audit.validate_records(records, self.acceptance, self.cp, self.contract)

    def test_failure_not_reaching_horizon_rejected(self):
        records = copy.deepcopy(self.records)
        next(r for r in records if not r['success'])['policy_steps'] = 599
        with self.assertRaisesRegex(audit.AuditError, 'reach horizon'):
            audit.validate_records(records, self.acceptance, self.cp, self.contract)

    def test_wrong_existing_metric_creates_blocked_not_pass(self):
        m = audit.read_json(self.root / 'metrics.json')
        m['NBT'] = -m['NBT']
        dump(self.root / 'metrics.json', m)
        out = self.base / 'bad'
        with self.assertRaisesRegex(audit.AuditError, 'Value mismatch'):
            audit.audit(self.root, out)
        self.assertTrue((out / 'BLOCKED.md').exists())
        self.assertFalse((out / 'audit_acceptance.json').exists())

    def test_output_cannot_be_inside_input(self):
        with self.assertRaisesRegex(audit.AuditError, 'separate directory'):
            audit.audit(self.root, self.root / 'new')

    def test_output_cannot_overwrite(self):
        out = self.base / 'old'
        out.mkdir()
        (out / 'sentinel').write_text('unchanged')
        with self.assertRaisesRegex(audit.AuditError, 'already exists'):
            audit.audit(self.root, out)
        self.assertEqual((out / 'sentinel').read_text(), 'unchanged')

    def test_json_duplicate_key_and_nan_rejected(self):
        with self.assertRaises(audit.AuditError):
            audit.parse_json('{"success":true,"success":false}')
        with self.assertRaises(audit.AuditError):
            audit.parse_json('{"value":NaN}')

    def test_patch_hash_mismatch_rejected(self):
        (self.root / 'source.patch').write_text('changed')
        with self.assertRaisesRegex(audit.AuditError, 'patch hash'):
            audit.audit(self.root, self.base / 'mismatch')


if __name__ == '__main__':
    unittest.main(verbosity=2)

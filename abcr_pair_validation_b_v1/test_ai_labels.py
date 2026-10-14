"""Provenance and uncertainty protections for the v2 AI annotation export."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from .export_ai_labels import atomic_new, export, json_text, sha, validate_decisions


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pack = self.root / 'pack'
        images = self.pack / 'annotation/images'
        images.mkdir(parents=True)
        self.decisions = self.root / 'decisions.json'
        self.protocol = self.root / 'protocol.md'
        self.protocol.write_text('AI_VISUAL_ANNOTATION; human_verified=false', encoding='utf-8')
        self.receipt = self.root / 'receipt.json'
        self.out = self.root / 'out'
        manifest = []
        decisions = []
        hashes = {}
        for i in range(450):
            role = 'calibration' if i < 300 else 'validation'
            demo = 1 if i < 300 else 26
            pair_id = '%s_t0_t1_%03d' % (role, i)
            manifest.append(dict(pair_id=pair_id, role=role,
                                 id_a='hdf5:task0/demo_%d/t%d' % (demo, i),
                                 id_b='hdf5:task1/demo_%d/t%d' % (demo, i), label=''))
            rel = 'annotation/images/' + pair_id + '.png'
            (self.pack / rel).write_bytes(b'fixture-image-' + str(i).encode())
            hashes[rel] = sha(self.pack / rel)
            decisions.append(dict(index=i, code='test', label=['0', '1', 'U'][i % 3],
                                  confidence='medium', reason='test evidence'))
        with (self.pack / 'ANNOTATION_MANIFEST.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
        hashes['ANNOTATION_MANIFEST.csv'] = sha(self.pack / 'ANNOTATION_MANIFEST.csv')
        self.receipt.write_text(json_text(dict(outputs=hashes)), encoding='utf-8')
        self.decisions.write_text(json_text(decisions), encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def run_export(self):
        return export(self.pack, self.decisions, self.protocol, self.out, self.receipt)

    def test_uncertain_is_not_negative_and_ai_source_is_explicit(self):
        summary = self.run_export()
        self.assertEqual(summary['uncertain_count'], 150)
        rows = list(csv.DictReader((self.out / 'AI_LABELS.csv').open(encoding='utf-8')))
        self.assertTrue(all(r['label'] == '' for r in rows if r['decision'] == 'U'))
        self.assertTrue(all(r['label_source'] == 'AI_VISUAL_ANNOTATION' and r['human_verified'] == 'False' for r in rows))
        self.assertEqual(len(list(csv.DictReader((self.out / 'CALIBRATION_AI_LABELS.csv').open(encoding='utf-8')))), 200)
        self.assertEqual(summary['trajectory_overlap'], 0)
        self.assertIsNone(summary['tau_v'])

    def test_resume_identical_and_refuse_changed_decisions(self):
        self.run_export()
        before = sha(self.out / 'COMPLETE.json')
        self.run_export()
        self.assertEqual(before, sha(self.out / 'COMPLETE.json'))
        decisions = json.loads(self.decisions.read_text())
        decisions[0]['label'] = '1'
        self.decisions.write_text(json_text(decisions), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Existing output differs'):
            self.run_export()

    def test_image_tamper_rejected(self):
        image = next((self.pack / 'annotation/images').glob('*.png'))
        image.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Image changed'):
            self.run_export()

    def test_input_manifest_tamper_rejected(self):
        with (self.pack / 'ANNOTATION_MANIFEST.csv').open('a', encoding='utf-8') as f:
            f.write('\n')
        with self.assertRaisesRegex(ValueError, 'Input manifest changed'):
            self.run_export()

    def test_missing_duplicate_or_bad_labels_rejected(self):
        good = dict(index=0, label='U', confidence='low', reason='unclear')
        validate_decisions([good], 1)
        for rows in ([], [good, good], [dict(good, label='human')], [dict(good, reason='')]):
            with self.assertRaises(ValueError):
                validate_decisions(rows, 1)

    def test_atomic_new_refuses_mutation(self):
        path = self.root / 'stable.json'
        atomic_new(path, '{}')
        atomic_new(path, '{}')
        with self.assertRaises(ValueError):
            atomic_new(path, '{"changed":true}')
        self.assertEqual(path.read_text(), '{}')


if __name__ == '__main__':
    unittest.main()

import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from capture_core import EpisodeWriter, audit_episode, valid_sample
from console import paired_frames
from timeline import Timeline, TimelineTail


def event(kind,cid,stamp,data,source=None):
    return dict(kind=kind,can_id=cid,mono_ns=stamp,data_hex=data.hex(),source_mono_ns=source)


def send(timeline,cid,stamp,data,source=None):
    timeline.ingest(event('follower_target_tx_accepted',cid,stamp,data,source))
    return timeline.ingest(event('can0_tx_echo',cid,stamp+1,data,source))


def ready_timeline(stamp):
    timeline = Timeline()
    send(timeline,0x151,stamp-1_000_000_000,bytes.fromhex('010164ad00000000'))
    for idx,cid in enumerate((0x155,0x156,0x157)):
        send(timeline,cid,stamp-900_000_000+idx,struct.pack('>ii',idx*1000,-idx*1000),42)
    send(timeline,0x159,stamp-800_000_000,struct.pack('>iHBB',20000,1000,1,0),43)
    for idx,cid in enumerate((0x2A5,0x2A6,0x2A7,0x2A8)):
        payload = struct.pack('>iHBB',32000,0,64,0) if cid==0x2A8 else bytes(8)
        timeline.ingest(event('follower_rx',cid,stamp-5_000_000+idx,payload))
    return timeline


def image(sequence,stamp):
    return dict(image=np.zeros((20,24,3),dtype=np.uint8),sequence=sequence,
                host_monotonic_ns=stamp)


class CaptureChecks(unittest.TestCase):
    def test_high_follow_and_holds_keep_true_echo_timestamp(self):
        now = 3_000_000_000
        timeline = ready_timeline(now)
        state,action = timeline.sample_at(now)
        self.assertEqual(state['gripper_mm'],32)
        self.assertEqual(action['gripper_mm'],20)
        self.assertAlmostEqual(action['joints_rad'][4],2*np.pi/180)
        self.assertEqual(action['sample_monotonic_ns'],now-800_000_000+1)
        cameras = {'external':image(1,now-10_000_000),'wrist':image(1,now)}
        self.assertEqual(valid_sample(state,action,cameras,now),[])

    def test_unconfirmed_or_cross_group_targets_cannot_replace_action(self):
        now = 3_000_000_000
        timeline = ready_timeline(now)
        previous = timeline.sample_at(now)[1]['joints_rad']
        for cid,source in ((0x155,50),(0x156,51),(0x157,50)):
            send(timeline,cid,now-1_000_000,struct.pack('>ii',9000,9000),source)
        timeline.ingest(event('follower_target_tx_accepted',0x159,now-10,bytes(8),60))
        action = timeline.sample_at(now)[1]
        self.assertEqual(action['joints_rad'],previous)
        self.assertEqual(action['gripper_mm'],20)

    def test_future_echo_and_missing_feedback_are_rejected_at_anchor(self):
        now = 3_000_000_000
        timeline = ready_timeline(now)
        send(timeline,0x159,now+5_000_000,struct.pack('>iHBB',15000,1000,1,0),60)
        self.assertEqual(timeline.sample_at(now)[1]['gripper_mm'],20)
        self.assertIsNone(timeline.sample_at(now+100_000_000))

    def test_tail_preserves_incomplete_line_until_next_read(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = directory/'can_timeline.jsonl'
            line = json.dumps(event('follower_rx',0x2A8,1,bytes(8)))+'\n'
            path.write_text(line[:-2])
            tail = TimelineTail(directory)
            try:
                self.assertEqual(list(tail.read()),[])
                with path.open('a') as out:
                    out.write(line[-2:])
                self.assertEqual(len(list(tail.read())),1)
            finally:
                tail.close()

    def test_pairs_are_ordered_unique_and_within_the_episode(self):
        frames = {'external':[image(1,1),image(2,35_000_000),image(3,70_000_000)],
                  'wrist':[image(1,20),image(2,40_000_000),image(3,71_000_000)]}
        pairs = list(paired_frames(frames,1000,73_000_000,{}))
        self.assertEqual([(p['external']['sequence'],p['wrist']['sequence']) for p in pairs],[(2,2),(3,3)])

    def test_written_episode_has_images_command_trace_and_valid_manifest(self):
        now = 3_000_000_000
        timeline = ready_timeline(now)
        with tempfile.TemporaryDirectory() as folder:
            writer = EpisodeWriter(dict(purpose='commissioning',task='offline_fixture',
                                         initial_condition_id='fixture',started_monotonic_ns=now-30_000_000),Path(folder))
            for n,stamp in enumerate((now,now+30_000_000),1):
                row = event('follower_rx',0x2A8,stamp,bytes(8))
                writer.append_raw(row)
                state,cmd = timeline.sample_at(stamp)
                camera = {role:image(n,stamp) for role in ('external','wrist')}
                self.assertTrue(writer.append(state,cmd,camera,stamp))
            result = writer.finish('success',now+60_000_000)
            self.assertEqual(audit_episode(result)['samples'],2)
            summary = json.loads((result/'summary.json').read_text())
            self.assertTrue(summary['sampling_checks_pass'])
            self.assertFalse(summary['training_candidate'])

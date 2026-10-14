"""Frozen R1-R4 follow-up; no training queue and no algorithm modifications."""
import csv, hashlib, json, os
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())

def write(p,x,immutable=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if immutable and p.exists():
        assert read(p)==x,('FROZEN_MISMATCH',str(p));return
    t=p.with_name(p.name+'.tmp')
    with t.open('w',encoding='utf-8') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(t,p)

def table(p,rows):
    p=Path(p);t=p.with_name(p.name+'.tmp')
    with t.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(t,p)

def save(p,**a):
    p=Path(p);t=p.with_name(p.name+'.tmp.npz')
    with t.open('wb') as f:np.savez_compressed(f,**a);f.flush();os.fsync(f.fileno())
    os.replace(t,p)

def normalized(a):return a/np.linalg.norm(a,axis=-1,keepdims=True)

PROTOCOL=dict(experiment='abcr_real_buffer_validation_v1',suite='OBJECT',stages=[7,8,9],seeds=[42,43,44],
    r1_pool='Exact prior audited native 50% incremental-window buffer; seed42 surviving A1 transaction selection verified identical to native; seeds43/44 deterministic native reconstruction, not claimed surviving baseline snapshots',
    anchor='terminal frame min(window_start+7,trajectory_length-1), matching native BC terminal output; identity from exact full8x7 float32 action-window bytes, ambiguous excluded and counted',
    thresholds='Reuse previous abcr_object_pilots_v1 stage beta0.5 thresholds and base0..5 normalization unchanged. NO recalibration on smaller buffer.',
    descriptor='pre-FiLM frozen descriptor of unaugmented raw terminal frame. Cached training H remains original augmented feature. Different representations disclosed.',
    labels='SHARE/BIF are q4 normalized command-distance labels, not certified optimization or semantic relations',
    time='obs[t] is after action[t]; initial q1=action[t+1], initial q4=actions[t+1:t+5]; common independent future=actions[t+5:t+9]',
    physical=dict(eef_m=.03,rotation_degrees=15.,gripper_fraction=.10,gripper_width_m=.08,object_relative_proxy_m=.05),
    r2='All candidate pairs physical scan independent of descriptor score/actions. Object-relative 5cm is a separately reported geometric proxy, NOT semantic truth. AI visual audit seed42, each stage/class: up to8 physical-PASS and4 physical-FAIL pairs, one per trajectory pair before repeats; fixed hash order; blinded to class, score and physical outcome.',
    semantic='AI labels COMPARABLE/NOT_COMPARABLE/UNCERTAIN from two camera views, target language and local target/eef interaction phase. Different object identity alone is not rejection. No human-ground-truth claim.',
    r3='Same q8-valid comparable pair cohort for q1 and q4; unchanged tau+/tau-. Test later q4 BIF rate conditional initial SHARE and later SHARE rate conditional initial BIF. Also physical-PASS subset; all tail exclusions reported. Non-overlap is a surrogate, not semantic truth or proof of optimal q.',
    r4='Use earliest directly saved complete seed45 native MLR AdamW anchor for task7 and task8 from actual_update_fidelity_v1. No previous-batch replay. Reuse R2 AI-COMPARABLE physical-PASS pairs if exact physical window is uniquely present in both current and retained-old source plan for seed45; do not replenish unavailable pairs. Task9 not measured without a complete optimizer state.',
    r4_update='Single new-window official terminal BC NLL; full trainable decoder+head, native clip/AdamW/LR/weight decay and restored moments. Deterministic eval removes dropout for every gradient and loss. Restore full local state per pair. This is controlled new-only BC probe, not submitted MLR+IFA training step.',
    r4_outcomes='Gradient cosine; old-gradient dot actual FP32 AdamW displacement; old/new finite NLL change; 32*float32eps*max(1,abs(base_loss)) neutral band. No selecting pairs by gradients or losses.',
    statistics='Report each seed/stage; pair reuse creates dependence. R3 bootstrap by current trajectory (2000,seed42), R2 stratified audit counts, R4 pair/trajectory and limited two-anchor evidence; no pooled independent-sample claim.',
    random_seed=42,training_active=False,formal_training_resumed=False,abcr_training=False,p4_training=False)

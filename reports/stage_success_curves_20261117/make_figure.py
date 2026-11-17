"""Plot accepted formal outcomes only; no scientific execution or source edits."""
from pathlib import Path
import csv
import hashlib
import json
import statistics as st
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'figures'
DATA = HERE / 'data'
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
MANIFEST = {}
SEEDS = [41, 42, 43]
SUITES = ['OBJECT', 'GOAL']
METHODS = ['MLR_IFA', 'C', 'Q', 'S']
LABELS = {'MLR_IFA': 'MLR + IFA', 'C': 'CADL',
          'Q': 'CADL + SHBR w/o Gating', 'S': 'DEBR (Ours)'}
STYLES = {'MLR_IFA': ('#707784', 's', '--'), 'C': ('#2869AD', 'D', '-.'),
          'Q': ('#C48419', '^', ':'), 'S': ('#007F75', 'o', '-')}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def bound(p, expected=None):
    p = Path(p)
    if not p.is_absolute():
        p = ROOT / p
    h = sha(p)
    if expected is not None:
        assert h == expected, str(p)
    MANIFEST[p.relative_to(ROOT).as_posix()] = h
    return p

def read(p, expected=None):
    return json.loads(bound(p, expected).read_text(encoding='utf8'))

def write_json(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf8')

def csv_write(name, rows):
    with (DATA / name).open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def indexed(rows, index_name='initial_index'):
    d = {(int(r['stage']), int(r['task']), int(r[index_name])): r for r in rows}
    expected = {(s,t,i) for s in range(6,10) for t in range(6,s+1) for i in range(20)}
    assert len(rows) == len(d) == 200 and set(d) == expected
    return d

def matrix(rows, field, index_name='initial_index'):
    d = indexed(rows, index_name)
    assert all(type(r[field]) is bool for r in rows)
    return {f'{s}_{t}': 5 * sum(d[s,t,i][field] for i in range(20))
            for s in range(6,10) for t in range(6,s+1)}

def metrics(a):
    return dict(AUC=st.mean(st.mean(a[f'{s}_{t}'] for s in range(t,10)) for t in range(6,10)),
                FWT=st.mean(a[f'{t}_{t}'] for t in range(6,10)),
                NBT=st.mean(st.mean(a[f'{t}_{t}']-a[f'{s}_{t}'] for s in range(t+1,10)) for t in range(6,9)),
                OldFinal=st.mean(a[f'9_{t}'] for t in range(6,9)),
                FinalAll=st.mean(a[f'9_{t}'] for t in range(6,10)))

def check_metrics(a, reference):
    for k, v in metrics(a).items():
        ref = reference[k]
        if isinstance(ref, dict):
            ref = ref['percent']
        assert abs(v-ref) < 1e-8, (k, v, ref)

BIND = read('execution/20261109_mlr_ifa_align_baseline/BASELINE_BINDING.json')
BASE = read('execution/20261109_mlr_ifa_align_baseline/BASELINE_OUTCOMES.json')
assert BASE['status'] == 'READONLY_ACCEPTED_RANDOM_1200_OUTCOMES'
QDIR = ROOT / 'execution/20261031_full_q_vs_latest_s_two_suites/final_readonly_20261101_032229'
Q = read(QDIR / 'PRIMARY_S_COMPARISON.json')
QA = read(QDIR / 'LOCAL_FINAL_AUDIT.json', Q['source_Q_final_sha256'])
assert QA['status'] == 'PASS_COMPLETE_LOCAL_ACCEPTANCE'
SDIRS = {
    'OBJECT': ROOT / 'execution/20261028_support_gate_full_s_object/final_readonly_20261029_201235',
    'GOAL': ROOT / 'execution/20261029_support_gate_full_s_goal/final_readonly_20261030_111546',
}
CDIR = ROOT / 'execution/20261027_c_goal42_43_object41_42_43/final_readonly_20261027_222119'
C41DIR = ROOT / 'execution/20261025_ccbp_v3_full/final_readonly_20261025_181226'
CA = read(CDIR / 'LOCAL_FINAL_AUDIT.json')
C41A = read(C41DIR / 'LOCAL_FINAL_AUDIT.json')
assert CA['status'] == C41A['status'] == 'PASS_COMPLETE_LOCAL_ACCEPTANCE'
bound(CDIR / 'EVIDENCE.json', CA['source_sha256'])
bound(C41DIR / 'EVIDENCE.json', C41A['evidence_sha256'])

M = {}
episodes = []
for suite in SUITES:
    sd = SDIRS[suite]
    sa = read(sd / 'LOCAL_FINAL_AUDIT.json', Q['source_S'][suite]['audit_sha256'])
    bound(sd / 'EVIDENCE.json', Q['source_S'][suite]['evidence_sha256'])
    assert sa['status'] == 'PASS_COMPLETE_LOCAL_ACCEPTANCE'
    c = read(sd / 'PRIMARY_C_COMPARISON.json')
    assert c['status'] == 'PASS_ACCEPTED_C_S_600_EPISODE_PAIRING'
    for seed in SEEDS:
        case = f'{suite}_{seed}'
        # Baseline binding preserves the manuscript's accepted OBJECT seed43 repeat1.
        b = BIND['cases'][case]
        parent = b['parent_light_root']
        accpath = parent + '/INDEPENDENT_ACCEPTANCE.json'
        acc = read(accpath, BIND['source_sha256'][accpath])
        assert acc['status'].startswith('PASS')
        raw = []
        for p, h in BIND['source_sha256'].items():
            if p.startswith(parent + '/') and '/random/evaluation/trials_stage' in p:
                raw += [json.loads(x) for x in bound(p, h).read_text(encoding='utf8').splitlines() if x]
        rawidx = indexed(raw, 'initial_state_index')
        baseidx = indexed(BASE['cases'][case], 'initial_state_index')
        for k in rawidx:
            assert all(rawidx[k][f] == baseidx[k][f] for f in ['success','seed','initial_state_sha256'])
        M[suite,seed,'MLR_IFA'] = matrix(raw, 'success', 'initial_state_index')
        assert M[suite,seed,'MLR_IFA'] == b['baseline_matrix']
        check_metrics(M[suite,seed,'MLR_IFA'], b['baseline_metrics'])

        cr = [r for r in c['paired_outcomes'] if int(r['seed']) == seed]
        qr = Q['cases'][case]['paired_outcomes']
        ci, qi = indexed(cr), indexed(qr)
        for k in qi:
            assert ci[k]['S_success'] == qi[k]['S']
            assert all(ci[k][f] == qi[k][f] for f in ['initial_sha256','noise_sha256'])
            assert rawidx[k]['initial_state_sha256'] == qi[k]['initial_sha256']
        for method, rows, field, idx in [
            ('MLR_IFA', raw, 'success', 'initial_state_index'),
            ('C', cr, 'C_success', 'initial_index'),
            ('Q', qr, 'Q', 'initial_index'), ('S', qr, 'S', 'initial_index')]:
            a = matrix(rows, field, idx)
            M[suite,seed,method] = a
            if method == 'C':
                ref = C41A['metrics']['C'] if case == 'GOAL_41' else CA['metrics'][case]
                check_metrics(a, ref)
                check_metrics(a, c['seeds'][str(seed)]['C_metrics'])
            elif method in ['Q','S']:
                check_metrics(a, Q['cases'][case][method + '_metrics'])
                if method == 'S':
                    assert a == {k:v['percent'] for k,v in sa['seeds'][str(seed)]['matrix'].items()}
            for r in rows:
                episodes.append(dict(suite=suite,method=method,seed=seed,stage=r['stage'],task=r['task'],
                    initial_index=r[idx],success=int(r[field])))

curves = []
per_seed = []
cells = []
metric_rows = []
for suite in SUITES:
    for method in METHODS:
        for seed in SEEDS:
            a = M[suite,seed,method]
            metric_rows.append(dict(suite=suite,method=method,seed=seed,**metrics(a)))
            for s in range(6,10):
                per_seed.append(dict(suite=suite,method=method,seed=seed,stage=s,
                    mean_success_percent=st.mean(a[f'{s}_{t}'] for t in range(6,s+1))))
                for t in range(6,s+1):
                    cells.append(dict(suite=suite,method=method,seed=seed,stage=s,task=t,
                        successes=int(a[f'{s}_{t}']/5),episodes=20,success_percent=a[f'{s}_{t}']))
        for s in range(6,10):
            v = [st.mean(M[suite,seed,method][f'{s}_{t}'] for t in range(6,s+1)) for seed in SEEDS]
            curves.append(dict(suite=suite,method=method,stage=s,mean=st.mean(v),sample_sd=st.stdev(v),
                **{f'seed{seed}':v[j] for j,seed in enumerate(SEEDS)}))

# Guard against accidentally using ALIGN or budget-matched C instead of manuscript methods.
expected = {'OBJECT': {'MLR_IFA':(68.3,66.7,-2.8),'C':(85.0,81.2,-8.1),'Q':(84.6,78.8,-10.9),'S':(86.5,80.4,-11.0)},
            'GOAL': {'MLR_IFA':(76.4,76.7,0.2),'C':(82.5,85.0,5.6),'Q':(83.5,85.8,4.4),'S':(87.3,89.6,5.3)}}
metric_summary = []
for suite in SUITES:
    for method in METHODS:
        rows = [r for r in metric_rows if r['suite']==suite and r['method']==method]
        for metric in ['AUC','FWT','NBT','OldFinal','FinalAll']:
            v = [r[metric] for r in rows]
            metric_summary.append(dict(suite=suite,method=method,metric=metric,mean=st.mean(v),sample_sd=st.stdev(v)))
            if metric in ['AUC','FWT','NBT']:
                assert abs(st.mean(v)-expected[suite][method][['AUC','FWT','NBT'].index(metric)]) <= .051
csv_write('formal_outcomes.csv', episodes)
csv_write('stage_task_success.csv', cells)
csv_write('stage_success_per_seed.csv', per_seed)
csv_write('stage_success_summary.csv', curves)
csv_write('metrics_per_seed.csv', metric_rows)
csv_write('metrics_summary.csv', metric_summary)

for name in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
    font_manager.fontManager.addfont(str(Path('C:/Windows/Fonts') / name))
plt.rcParams.update({'font.family':'Times New Roman','font.size':12,'axes.labelsize':13,
    'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,
    'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
    'savefig.facecolor':'white','axes.unicode_minus':True})
fig, axes = plt.subplots(1,2,figsize=(11.6,5.25),sharey=True)
fig.subplots_adjust(left=.082,right=.978,top=.96,bottom=.30,wspace=.19)
handles = []
for j,(suite,ax) in enumerate(zip(SUITES,axes)):
    for method in METHODS:
        rows = [r for r in curves if r['suite']==suite and r['method']==method]
        x = np.array([r['stage'] for r in rows])
        mean = np.array([r['mean'] for r in rows])
        sd = np.array([r['sample_sd'] for r in rows])
        color, marker, linestyle = STYLES[method]
        assert np.all(mean-sd >= 0) and np.all(mean+sd <= 100), (suite,method)
        ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.095,linewidth=0,zorder=1)
        line, = ax.plot(x,mean,color=color,linestyle=linestyle,marker=marker,
            linewidth=2.4 if method=='S' else 1.8,markersize=6.5,
            markerfacecolor=color if method=='S' else 'white',markeredgewidth=1.4,
            label=LABELS[method],zorder=5 if method=='S' else 3)
        if j == 0:
            handles.append(line)
    ax.set(xlim=(5.88,9.12),ylim=(0,100),xticks=[6,7,8,9],
           xticklabels=['Task 6','Task 7','Task 8','Task 9'],yticks=range(0,101,20),xlabel='Completed training stage')
    ax.grid(axis='y',color='#DDE2E7',linewidth=.7,zorder=0)
    ax.tick_params(axis='both',length=4,width=.8)
    ax.text(.5,-.25,f'({chr(97+j)}) LIBERO-{suite}',transform=ax.transAxes,
            ha='center',va='top',fontsize=14)
axes[0].set_ylabel('Mean success rate on seen tasks (%)',labelpad=9)
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.015),ncol=4,
           frameon=False,columnspacing=1.75,handlelength=2.5,handletextpad=.65)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
assert all(t.get_fontproperties().get_name()=='Times New Roman' for t in texts)
outside = [t.get_text() for t in texts if any(not fig.bbox.contains(*p) for p in t.get_window_extent(renderer).get_points())]
assert not outside, outside
for ext in ['svg','png','pdf']:
    fig.savefig(OUT / f'DEBR_stage_success_curves.{ext}',dpi=600)
fig.savefig(OUT / 'preview.png',dpi=170)
svg = OUT / 'DEBR_stage_success_curves.svg'
svg.write_text('\n'.join(x.rstrip() for x in svg.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
svgroot = ET.parse(svg).getroot()
assert len(svgroot.findall('.//{http://www.w3.org/2000/svg}text')) > 20
assert len(svgroot.findall('.//{http://www.w3.org/2000/svg}image')) == 0
assert len(episodes)==4800 and len(cells)==240 and len(curves)==32
assert all(sha(ROOT/p)==h for p,h in MANIFEST.items())
write_json(HERE / 'SOURCE_MANIFEST.json', MANIFEST)
write_json(HERE / 'DATA_LAYOUT_AUDIT.json', dict(status='PASS_DATA_AND_LAYOUT',
    accepted_runs=24,formal_outcomes=4800,cells=240,stage_means=32,seeds=SEEDS,
    trials_per_cell=20,formal_initial_indices=list(range(20)),
    all_original_baseline_trial_files_checked=60,baseline_binding_matches_original_records=True,
    C_Q_S_paired_initial_and_noise_verified=True,baseline_initial_states_verified=True,
    baseline_action_noise_pairing_claimed=False,standard_budget_C=True,
    manuscript_AUC_FWT_NBT_mean_rounding_matches=True,
    variability='Sample standard deviation across three seed-level means (ddof=1), not CI',
    ordinate='For each seed/stage, arithmetic mean over Task6 through that stage; then seed mean',
    font='Times New Roman',layout='Horizontal 1x2',panel_titles='Below respective panels',
    legend='Below panels',ylim=[0,100],SD_bands_clipped=False,smoothed=False,
    svg_text_editable=True,svg_raster_images=0,text_outside_canvas=outside,
    sources_unchanged=True,new_training=0,new_model_forward=0,new_environment_episodes=0))
print('PASS_DATA_AND_LAYOUT: 4800 accepted outcomes; 24 runs; 32 stage means; no new experiments.')
for suite in SUITES:
    for method in METHODS:
        vals = [r['mean'] for r in curves if r['suite']==suite and r['method']==method]
        print(suite,method,','.join(f'{v:.2f}' for v in vals))

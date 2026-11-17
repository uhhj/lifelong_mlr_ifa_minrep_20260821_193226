"""Stage-wise differences from accepted formal outcomes; no experimental calls."""
from pathlib import Path
from fractions import Fraction
import csv
import hashlib
import json
import math
import statistics as st
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE.parent / '20261117_stage_success_curves'
OUT = HERE / 'figures'
DATA = HERE / 'data'
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
SOURCES = {}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    SOURCES[p.relative_to(ROOT).as_posix()] = sha(p)
    return json.loads(p.read_text(encoding='utf8'))

def read_csv(p):
    SOURCES[p.relative_to(ROOT).as_posix()] = sha(p)
    with p.open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(name, rows):
    with (DATA/name).open('w',encoding='utf8',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def write_json(name, obj):
    (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

audit = read(BASE/'DATA_LAYOUT_AUDIT.json')
assert audit['status']=='PASS_DATA_AND_LAYOUT' and audit['formal_outcomes']==4800
manifest = read(BASE/'SOURCE_MANIFEST.json')
assert all(sha(ROOT/p)==h for p,h in manifest.items())
published = read(ROOT/'publish_mlr_branch/reports/stage_success_curves_20261117/PUBLICATION_MANIFEST.json')
for name in ['formal_outcomes.csv','stage_success_per_seed.csv','stage_success_summary.csv']:
    assert sha(BASE/'data'/name)==published['data/'+name]['sha256']
rows = read_csv(BASE/'data/formal_outcomes.csv')
index = {(r['suite'],r['method'],int(r['seed']),int(r['stage']),int(r['task']),int(r['initial_index'])):int(r['success']) for r in rows}
suites = ['OBJECT','GOAL']
seeds = [41,42,43]
methods = ['C','Q','S']
expected = {(su,m,se,s,t,i) for su in suites for m in ['MLR_IFA',*methods] for se in seeds
            for s in range(6,10) for t in range(6,s+1) for i in range(20)}
assert len(index)==len(rows)==4800 and set(index)==expected and set(index.values())=={0,1}
# Compute all stage means directly from binary accepted records, with exact fractions.
absolute = {}
for su in suites:
    for m in ['MLR_IFA',*methods]:
        for se in seeds:
            for s in range(6,10):
                successes = sum(index[su,m,se,s,t,i] for t in range(6,s+1) for i in range(20))
                absolute[su,m,se,s] = Fraction(100*successes,20*(s-5))
for r in read_csv(BASE/'data/stage_success_per_seed.csv'):
    key = (r['suite'],r['method'],int(r['seed']),int(r['stage']))
    assert abs(float(absolute[key])-float(r['mean_success_percent']))<1e-10
old = {(r['suite'],r['method'],int(r['stage'])):float(r['mean'])
       for r in read_csv(BASE/'data/stage_success_summary.csv')}

seed_rows = []
summary = []
for su in suites:
    for m in methods:
        for s in range(6,10):
            differences = []
            for se in seeds:
                method_value = absolute[su,m,se,s]
                baseline_value = absolute[su,'MLR_IFA',se,s]
                d = method_value-baseline_value
                differences.append(float(d))
                seed_rows.append(dict(suite=su,method=m,seed=se,stage=s,
                    method_success_percent=float(method_value),baseline_success_percent=float(baseline_value),
                    delta_pp=float(d),delta_fraction=str(d)))
            mean = st.mean(differences)
            sd = st.stdev(differences)
            assert abs(mean-(old[su,m,s]-old[su,'MLR_IFA',s]))<1e-10
            summary.append(dict(suite=su,method=m,stage=s,mean_delta_pp=mean,sample_sd_delta_pp=sd,
                **{f'seed{se}':differences[j] for j,se in enumerate(seeds)}))
write_csv('delta_per_seed.csv',seed_rows)
write_csv('delta_summary.csv',summary)

for n in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
    font_manager.fontManager.addfont(str(Path('C:/Windows/Fonts')/n))
plt.rcParams.update({'font.family':'Times New Roman','font.size':12,'axes.labelsize':13,
    'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':11.8,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,
    'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
style = {'C':('#2869AD','D','-.'),'Q':('#C48419','^',':'),'S':('#007F75','o','-')}
labels = {'C':'CADL','Q':'CADL + SHBR w/o Gating','S':'DEBR (Ours)'}
low = math.floor(min(r['mean_delta_pp']-r['sample_sd_delta_pp'] for r in summary)/5)*5
high = math.ceil(max(r['mean_delta_pp']+r['sample_sd_delta_pp'] for r in summary)/5)*5
assert (low,high)==(-15,35)
fig,axes = plt.subplots(1,2,figsize=(11.6,5.25),sharey=True)
fig.subplots_adjust(left=.08,right=.978,top=.96,bottom=.30,wspace=.19)
handles = [Line2D([],[],color='#747B85',linestyle='--',linewidth=1.2,label='MLR + IFA (reference)')]
for j,(suite,ax) in enumerate(zip(suites,axes)):
    ax.axhline(0,color='#747B85',linestyle='--',linewidth=1.2,zorder=2)
    for m in methods:
        rr = [r for r in summary if r['suite']==suite and r['method']==m]
        x = np.array([r['stage'] for r in rr])
        y = np.array([r['mean_delta_pp'] for r in rr])
        sd = np.array([r['sample_sd_delta_pp'] for r in rr])
        color,marker,linestyle = style[m]
        ax.fill_between(x,y-sd,y+sd,color=color,alpha=.08,linewidth=0,zorder=1)
        line, = ax.plot(x,y,color=color,linestyle=linestyle,marker=marker,
            linewidth=2.5 if m=='S' else 1.8,markersize=6.5,
            markerfacecolor=color if m=='S' else 'white',markeredgewidth=1.4,
            label=labels[m],zorder=5 if m=='S' else 3)
        if j==0:
            handles.append(line)
    ax.set(xlim=(5.88,9.12),ylim=(low,high),xticks=[6,7,8,9],
           xticklabels=['Task 6','Task 7','Task 8','Task 9'],yticks=[-10,0,10,20,30],
           xlabel='Completed training stage')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p: '0' if v==0 else f'{v:+.0f}'))
    ax.grid(axis='y',color='#E1E5E9',linewidth=.7,zorder=0)
    ax.tick_params(axis='both',length=4,width=.8)
    ax.text(.5,-.25,f'({chr(97+j)}) LIBERO-{suite}',transform=ax.transAxes,
            ha='center',va='top',fontsize=14)
axes[0].set_ylabel('Δ success rate vs. MLR + IFA (pp)',labelpad=10)
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.015),ncol=4,
    frameon=False,columnspacing=1.4,handlelength=2.4,handletextpad=.55)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
assert all(t.get_fontproperties().get_name()=='Times New Roman' for t in texts)
outside = [t.get_text() for t in texts if any(not fig.bbox.contains(*p) for p in t.get_window_extent(renderer).get_points())]
assert not outside,outside
for ext in ['svg','png','pdf']:
    fig.savefig(OUT/f'DEBR_stage_success_delta.{ext}',dpi=600)
fig.savefig(OUT/'preview.png',dpi=170)
svg = OUT/'DEBR_stage_success_delta.svg'
svg.write_text('\n'.join(x.rstrip() for x in svg.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
tree = ET.parse(svg).getroot()
assert len(tree.findall('.//{http://www.w3.org/2000/svg}text'))>20
assert not tree.findall('.//{http://www.w3.org/2000/svg}image')
assert all(sha(ROOT/p)==h for p,h in {**manifest,**SOURCES}.items())
write_json('SOURCE_MANIFEST.json',{**manifest,**SOURCES})
write_json('DATA_LAYOUT_AUDIT.json',dict(status='PASS_DATA_AND_LAYOUT',
    accepted_outcomes_recomputed=4800,seed_level_differences=72,mean_difference_points=24,
    reference='MLR + IFA for the same suite, seed and completed stage',units='percentage points',
    variability='Sample SD of the three seed-wise differences (ddof=1), not differences of SDs or a CI',
    baseline_random_stream_pairing_claimed=False,mean_deltas_equal_difference_of_absolute_means=True,
    y_range=[low,high],zero_reference_visible=True,sd_bands_clipped=False,smoothed=False,
    common_y_scale=True,font='Times New Roman',layout='1x2 horizontal',titles_and_legend='Below panels',
    text_outside_canvas=outside,svg_text_editable=True,svg_raster_images=0,
    original_absolute_figure_preserved=True,sources_unchanged=True,
    new_training=0,new_environment_episodes=0,new_model_forward=0))
print('PASS_DATA_AND_LAYOUT: 4800 outcomes, 72 seed deltas, 24 mean deltas; full SD bands retained.')
for suite in suites:
    for m in methods:
        print(suite,m,[round(r['mean_delta_pp'],3) for r in summary if r['suite']==suite and r['method']==m])

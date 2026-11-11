"""Horizontal Times New Roman heatmaps, recomputed from paired outcomes."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
EXEC = HERE.parent
OUT = HERE/'figures'
OUT.mkdir(exist_ok=True)
manifest = {}
def read(p):
    manifest[str(p.relative_to(EXEC))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text(encoding='utf8'))
sources = {
    'C': read(EXEC/'20261030_equal_optimizer_budget_cs_two_suites/final_readonly_20261031_065009/PRIMARY_S_COMPARISON.json'),
    'Q': read(EXEC/'20261031_full_q_vs_latest_s_two_suites/final_readonly_20261101_032229/PRIMARY_S_COMPARISON.json'),
}
csvpath = EXEC/'20261107_main_text_figures/data/stage_task_differences.csv'
manifest[str(csvpath.relative_to(EXEC))] = hashlib.sha256(csvpath.read_bytes()).hexdigest()
with csvpath.open(encoding='utf8', newline='') as f:
    original = {(r['suite'],r['comparison'],int(r['stage']),int(r['task'])):float(r['delta_pp']) for r in csv.DictReader(f)}
for name in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
    font_manager.fontManager.addfont(str(Path('C:/Windows/Fonts')/name))
plt.rcParams.update({'font.family':'Times New Roman','font.size':11,
                     'axes.titlesize':13,'axes.labelsize':11,
                     'xtick.labelsize':10,'ytick.labelsize':10,
                     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                     'axes.unicode_minus':True})
fig, axes = plt.subplots(1,4,figsize=(15.5,4.6))
fig.subplots_adjust(left=.045,right=.985,bottom=.37,top=.98,wspace=.27)
cmap = plt.get_cmap('RdBu').copy()
cmap.set_bad('white')
panels = []
for i,(suite, method) in enumerate([('OBJECT','C'),('OBJECT','Q'),('GOAL','C'),('GOAL','Q')]):
    a = np.full((4,4),np.nan)
    for stage in range(6,10):
        for task in range(6,stage+1):
            per_seed = []
            for seed in [41,42,43]:
                rr = [r for r in sources[method]['cases'][f'{suite}_{seed}']['paired_outcomes'] if r['stage']==stage and r['task']==task]
                assert len(rr)==20 and {r['initial_index'] for r in rr}==set(range(20))
                per_seed.append(5*sum(int(r['S'])-int(r[method]) for r in rr))
            v = float(np.mean(per_seed))
            assert abs(v-original[suite,f'S-{method}',stage,task]) < 1e-10
            a[stage-6,task-6] = v
    ax = axes[i]
    im = ax.imshow(a,cmap=cmap,vmin=-25,vmax=25,interpolation='nearest')
    for y in range(4):
        for x in range(y+1):
            v = a[y,x]
            ax.text(x,y,'0.0' if abs(v)<1e-8 else f'{v:+.1f}',ha='center',va='center',
                    fontsize=11,color='white' if abs(v)>=15 else 'black')
    ax.set(xticks=range(4),xticklabels=[6,7,8,9],yticks=range(4),yticklabels=[6,7,8,9],
           xlabel='Task',ylabel='Training stage')
    ax.text(.5,-.23,f'({chr(97+i)}) {suite}',transform=ax.transAxes,
            ha='center',va='top',fontsize=13)
    ax.set_xticks(np.arange(-.5,4,1),minor=True)
    ax.set_yticks(np.arange(-.5,4,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1)
    ax.tick_params(which='both',length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    panels.append(dict(panel=chr(97+i),suite=suite,comparison=f'S-{method}',
                       values=[[None if np.isnan(v) else float(v) for v in row] for row in a]))
cax = fig.add_axes([.30,.135,.40,.035])
cb = fig.colorbar(im,cax=cax,orientation='horizontal')
cb.set_ticks([-20,-10,0,10,20])
cb.set_label('Success-rate difference (percentage points)',labelpad=5)
cb.outline.set_linewidth(.6)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
assert all(t.get_fontproperties().get_name()=='Times New Roman' for t in texts)
outside = [t.get_text() for t in texts if any(not fig.bbox.contains(*point) for point in t.get_window_extent(renderer).get_points())]
assert not outside, outside
for ext in ['png','pdf','svg']:
    fig.savefig(OUT/f'task_gains_horizontal_times.{ext}',dpi=600,facecolor='white')
svg = OUT/'task_gains_horizontal_times.svg'
svg.write_bytes(('\n'.join(s.rstrip() for s in svg.read_text(encoding='utf8').splitlines())+'\n').encode('utf8'))
fig.savefig(OUT/'preview.png',dpi=150,facecolor='white')
assert all(hashlib.sha256((EXEC/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
audit = dict(status='PASS_DATA_AND_LAYOUT',panels=panels,original_40_cells_unchanged=True,
             paired_outcomes_recomputed=True,all_text_font_verified=True,font='Times New Roman',
             layout='1x4',panel_titles_position='below_each_panel',colorbar_position='below_all_panels',color_range=[-25,25],
             text_outside_canvas=outside,new_training_calls=0,new_environment_episodes=0)
(HERE/'DATA_LAYOUT_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
(HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
print('PASS: all 40 cells match original; Times New Roman; horizontal panels; bottom colorbar.')

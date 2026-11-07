"""Recompute and plot accepted scalar/outcome evidence. No model or environment use."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import platform
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, colors
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
EXEC = HERE.parent
OUT = HERE/'figures'
DATA = HERE/'data'
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
SUITES = ['OBJECT','GOAL']
SEEDS = [41,42,43]
METHODS = ['C','Q','S']
COLORS = {'C':'#64748B','Q':'#D97706','S':'#007F75'}
LABELS = {'C':'+Execution Supervision(Matched Budget)', 'Q':'+Support-Weighted Retention', 'S':'Ours'}
MARKERS = {'C':'s','Q':'^','S':'o'}
SOURCE_HASHES = {}
ARTISTS = []
LAYOUT_AUDITS = []

for name in ['arial.ttf','arialbd.ttf','ariali.ttf','arialbi.ttf']:
    p = Path('C:/Windows/Fonts')/name
    if p.exists(): font_manager.fontManager.addfont(str(p))
plt.rcParams.update({
    'font.family':'Arial','font.size':9,'axes.titlesize':9.5,'axes.labelsize':9,
    'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8.5,
    'axes.linewidth':.65,'lines.linewidth':1.5,'lines.markersize':4,
    'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':3,'ytick.major.size':3,
    'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'savefig.facecolor':'white','axes.unicode_minus':True,
    'mathtext.fontset':'dejavusans',
})

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    SOURCE_HASHES[str(p.relative_to(EXEC))] = sha(p)
    return json.loads(p.read_text(encoding='utf8'))
def write_csv(name, rows):
    if name not in {'success_curves.csv','stage_task_success.csv','stage_task_differences.csv','gate_condition_patterns.csv','method_metrics_per_seed.csv'}: return
    assert rows
    with (DATA/name).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

original=EXEC/'20261103_lambda_r_goal_sensitivity/publication_figures_20261106'
original_manifest=json.loads((original/'SOURCE_MANIFEST.json').read_text(encoding='utf8'))
assert all(sha(EXEC/p)==h for p,h in original_manifest.items())
SOURCE_HASHES[str((original/'SOURCE_MANIFEST.json').relative_to(EXEC))]=sha(original/'SOURCE_MANIFEST.json')

qpath = EXEC/'20261031_full_q_vs_latest_s_two_suites/final_readonly_20261101_032229/PRIMARY_S_COMPARISON.json'
cpath = EXEC/'20261030_equal_optimizer_budget_cs_two_suites/final_readonly_20261031_065009/PRIMARY_S_COMPARISON.json'
lpaths = {
    'OBJECT':EXEC/'20261101_lambda_r_object_sensitivity/final_acceptance_20261103_v1/final_readonly_20261103_124306/PRIMARY_S_COMPARISON.json',
    'GOAL':EXEC/'20261103_lambda_r_goal_sensitivity/final_readonly_20261104_122613/PRIMARY_S_COMPARISON.json',
}
gdirs = {
    'OBJECT':EXEC/'20261028_support_gate_full_s_object/final_readonly_20261029_201235',
    'GOAL':EXEC/'20261029_support_gate_full_s_goal/final_readonly_20261030_111546',
}
qcomp,ccomp=read(qpath),read(cpath)
for folder in [qpath.parent,cpath.parent,*[p.parent for p in lpaths.values()],*gdirs.values()]:
    read(folder/'LOCAL_FINAL_AUDIT.json')

def keyed(rows):
    result={(r['stage'],r['task'],r['initial_index']):r for r in rows}
    assert len(result)==len(rows)==200
    assert set(result)=={(s,t,i) for s in range(6,10) for t in range(6,s+1) for i in range(20)}
    return result

def matrix(rows,which):
    rr=keyed(rows);a=np.full((4,4),np.nan)
    for s in range(6,10):
        for t in range(6,s+1):
            a[s-6,t-6]=5*sum(bool(rr[s,t,i][which]) for i in range(20))
    return a

def metrics(a):
    auc=np.mean([np.mean(a[t:,t]) for t in range(4)])
    fwt=np.mean(np.diag(a))
    nbt=np.mean([np.mean(a[t,t]-a[t+1:,t]) for t in range(3)])
    return dict(AUC=auc,FWT=fwt,NBT=nbt,OldFinal=np.mean(a[3,:3]),NewFinal=a[3,3],FinalAll=np.mean(a[3]))

def check_metrics(actual,expected):
    for k,v in expected.items():
        if k in actual: assert abs(actual[k]-float(v))<1e-8,(k,actual[k],v)

M={};L={};episode_rows=[];metric_rows=[]
for suite in SUITES:
    for seed in SEEDS:
        key=f'{suite}_{seed}';qcase=qcomp['cases'][key];ccase=ccomp['cases'][key]
        qr=keyed(qcase['paired_outcomes']);cr=keyed(ccase['paired_outcomes'])
        for k in qr:
            assert all(qr[k][f]==cr[k][f] for f in ['S','initial_sha256','noise_sha256','S_cp_sha256'])
        for method,case in [('C',ccase),('Q',qcase),('S',qcase)]:
            a=matrix(case['paired_outcomes'],method);M[suite,seed,method]=a
            mm=metrics(a);check_metrics(mm,case[f'{method}_metrics'])
            metric_rows.append(dict(suite=suite,seed=seed,method=method,**mm))
            for r in case['paired_outcomes']:
                episode_rows.append(dict(suite=suite,method=method,seed=seed,stage=r['stage'],task=r['task'],initial_index=r['initial_index'],success=int(r[method]),initial_sha256=r['initial_sha256'],noise_sha256=r['noise_sha256']))
    lc=read(lpaths[suite])
    for seed in SEEDS:
        for tag,lam in [('L005',.05),('L020',.2)]:
            case=lc['cases'][f'{tag}_{seed}']
            a=matrix(case['paired_outcomes'],'Variant');L[suite,seed,lam]=metrics(a)
            check_metrics(L[suite,seed,lam],case['Variant_metrics'])
            base=matrix(case['paired_outcomes'],'S');assert np.allclose(base,M[suite,seed,'S'],equal_nan=True)
            rr=keyed(case['paired_outcomes']);sr=keyed(qcomp['cases'][f'{suite}_{seed}']['paired_outcomes'])
            assert all(all(rr[k][f]==sr[k][f] for f in ['S','S_cp_sha256','initial_sha256','noise_sha256']) for k in rr)
        L[suite,seed,.1]=metrics(M[suite,seed,'S'])
write_csv('method_outcomes.csv',episode_rows)
write_csv('method_metrics_per_seed.csv',metric_rows)
lambda_rows=[dict(suite=s,seed=k,lambda_R=l,**L[s,k,l]) for s in SUITES for l in [.05,.1,.2] for k in SEEDS]
write_csv('lambda_metrics_per_seed.csv',lambda_rows)

cells=[];curves=[];summary=[];lambda_summary=[]
for suite in SUITES:
    for method in METHODS:
        mm=[metrics(M[suite,seed,method]) for seed in SEEDS]
        summary.append(dict(suite=suite,method=method,**{k:float(np.mean([v[k] for v in mm])) for k in mm[0]}))
        for si,stage in enumerate(range(6,10)):
            for ti,task in enumerate(range(6,stage+1)):
                v=[M[suite,seed,method][si,ti] for seed in SEEDS]
                cells.append(dict(suite=suite,method=method,stage=stage,task=task,mean=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1)),**{f'seed{k}':v[j] for j,k in enumerate(SEEDS)}))
            for kind in ['All seen','Old tasks','New task']:
                if kind=='Old tasks' and si==0:continue
                v=[]
                for seed in SEEDS:
                    a=M[suite,seed,method]
                    v.append(float(np.mean(a[si,:si+1])) if kind=='All seen' else float(np.mean(a[si,:si])) if kind=='Old tasks' else float(a[si,si]))
                curves.append(dict(suite=suite,method=method,stage=stage,subset=kind,mean=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1)),**{f'seed{k}':v[j] for j,k in enumerate(SEEDS)}))
    for lam in [.05,.1,.2]:
        for metric in L[suite,41,lam]:
            v=[L[suite,k,lam][metric] for k in SEEDS]
            lambda_summary.append(dict(suite=suite,lambda_R=lam,metric=metric,mean=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1))))
write_csv('method_metrics_mean.csv',summary);write_csv('stage_task_success.csv',cells)
write_csv('success_curves.csv',curves);write_csv('lambda_metrics_summary.csv',lambda_summary)

def rel(q,c):
    den=abs(q)+abs(c)
    return 200*(q-c)/den if den else 0.

G={};gate_rows=[];gate_stage=[];patterns=[]
for suite in SUITES:
    allrows=[]
    for seed in SEEDS:
        count=0
        for stage in [7,8,9]:
            p=gdirs[suite]/f'light/seed{seed}/GATES_stage{stage}.jsonl'
            SOURCE_HASHES[str(p.relative_to(EXEC))]=sha(p)
            rows=[json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x]
            keys=set()
            for r in rows:
                assert r['stage']==stage
                key=(r['epoch'],r['batch']);assert key not in keys;keys.add(key)
                c,q=r['C'],r['Q']
                ck=dict(bc_new=q['bc_new']<=c['bc_new'],exec_new=q['exec_new']<=c['exec_new'],retain_improved=q['retain_support']<c['retain_support'])
                assert ck==r['checks'];accepted=all(ck.values());assert accepted==(r['accepted']=='Q')
                fail=(not ck['bc_new'])*1+(not ck['exec_new'])*2+(not ck['retain_improved'])*4
                d=dict(suite=suite,seed=seed,stage=stage,epoch=r['epoch'],batch=r['batch'],accepted_Q=int(accepted),failure_mask=fail,
                    delta_bc=q['bc_new']-c['bc_new'],delta_exec=q['exec_new']-c['exec_new'],delta_retain=q['retain_support']-c['retain_support'],
                    relative_bc_pct=rel(q['bc_new'],c['bc_new']),relative_exec_pct=rel(q['exec_new'],c['exec_new']),relative_retain_pct=rel(q['retain_support'],c['retain_support']))
                assert all(np.isfinite(d[k]) for k in ['delta_bc','delta_exec','delta_retain','relative_bc_pct','relative_exec_pct','relative_retain_pct'])
                allrows.append(d);gate_rows.append(d)
            count+=len(rows)
            gate_stage.append(dict(suite=suite,seed=seed,stage=stage,blocks=len(rows),accepted_Q=sum(r['accepted']=='Q' for r in rows),acceptance_pct=100*np.mean([r['accepted']=='Q' for r in rows])))
        assert count==({'OBJECT':9100,'GOAL':6700}[suite])
    G[suite]=allrows
    cc=Counter(r['failure_mask'] for r in allrows)
    for mask in range(8):patterns.append(dict(suite=suite,failure_mask=mask,blocks=cc[mask],percent=100*cc[mask]/len(allrows)))
write_csv('gate_blocks.csv',gate_rows);write_csv('gate_acceptance_per_stage_seed.csv',gate_stage);write_csv('gate_condition_patterns.csv',patterns)

def tidy(ax):
    ax.grid(axis='y',color='#DCE2E5',lw=.55,zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(pad=3)

def save(fig,name):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();box=fig.bbox
    outside=[]
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():continue
        bb=artist.get_window_extent(renderer)
        if bb.x0 < box.x0-2 or bb.x1 > box.x1+2 or bb.y0 < box.y0-2 or bb.y1 > box.y1+2:
            outside.append(artist.get_text())
    LAYOUT_AUDITS.append(dict(figure=name,text_outside_canvas=outside))
    assert not outside,(name,outside)
    for ext in ['pdf','svg','png']:
        options=dict(dpi=600)
        if ext=='pdf':options['metadata']=dict(Title=name,Author='MLR+IFA experiment analysis',Subject='Accepted scalar and episode evidence')
        fig.savefig(OUT/f'{name}.{ext}',**options)
    svg=OUT/f'{name}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
    fig.savefig(OUT/f'{name}_preview.png',dpi=180)
    ARTISTS.append(name);plt.close(fig)

# Main Figure 1: all evaluated incremental tasks and previously learned tasks.
fig,axes=plt.subplots(2,2,figsize=(7.2,5.25))
fig.subplots_adjust(left=.085,right=.98,bottom=.10,top=.77,wspace=.24,hspace=.49)
for ri,suite in enumerate(SUITES):
    for ci,kind in enumerate(['All seen','Old tasks']):
        ax=axes[ri,ci];tidy(ax)
        for method in METHODS:
            rr=[r for r in curves if r['suite']==suite and r['method']==method and r['subset']==kind]
            xx=np.array([r['stage'] for r in rr]);yy=np.array([r['mean'] for r in rr]);sd=np.array([r['sample_sd'] for r in rr])
            ax.fill_between(xx,yy-sd,yy+sd,color=COLORS[method],alpha=.12,linewidth=0)
            ax.plot(xx,yy,color=COLORS[method],marker=MARKERS[method],lw=1.7,ms=4.4,zorder=3)
        ax.set(xlim=(5.85,9.15),ylim=(0,105),xticks=[6,7,8,9],yticks=[0,25,50,75,100],xlabel='Training stage')
        if ci==0:ax.set_ylabel('Success rate (%)')
        title='All seen tasks' if ci==0 else 'Old tasks'
        ax.set_title(f'({chr(97+ri*2+ci)}) {suite} | {title}',loc='left',pad=8)
fig.legend(handles=[Line2D([],[],color=COLORS[m],marker=MARKERS[m],label=LABELS[m],lw=1.7) for m in METHODS],
           loc='upper center',bbox_to_anchor=(.52,.995),ncol=1,frameon=False,labelspacing=.42,handlelength=2.2)
fig.text(.52,.845,'Mean ± 1 SD across three seeds',ha='center',fontsize=8,color='#4B5563')
save(fig,'fig01_continual_performance')

# Main Figure 2: local candidate decisions, not independently trained methods.
fig,axes=plt.subplots(2,2,figsize=(7.2,5.5),gridspec_kw={'width_ratios':[1,1.12]})
fig.subplots_adjust(left=.105,right=.96,bottom=.105,top=.86,wspace=.58,hspace=.48)
reject='#D66A46';accept=COLORS['S'];order=[0,1,2,4,3,5,6,7]
names=['All pass','BC','Execution','Retention','BC + Execution','BC + Retention','Execution + Retention','All three']
for ri,suite in enumerate(SUITES):
    rr=G[suite];ax=axes[ri,0]
    for flag,color in [(0,reject),(1,accept)]:
        selected=[r for r in rr if r['accepted_Q']==flag]
        ax.scatter([r['relative_exec_pct'] for r in selected],[r['relative_retain_pct'] for r in selected],
                   s=2,c=color,alpha=.22,edgecolors='none',rasterized=True,zorder=2+flag)
    for axis in ['x','y']:
        getattr(ax,f'set_{axis}scale')('symlog',linthresh=.01,linscale=.65,base=10)
        getattr(ax,f'set_{axis}lim')(-210,210)
        getattr(ax,f'set_{axis}ticks')([-100,-1,0,1,100])
        getattr(ax,f'set_{axis}ticklabels')(['−100','−1','0','1','100'])
    ax.axhline(0,color='#6B7280',lw=.65,zorder=1);ax.axvline(0,color='#6B7280',lw=.65,zorder=1)
    ax.set_xlabel('Relative execution-error change (%)');ax.set_ylabel('Relative retention-error change (%)')
    ax.set_title(f'({chr(97+ri*2)}) {suite} | Candidate tradeoffs',loc='left',pad=8)
    ax.text(.035,.97,f'n = {len(rr):,} blocks',transform=ax.transAxes,va='top',fontsize=8,
            bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=2))
    ax=axes[ri,1];cc=Counter(r['failure_mask'] for r in rr);vals=[100*cc[m]/len(rr) for m in order]
    assert abs(sum(vals)-100)<1e-10
    ax.barh(range(8),vals,color=[accept]+[reject]*7,height=.67,zorder=3)
    ax.set_yticks(range(8),names,fontsize=7.8);ax.invert_yaxis()
    ax.set(xlim=(0,60),xticks=[0,20,40,60],xlabel='Share of all blocks (%)')
    ax.set_title(f'({chr(98+ri*2)}) {suite} | Violated conditions',loc='left',pad=8)
    ax.grid(axis='x',color='#DCE2E5',lw=.55);ax.set_axisbelow(True)
    for j,v in enumerate(vals):ax.text(v+.8,j,f'{v:.1f}',va='center',fontsize=8)
fig.legend(handles=[Line2D([],[],ls='none',marker='o',color=accept,label='Retention candidate accepted'),
                    Line2D([],[],ls='none',marker='o',color=reject,label='Retention candidate rejected')],
           loc='upper center',bbox_to_anchor=(.53,.997),ncol=2,frameon=False,columnspacing=1.6,handletextpad=.4)
fig.text(.53,.932,r'Ours ($\lambda_R=0.1$) | All decision blocks | Symmetric-log scatter axes',ha='center',fontsize=8,color='#4B5563')
save(fig,'fig02_gate_tradeoffs')

# Main Figure 3: preserve every signed task-stage difference.
diffs={(s,m):np.mean(np.stack([M[s,k,'S']-M[s,k,m] for k in SEEDS]),axis=0) for s in SUITES for m in ['C','Q']}
vmax=float(5*np.ceil(max(np.nanmax(np.abs(a)) for a in diffs.values())/5))
assert vmax==25
fig,axes=plt.subplots(2,2,figsize=(7.2,5.8))
fig.subplots_adjust(left=.105,right=.965,bottom=.18,top=.83,wspace=.28,hspace=.42)
dcmap=plt.get_cmap('RdBu').copy();dcmap.set_bad('#EDF0F2')
diffrows=[]
for ri,suite in enumerate(SUITES):
    for ci,method in enumerate(['C','Q']):
        ax=axes[ri,ci];a=diffs[suite,method]
        im=ax.imshow(a,cmap=dcmap,vmin=-vmax,vmax=vmax,interpolation='nearest')
        for y in range(4):
            for x in range(y+1):
                v=a[y,x];txt='0.0' if abs(v)<1e-8 else f'{v:+.1f}'
                ax.text(x,y,txt,ha='center',va='center',fontsize=9,color='white' if abs(v)>vmax*.62 else '#17212B')
                diffrows.append(dict(suite=suite,comparison=f'S-{method}',stage=y+6,task=x+6,delta_pp=v))
        ax.set(xticks=range(4),xticklabels=[6,7,8,9],yticks=range(4),yticklabels=[6,7,8,9],xlabel='Task',ylabel='Training stage')
        ax.set_title(f'({chr(97+ri*2+ci)}) {suite}',loc='left',pad=8)
        for sp in ax.spines.values():sp.set_visible(False)
        ax.tick_params(length=0)
for ci,title in enumerate(['Ours − Execution Supervision\n(Matched Budget)','Ours − Support-Weighted Retention\n(No Gate)']):
    pos=axes[0,ci].get_position()
    fig.text((pos.x0+pos.x1)/2,.96,title,ha='center',va='top',fontsize=9.5,linespacing=1.35)
cax=fig.add_axes([.24,.075,.58,.024]);cb=fig.colorbar(im,cax=cax,orientation='horizontal')
cb.set_label('Success-rate difference (percentage points)',labelpad=5);cb.set_ticks([-25,-10,0,10,25]);cb.outline.set_linewidth(.5)
write_csv('stage_task_differences.csv',diffrows)
save(fig,'fig03_task_level_gains')

# Compare unchanged numbers with the original figure package.
for name in ['success_curves.csv','stage_task_success.csv','stage_task_differences.csv','gate_condition_patterns.csv','method_metrics_per_seed.csv']:
    with (DATA/name).open(encoding='utf8',newline='') as f:current=list(csv.DictReader(f))
    with (original/'data'/name).open(encoding='utf8',newline='') as f:previous=list(csv.DictReader(f))
    assert current==previous,name
assert all(sha(EXEC/p)==h for p,h in SOURCE_HASHES.items())
audit=dict(status='PASS_RECOMPUTATION_AND_MAIN_TEXT_EXPORT',figures=ARTISTS,layout=LAYOUT_AUDITS,
           gate_blocks={s:len(G[s]) for s in SUITES},all_gate_decisions_recomputed=True,
           all_gate_points_retained=True,all_positive_zero_negative_differences_retained=True,
           original_plot_numeric_tables_exact=True,seeds=SEEDS,new_optimizer_calls=0,new_environment_episodes=0,
           method_labels=LABELS,scatter_raster_dpi=600,other_pdf_elements='vector',
           relative_delta='200*(retention_candidate-execution_candidate)/(abs(retention_candidate)+abs(execution_candidate)); zero if both zero',
           figure_width_inches=7.2,png_dpi=600)
(HERE/'DATA_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
(HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(SOURCE_HASHES,indent=2)+'\n',encoding='utf8')
print(json.dumps(audit,indent=2))

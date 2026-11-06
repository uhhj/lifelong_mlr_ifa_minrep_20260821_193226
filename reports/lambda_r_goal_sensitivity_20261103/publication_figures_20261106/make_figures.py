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
EXEC = HERE.parents[1]
OUT = HERE/'figures'
DATA = HERE/'data'
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
SUITES = ['OBJECT','GOAL']
SEEDS = [41,42,43]
METHODS = ['C','Q','S']
COLORS = {'C':'#64748B','Q':'#D97706','S':'#007F75'}
LABELS = {'C':r'C (matched optimizer budget)','Q':'Q (no gate)','S':'S (gated)'}
MARKERS = {'C':'s','Q':'^','S':'o'}
SOURCE_HASHES = {}
ARTISTS = []
LAYOUT_AUDITS = []

for name in ['arial.ttf','arialbd.ttf','ariali.ttf','arialbi.ttf']:
    p = Path('C:/Windows/Fonts')/name
    if p.exists(): font_manager.fontManager.addfont(str(p))
plt.rcParams.update({
    'font.family':'Arial','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
    'xtick.labelsize':7.3,'ytick.labelsize':7.3,'legend.fontsize':7.7,
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
    assert rows
    with (DATA/name).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

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
    fig.savefig(OUT/f'{name}_preview.png',dpi=170)
    ARTISTS.append(name);plt.close(fig)

# Figure 1: stage performance. Equal seed weighting; ribbons are sample SD.
fig,axes=plt.subplots(2,3,figsize=(7.5,4.8))
fig.subplots_adjust(left=.075,right=.985,bottom=.10,top=.86,wspace=.29,hspace=.52)
for ri,suite in enumerate(SUITES):
    for ci,kind in enumerate(['All seen','Old tasks','New task']):
        ax=axes[ri,ci];tidy(ax)
        for method in METHODS:
            rows=[r for r in curves if r['suite']==suite and r['method']==method and r['subset']==kind]
            xx=np.array([r['stage'] for r in rows]);yy=np.array([r['mean'] for r in rows]);sd=np.array([r['sample_sd'] for r in rows])
            ax.fill_between(xx,yy-sd,yy+sd,color=COLORS[method],alpha=.10,linewidth=0)
            ax.plot(xx,yy,color=COLORS[method],marker=MARKERS[method],label=LABELS[method],zorder=3)
        ax.set(xlim=(5.85,9.15),ylim=(0,105),xticks=[6,7,8,9],yticks=[0,25,50,75,100],xlabel='Training stage')
        if ci==0:ax.set_ylabel('Success rate (%)')
        ax.set_title(f'({chr(97+ri*3+ci)}) {suite} | {kind}',loc='left',pad=7)
fig.legend(handles=[Line2D([],[],color=COLORS[m],marker=MARKERS[m],label=LABELS[m]) for m in METHODS],loc='upper center',bbox_to_anchor=(.51,.99),ncol=3,frameon=False,columnspacing=1.6)
fig.text(.5,.913,'Mean ± 1 SD across three seeds',ha='center',fontsize=7.5,color='#4B5563')
save(fig,'fig01_success_curves')

# Figure 2: all methods and all observed cells; common absolute color scale.
fig,axes=plt.subplots(2,3,figsize=(7.5,5.35))
fig.subplots_adjust(left=.075,right=.985,bottom=.17,top=.925,wspace=.32,hspace=.43)
cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#EDF0F2')
for ri,suite in enumerate(SUITES):
    for ci,method in enumerate(METHODS):
        ax=axes[ri,ci];a=np.mean(np.stack([M[suite,k,method] for k in SEEDS]),axis=0)
        im=ax.imshow(a,cmap=cmap,vmin=0,vmax=100,interpolation='nearest')
        for y in range(4):
            for x in range(y+1):
                val=a[y,x];rgba=cmap(val/100);lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                ax.text(x,y,f'{val:.1f}',ha='center',va='center',color='white' if lum<.52 else '#142020',fontsize=8)
        ax.set(xticks=range(4),xticklabels=[6,7,8,9],yticks=range(4),yticklabels=[6,7,8,9],xlabel='Task')
        if ci==0:ax.set_ylabel('Training stage')
        ax.set_title(f'({chr(97+ri*3+ci)}) {suite} | '+('C (matched)' if method=='C' else method),loc='left',pad=7)
        metric=next(r for r in summary if r['suite']==suite and r['method']==method)
        ax.text(2.2,.05,f"AUC  {metric['AUC']:.1f}\nFWT  {metric['FWT']:.1f}\nNBT  {metric['NBT']:.1f}",ha='center',va='center',fontsize=7,color='#374151',linespacing=1.35)
        for sp in ax.spines.values():sp.set_visible(False)
        ax.tick_params(length=0)
cax=fig.add_axes([.21,.075,.60,.024]);cb=fig.colorbar(im,cax=cax,orientation='horizontal');cb.set_label('Success rate (%)',labelpad=3);cb.set_ticks([0,20,40,60,80,100]);cb.outline.set_linewidth(.5)
save(fig,'fig02_task_heatmaps')

# Figure 3: every gate block; two-axis view + mutually exclusive three-condition patterns.
fig,axes=plt.subplots(2,3,figsize=(7.7,5.7),gridspec_kw={'width_ratios':[1.05,1.17,.9]})
fig.subplots_adjust(left=.075,right=.985,bottom=.10,top=.87,wspace=.59,hspace=.53)
reject='#D66A46';accept=COLORS['S'];order=[0,1,2,4,3,5,6,7]
names=['All pass','BC','Exec','Ret.','BC + Exec','BC + Ret.','Exec + Ret.','All three']
for ri,suite in enumerate(SUITES):
    rows=G[suite];ax=axes[ri,0]
    for flag,color in [(0,reject),(1,accept)]:
        rr=[r for r in rows if r['accepted_Q']==flag]
        ax.scatter([r['relative_exec_pct'] for r in rr],[r['relative_retain_pct'] for r in rr],s=2.0,c=color,alpha=.22,edgecolors='none',rasterized=True,zorder=2+flag)
    for axis in ['x','y']:
        getattr(ax,f'set_{axis}scale')('symlog',linthresh=.01,linscale=.65,base=10)
        getattr(ax,f'set_{axis}lim')(-210,210)
        getattr(ax,f'set_{axis}ticks')([-100,-1,0,1,100])
        getattr(ax,f'set_{axis}ticklabels')(['−100','−1','0','1','100'])
    ax.axhline(0,color='#6B7280',lw=.65,zorder=1);ax.axvline(0,color='#6B7280',lw=.65,zorder=1)
    ax.set_xlabel(r'Relative $\Delta$ execution error (%)');ax.set_ylabel(r'Relative $\Delta$ retention error (%)')
    ax.set_title(f'({chr(97+ri*3)}) {suite} | Candidates',loc='left',pad=7)
    ax.text(.03,.97,f'n = {len(rows):,} blocks',transform=ax.transAxes,ha='left',va='top',fontsize=7.1,bbox=dict(facecolor='white',alpha=.88,edgecolor='none',pad=2))
    ax=axes[ri,1];cc=Counter(r['failure_mask'] for r in rows);val=[100*cc[m]/len(rows) for m in order]
    ax.barh(range(8),val,color=[accept]+[reject]*7,height=.67,zorder=3)
    ax.set_yticks(range(8),names,fontsize=6.9);ax.invert_yaxis();ax.set_xlim(0,60);ax.set_xticks([0,20,40,60]);ax.set_xlabel('Share of all blocks (%)');ax.set_title(f'({chr(98+ri*3)}) Violated conditions',loc='left',pad=7)
    ax.grid(axis='x',color='#DCE2E5',lw=.55,zorder=0);ax.set_axisbelow(True)
    for j,v in enumerate(val):ax.text(v+.8,j,f'{v:.1f}',va='center',fontsize=6.8)
    ax=axes[ri,2];tidy(ax);values=[]
    for j,seed in enumerate(SEEDS):
        v=[next(r['acceptance_pct'] for r in gate_stage if r['suite']==suite and r['seed']==seed and r['stage']==s) for s in [7,8,9]];values.append(v)
        ax.plot([7,8,9],v,color=accept,alpha=.25,lw=.9,marker=['o','^','s'][j],ms=3)
    av=np.mean(values,axis=0);ax.plot([7,8,9],av,color=accept,marker='D',lw=1.9,ms=4)
    ax.set(xlim=(6.85,9.15),ylim=(0,100),xticks=[7,8,9],yticks=[0,25,50,75,100],xlabel='Training stage',ylabel='Q acceptance (%)')
    ax.set_title(f'({chr(99+ri*3)}) Gate activity',loc='left',pad=7)
    for x,y in zip([7,8,9],av):ax.annotate(f'{y:.1f}',(x,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7)
fig.legend(handles=[Line2D([],[],linestyle='none',marker='o',color=accept,label='Q accepted'),Line2D([],[],linestyle='none',marker='o',color=reject,label='Q rejected')],loc='upper center',bbox_to_anchor=(.5,.995),ncol=2,frameon=False)
fig.text(.5,.931,r'S at $\lambda_R=0.1$ · All blocks · Q − C · Symmetric-log scatter axes',ha='center',fontsize=7.5,color='#4B5563')
save(fig,'fig03_gate_tradeoffs')

LAM_COLORS=['#0072B2','#007F75','#D55E00']
def lambda_panel(ax,suite,metric,title,ylim):
    tidy(ax);values=[]
    for j,seed in enumerate(SEEDS):
        v=[L[suite,seed,l][metric] for l in [.05,.1,.2]];values.append(v)
        xx=np.arange(3)+(j-1)*.045
        ax.plot(xx,v,color='#9CA3AF',alpha=.7,lw=.65,zorder=2)
        for x,y,c in zip(xx,v,LAM_COLORS):ax.scatter(x,y,s=17,color=c,marker=['o','^','s'][j],linewidth=.45,edgecolor='white',zorder=3)
    ax.plot(range(3),np.mean(values,axis=0),color='#17212B',marker='D',lw=1.5,ms=4,zorder=4)
    ax.set(xlim=(-.22,2.22),ylim=ylim,xticks=[0,1,2],xticklabels=['0.05','0.1','0.2'],xlabel=r'$\lambda_R$')
    ax.set_title(title,loc='left',pad=8)

legend=[Line2D([],[],color='#9CA3AF',marker=m,lw=.6,label=f'Seed {s}',ms=4) for s,m in zip(SEEDS,['o','^','s'])]+[Line2D([],[],color='#17212B',marker='D',lw=1.5,label='Mean',ms=4)]
fig,axes=plt.subplots(2,4,figsize=(7.7,4.65));fig.subplots_adjust(left=.07,right=.985,bottom=.11,top=.87,wspace=.42,hspace=.51)
for ri,suite in enumerate(SUITES):
    for ci,(metric,name,ylim) in enumerate([('AUC','AUC',(70,95)),('FinalAll','Final all',(65,100)),('OldFinal','Final old',(65,100)),('NewFinal','Final new',(35,105))]):
        ax=axes[ri,ci];lambda_panel(ax,suite,metric,f'({chr(97+ri*4+ci)}) {suite} | {name}',ylim)
        if ci==0:ax.set_ylabel('Score (%)')
fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.995),ncol=4,frameon=False,columnspacing=1.8)
save(fig,'fig04_lambda_sensitivity')

# Supplement: differences, with every positive and negative observed cell retained.
diffs={(s,m):np.mean(np.stack([M[s,k,'S']-M[s,k,m] for k in SEEDS]),axis=0) for s in SUITES for m in ['C','Q']}
vmax=float(5*np.ceil(max(np.nanmax(np.abs(a)) for a in diffs.values())/5))
fig,axes=plt.subplots(2,2,figsize=(5.9,5.5));fig.subplots_adjust(left=.10,right=.98,bottom=.17,top=.93,wspace=.38,hspace=.45)
dcmap=plt.get_cmap('RdBu').copy();dcmap.set_bad('#EDF0F2')
diffrows=[]
for ri,suite in enumerate(SUITES):
    for ci,method in enumerate(['C','Q']):
        ax=axes[ri,ci];a=diffs[suite,method];im=ax.imshow(a,cmap=dcmap,vmin=-vmax,vmax=vmax,interpolation='nearest')
        for y in range(4):
            for x in range(y+1):
                v=a[y,x];txt='0.0' if abs(v)<1e-8 else f'{v:+.1f}'
                ax.text(x,y,txt,ha='center',va='center',fontsize=8,color='white' if abs(v)>vmax*.62 else '#17212B')
                diffrows.append(dict(suite=suite,comparison=f'S-{method}',stage=y+6,task=x+6,delta_pp=v))
        ax.set(xticks=range(4),xticklabels=[6,7,8,9],yticks=range(4),yticklabels=[6,7,8,9],xlabel='Task',ylabel='Training stage')
        ax.set_title(f'({chr(97+ri*2+ci)}) {suite} | S − '+('C (matched)' if method=='C' else 'Q'),loc='left',pad=8)
        for sp in ax.spines.values():sp.set_visible(False)
        ax.tick_params(length=0)
cax=fig.add_axes([.22,.075,.59,.023]);cb=fig.colorbar(im,cax=cax,orientation='horizontal');cb.set_label('Success-rate difference (percentage points)',labelpad=4);cb.outline.set_linewidth(.5)
write_csv('stage_task_differences.csv',diffrows);save(fig,'figS01_task_differences')

# Supplement: retain the remaining two sensitivity metrics, including negative NBT.
fig,axes=plt.subplots(1,4,figsize=(7.7,2.65));fig.subplots_adjust(left=.07,right=.985,bottom=.20,top=.77,wspace=.45)
for i,(suite,metric) in enumerate([(s,m) for s in SUITES for m in ['FWT','NBT']]):
    ax=axes[i];lambda_panel(ax,suite,metric,f'({chr(97+i)}) {suite} | {metric}',(60,105) if metric=='FWT' else (-25,20))
    ax.set_ylabel('Score (%)' if metric=='FWT' else 'NBT (pp; lower is better)')
    if metric=='NBT':ax.axhline(0,color='#6B7280',ls='--',lw=.65,zorder=1)
fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,1.0),ncol=4,frameon=False)
save(fig,'figS02_lambda_fwt_nbt')

checks=dict(status='PASS_RECOMPUTED_AND_RENDERED_PENDING_VISUAL_REVIEW',
    comparison_cases=24,method_episode_rows=len(episode_rows),seed_count=3,episodes_per_stage_task_seed=20,
    gate_blocks={s:len(G[s]) for s in SUITES},gate_decisions_all_recomputed=True,
    baseline_outcomes_and_hashes_identical_across_comparisons=True,
    method_and_lambda_metrics_recomputed_from_paired_booleans=True,
    all_negative_cells_preserved=True,new_optimizer_calls=0,new_environment_episodes=0,
    scatter_sampling='NONE: every gate block rendered; scatter layer rasterized at 600dpi',
    relative_delta='200*(Q-C)/(abs(Q)+abs(C)); zero if both zero; percent, bounded [-200,200]',
    gate_rule='Q.BC<=C.BC and Q.Exec<=C.Exec and Q.Retain<C.Retain; exact original scalar comparisons',
    success_ribbons='Sample SD across three seeds; untruncated mean +/- SD, y axis 0..105',
    lambda_points='All seeds; small fixed horizontal offsets for visibility; mean line unjittered',
    layout=LAYOUT_AUDITS,versions=dict(python=platform.python_version(),matplotlib=matplotlib.__version__,numpy=np.__version__),
    limitations=['C matched optimizer budget differs from Q/S in accepted updates and collection','Gate diagnostics use training probes, not an independent outcome test','Historical lambda=0.1 baseline crosses hardware with new sensitivities','Repeated development evaluations; three seeds; no independent-test or significance claim'])
assert all(sha(EXEC/p)==h for p,h in SOURCE_HASHES.items()), 'Input evidence changed during rendering'
(HERE/'DATA_AUDIT.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf8')
(HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(SOURCE_HASHES,indent=2)+'\n',encoding='utf8')
art={str(p.relative_to(HERE)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file()}
(HERE/'FIGURE_MANIFEST.json').write_text(json.dumps(art,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=checks['status'],figures=ARTISTS,gate_blocks=checks['gate_blocks']),indent=2))

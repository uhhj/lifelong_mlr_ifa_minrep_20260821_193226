"""Plot returned scalar distances and private display coordinates; no scientific runs."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

R=Path(__file__).resolve().parent
P=Path(sys.argv[1]).resolve();assert P.parent==R
ev=json.loads((P/'EVIDENCE.json').read_text(encoding='utf8'))
assert ev['status']=='PASS_ALL_24_ORIGINAL_DISTANCE_RECOMPUTATIONS' and ev['inputs_before_after_unchanged']
assert all(ev[k]==0 for k in ['optimizer_calls','environment_steps','model_forward_calls','selection_calls'])
for sub in ['data_distance','figures_distance']:(R/sub).mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):(R/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
def csvout(name,rows):
    with (R/'data_distance'/name).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for name,info in ev['diagnostic_files'].items():assert sha(P/name)==info['sha256']
def diag(name):return json.loads((P/'diagnostics'/f'{name}.json').read_text(encoding='utf8'))
fields={'Random':'random_distance','K-Center':'kcenter_distance','RA-Coreset':'ra_distance'}
cols={'Random':'#64748B','K-Center':'#4777AC','RA-Coreset':'#007F75','Anchors':'#D97706'}
styles={'Random':':','K-Center':'--','RA-Coreset':'-'}
allrows={};stats=[];ecdfs=[];coverage=[]
for case in ev['results']:
    suite,seed,task=case['suite'],case['seed'],case['task'];key=f'{suite}_{seed}_{task}'
    rows=diag('nearest_'+key);assert len(rows)==case['N'];allrows[key]=rows
    for method,field in fields.items():
        a=np.array([r[field] for r in rows]);st=case['arms'][method]
        assert abs(a.mean()-st['mean'])<1e-12 and abs(a.max()-st['maximum'])<1e-12
        stats.append(dict(suite=suite,seed=seed,task=task,N=case['N'],K=case['K'],method=method,**st))
        values,counts=np.unique(a,return_counts=True);cdf=np.cumsum(counts)/len(a)
        ecdfs += [dict(suite=suite,seed=seed,task=task,method=method,distance=float(x),fraction=float(y),count_at_distance=int(n)) for x,y,n in zip(values,cdf,counts)]
        for demo,n in case['full_demo_counts'].items():
            count=case['demo_counts'][method].get(demo,0)
            coverage.append(dict(suite=suite,seed=seed,task=task,method=method,demo=demo,candidates=n,selected=count,fraction=count/n))
csvout('nearest_distance_summary.csv',stats);csvout('nearest_distance_ecdf.csv',ecdfs);csvout('all_cases_trajectory_allocation.csv',coverage)
summary={}
for suite in ['OBJECT','GOAL']:
    cs=[c for c in ev['results'] if c['suite']==suite]
    out={'cases':len(cs),'macro_mean_distances':{m:float(np.mean([c['arms'][m]['mean'] for c in cs])) for m in fields}}
    for comparator in ['Random','K-Center']:
        paired={}
        for metric in ['mean','p95','maximum']:
            changes=[100*(c['arms']['RA-Coreset'][metric]/c['arms'][comparator][metric]-1) for c in cs]
            paired[metric]=dict(better=sum(x< -1e-8 for x in changes),equal=sum(abs(x)<=1e-8 for x in changes),worse=sum(x>1e-8 for x in changes),change_percent_range=[min(changes),max(changes)],mean_paired_change_percent=float(np.mean(changes)))
        out['RA_vs_'+comparator]=paired
    summary[suite]=out
save('DISTANCE_SUMMARY.json',summary)
for f in ['arial.ttf','arialbd.ttf']:
    if (Path('C:/Windows/Fonts')/f).exists():font_manager.fontManager.addfont(str(Path('C:/Windows/Fonts')/f))
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7.5,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
exports=[]
def export(fig,name):
    for ext in ['pdf','svg','png']:
        p=R/'figures_distance'/f'{name}.{ext}';fig.savefig(p,dpi=600)
        if ext=='svg':p.write_text('\n'.join(x.rstrip() for x in p.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
        exports.append(dict(file=p.relative_to(R).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    plt.close(fig)
def ecdf(ax,rows,legend=False):
    for method,field in fields.items():
        a=np.sort([r[field] for r in rows]);x,n=np.unique(a,return_counts=True);y=np.cumsum(n)/len(a)
        ax.step(np.r_[0,x],np.r_[0,y],where='post',color=cols[method],ls=styles[method],lw=1.4,label=method)
    ax.set_ylim(0,1.025);ax.set_xlim(left=0);ax.grid(alpha=.15);ax.set_xlabel('Original nearest-center distance')
    ax.set_ylabel('Fraction of candidate windows')
    if legend:ax.legend(frameon=False,loc='lower right')

fig=plt.figure(figsize=(7.2,5.2));grid=fig.add_gridspec(2,2,width_ratios=[1.48,1],left=.045,right=.985,bottom=.15,top=.88,wspace=.3,hspace=.64)
projection_audit=[]
for i,suite in enumerate(['OBJECT','GOAL']):
    key=f'{suite}_41_6';rows=allrows[key];pr=diag('projection_'+key);assert [r['window_id'] for r in rows]==[r['window_id'] for r in pr]
    xy=np.array([[r['x'],r['y']] for r in pr]);span=np.ptp(xy,axis=0);lo=xy.min(0)-.05*span;hi=xy.max(0)+.05*span
    sub=grid[i,0].subgridspec(1,3,wspace=.08)
    for j,(method,flag) in enumerate([('K-Center','kcenter_selected'),('Anchors','anchor'),('RA-Coreset','ra_selected')]):
        ax=fig.add_subplot(sub[j]);mask=np.array([r[flag] for r in rows],dtype=bool)
        ax.scatter(xy[:,0],xy[:,1],s=2,color='#DEE2E7',lw=0,rasterized=False)
        ax.scatter(xy[mask,0],xy[mask,1],s=3.5,color=cols[method],lw=0,alpha=.8,rasterized=False)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xticks=[],yticks=[]);ax.set_aspect('equal',adjustable='box')
        ax.set_title(f'{method}\n$n={mask.sum()}$',fontsize=8,pad=6)
        for sp in ax.spines.values():sp.set_visible(False)
    ax=fig.add_subplot(grid[i,1]);ecdf(ax,rows,True)
    ax.set_title(f'Same query population; $N={len(rows)}$',fontsize=8,pad=6)
    p=grid[i,0].get_position(fig);p2=grid[i,1].get_position(fig)
    fig.text(p.x0,p.y1+.07,f'({chr(97+2*i)}) {suite}: selected subsets',fontsize=9,weight='bold')
    fig.text(p2.x0,p2.y1+.07,f'({chr(98+2*i)}) {suite}: original-distance ECDF',fontsize=9,weight='bold')
    fig.text(p.x0,p.y0-.045,'Seed 41, Task 6; one shared t-SNE projection.',fontsize=7,color='#475569')
    projection_audit.append(dict(case=key,shared_axes=[lo.tolist(),hi.tolist()],selection_flags=['kcenter_selected','anchor','ra_selected']))
fig.text(.045,.025,'Left: t-SNE is an illustration only. Right: original visual–action distances at equal selection quotas.\nThe right-hand query population includes selected windows with zero self-distance.',fontsize=7,color='#475569')
export(fig,'fig04b_ra_original_distance')

for suite in ['OBJECT','GOAL']:
    fig,axs=plt.subplots(4,3,figsize=(7.2,7.6))
    for i,t in enumerate(range(6,10)):
        for j,seed in enumerate([41,42,43]):
            ax=axs[i,j];ecdf(ax,allrows[f'{suite}_{seed}_{t}'])
            ax.set_title(f'Task {t}, seed {seed}',fontsize=8,loc='left')
            ax.set_xlabel('Original distance' if i==3 else '')
            ax.set_ylabel('Candidate fraction' if j==0 else '')
            if j>0:ax.set_yticklabels([])
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.958))
    fig.suptitle(suite+' — all 12 cases, identical query populations and quotas',fontsize=10,y=.985)
    fig.subplots_adjust(left=.08,right=.99,bottom=.115,top=.89,wspace=.24,hspace=.55)
    fig.text(.08,.02,'Original visual–action distances. No projection distances or method-specific query exclusions.\nTask 9 is post-final-training storage. Each curve includes selected-window zero self-distances.',fontsize=7)
    export(fig,'figS04_distance_ecdf_'+suite.lower())
save('DISTANCE_EXPORT_AUDIT.json',dict(figures=exports,projection_views=projection_audit))
save('DISTANCE_LOCAL_BINDING.json',dict(directory=P.name,evidence_sha256=sha(P/'EVIDENCE.json'),request_sha256=sha(R/'distance_readonly.py'),source_protocol_sha256=sha(R/'DISTANCE_PROTOCOL.md'),private_projection_files=['projection_OBJECT_41_6.json','projection_GOAL_41_6.json'],all_remote_inputs_unchanged=True))
# Public evidence contains hashes and scalars, not raw latent tensors or projection coordinate payloads.
save('data_distance/REMOTE_AUDIT_SUMMARY.json',ev)
print(json.dumps(summary,indent=2))

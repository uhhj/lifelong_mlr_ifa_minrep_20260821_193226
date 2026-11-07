"""Read archived selection metadata only; no model, environment, or reselection."""
from pathlib import Path
from collections import Counter
import json, hashlib, csv, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

R=Path(__file__).resolve().parent
W=R.parents[1]
E=W/'execution'
for sub in ['data','figures']:(R/sub).mkdir(exist_ok=True)
HASHES={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    HASHES[p.relative_to(W).as_posix()]=sha(p)
    return json.loads(p.read_text(encoding='utf-8'))
def save(name,obj):
    (R/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def csvout(name,rows):
    with (R/'data'/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def source(suite,seed):
    if suite=='OBJECT':return E/f'20261019_ra_object_conditional/completion_20261020/download/evidence/OBJECT_RA_seed{seed}'
    if seed==41:return E/'20261019_ra_coreset_goal_seed41/progress_20261019_222601/ra_evidence'
    if seed==42:return E/'20261018_ra_coreset_goal_seed42/final_evidence'
    return E/'20261019_ra_coreset_goal_extension/progress_20261019_160822/seed43/evidence'

records=[]; demos=[]; examples={}; details=[]
for suite in ['OBJECT','GOAL']:
    for seed in [41,42,43]:
        root=source(suite,seed);fm=read(root/'FEATURES.json')
        for t in range(6,10):
            m=read(root/f'ra_coreset/storage/task{t}.json');s=m['selection']
            pp=root/f'selection_preflight/task{t}.json'
            if suite=='GOAL' and seed==41:
                pp=root.parent/f'completion_ra/selection_preflight/task{t}.json'
            pre=read(pp)
            for k in ['original_ids','selected_ids','anchor_ids','random_draw_ids','swaps','max_before','max_after','mean_before','mean_after']:
                assert s[k]==pre[k],(suite,seed,t,k)
            ids=sorted(k.split('/',1)[1] for k in fm['hashes'] if k.startswith(str(t)+'/'))
            n=len(ids);q=s['quota'];a=set(s['anchor_ids']);kc=set(s['original_ids']);ra=set(s['selected_ids'])
            assert n==s['population']==fm['counts'][str(t)]
            assert q==(n+1)//2==m['K']==len(kc)==len(ra)==len(m['selected'])
            draw=random.Random(seed).sample(ids,q)
            assert draw==s['random_draw_ids'] and draw[:q//2]==s['anchor_ids']
            assert len(a)==q//2 and a<=ra and kc|ra|set(draw)<=set(ids)
            current=s['original_ids'].copy()
            for swap in s['swaps']:
                i=swap['parent_center_slot'];assert current[i]==swap['outgoing'] and current[i] not in a
                assert swap['incoming'] in a and swap['incoming'] not in current
                current[i]=swap['incoming']
            assert current==s['selected_ids'] and len(s['swaps'])==s['swap_count']==len(a-kc)
            assert [Path(x['source']).name for x in m['selected']]==current
            assert all(x['sha256']==fm['hashes'][f'{t}/'+Path(x['source']).name] for x in m['selected'])
            assert abs(s['original_selection']['covering_radius'][-1]-s['max_before'])<1e-10
            row=dict(suite=suite,seed=seed,task=t,population=n,quota=q,anchors=len(a),exchanges=s['swap_count'],anchor_retained_before=len(a&kc),anchor_retained_after=len(a),mean_kcenter=s['mean_before'],mean_ra=s['mean_after'],radius_kcenter=s['max_before'],radius_ra=s['max_after'],mean_ratio=s['mean_after']/s['mean_before'],radius_ratio=s['max_after']/s['max_before'],replayed_later=t<9)
            records.append(row)
            for demo,total in s['demo_full'].items():
                demos.append(dict(suite=suite,seed=seed,task=t,demo=demo,population=total,kcenter=s['demo_original'].get(demo,0),ra=s['demo_final'].get(demo,0)))
            if seed==41 and t==6:
                bp=(E/'20261019_ra_object_conditional/progress_20261019_230336/seed41/evidence/binding_metadata/task6.json' if suite=='OBJECT' else E/'20261019_ra_coreset_goal_seed41/progress_20261019_222601/completion_ra/binding_metadata/task6.json')
                b=read(bp)['windows'];assert set(b)==set(ids)
                assert all(v['sha256']==fm['hashes'][f'6/{k}'] for k,v in b.items())
                count=lambda ss:dict(Counter(b[k]['demo_id'] for k in ss))
                assert count(ids)==s['demo_full'] and count(kc)==s['demo_original'] and count(ra)==s['demo_final']
                ex=dict(binding=b,sets={'Random':set(draw),'K-Center':kc,'Anchors':a,'RA-Coreset':ra},counts={k:count(v) for k,v in [('Random',draw),('K-Center',kc),('Anchors',a),('RA-Coreset',ra)]},selection=s,full=count(ids))
                examples[suite]=ex
                for k,v in b.items():
                    details.append(dict(suite=suite,seed=41,task=6,window_id=k,demo=v['demo_id'],target=v['target'],length=v['length'],random=int(k in ex['sets']['Random']),kcenter=int(k in kc),anchor=int(k in a),ra=int(k in ra)))

csvout('all_cases_original_distance.csv',records)
csvout('all_cases_demonstration_counts.csv',demos)
csvout('illustrated_window_membership.csv',details)
coverage=[]
for suite,ex in examples.items():
    for method in ['Random','K-Center','RA-Coreset']:
        for demo,n in ex['full'].items():
            coverage.append(dict(suite=suite,seed=41,task=6,method=method,demo=demo,candidates=n,selected=ex['counts'][method].get(demo,0),fraction=ex['counts'][method].get(demo,0)/n))
csvout('illustrated_trajectory_coverage.csv',coverage)

for f in ['arial.ttf','arialbd.ttf','ariali.ttf']:
    if (Path('C:/Windows/Fonts')/f).exists():font_manager.fontManager.addfont(str(Path('C:/Windows/Fonts')/f))
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7.5,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
C={'Random':'#64748B','K-Center':'#4777AC','Anchors':'#D97706','RA-Coreset':'#007F75'}
exports=[]
def export(fig,name):
    for ext in ['pdf','svg','png']:
        p=R/'figures'/f'{name}.{ext}';fig.savefig(p,dpi=600)
        if ext=='svg':p.write_text('\n'.join(x.rstrip() for x in p.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
        exports.append(dict(file=p.relative_to(R).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    plt.close(fig)

# Same-case comparisons, with every demonstration retained and unknown identities disclosed.
fig=plt.figure(figsize=(7.2,7.0))
outer=fig.add_gridspec(2,2,width_ratios=[1.45,1],left=.07,right=.985,bottom=.18,top=.89,wspace=.30,hspace=.85)
for row,(suite,ex) in enumerate(examples.items()):
    sub=outer[row,0].subgridspec(1,3,wspace=.12)
    for col,method in enumerate(['K-Center','Anchors','RA-Coreset']):
        ax=fig.add_subplot(sub[col]);b=ex['binding'];sel=ex['sets'][method]
        for ids,color,size,z in [(set(b),'#DDE1E6',2,1),(sel,C[method],3.5,2)]:
            known=[b[k] for k in ids if b[k]['demo_id']!='AMBIGUOUS']
            ax.scatter([v['target']/max(1,v['length']-1) for v in known],[int(v['demo_id'].split('_')[1])+1 for v in known],s=size,c=color,linewidths=0,alpha=.8,zorder=z)
        if method in ['K-Center','RA-Coreset']:
            changed=(ex['sets']['K-Center']-ex['sets']['RA-Coreset'] if method=='K-Center' else ex['sets']['RA-Coreset']-ex['sets']['K-Center'])
            kk=[b[k] for k in changed if b[k]['demo_id']!='AMBIGUOUS']
            ax.scatter([v['target']/max(1,v['length']-1) for v in kk],[int(v['demo_id'].split('_')[1])+1 for v in kk],marker='x' if method=='K-Center' else '|',s=9,c='#B54141' if method=='K-Center' else C['Anchors'],linewidths=.55,zorder=3)
        ax.set(xlim=(-.03,1.03),ylim=(10.7,.3),xticks=[0,.5,1],yticks=range(1,11))
        ax.set_title(f'{method}\n$n={len(sel)}$',pad=6)
        ax.tick_params(length=2,pad=2)
        if col==0:ax.set_ylabel('Demonstration')
        else:ax.set_yticklabels([])
        if col==1:ax.set_xlabel('Normalized target time')
    ax0=fig.add_subplot(outer[row,1])
    methods=['Random','K-Center','RA-Coreset'];mat=np.array([[100*ex['counts'][method].get(f'demo_{d}',0)/ex['full'][f'demo_{d}'] for d in range(10)] for method in methods])
    im=ax0.imshow(mat,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto',interpolation='nearest')
    ax0.set(xticks=range(10),xticklabels=range(1,11),yticks=range(3),yticklabels=['Random','K-Center','RA'],xlabel='Demonstration')
    ax0.set_title(f'Equal quota: $K={ex["selection"]["quota"]}$\nRetained candidate windows (%)',pad=6)
    ax0.tick_params(length=0,pad=3)
    for y in range(3):
        for x in range(10):ax0.text(x,y,f'{mat[y,x]:.0f}',ha='center',va='center',fontsize=6.2,color='white' if mat[y,x]>65 else '#17212B')
    for sp in ax0.spines.values():sp.set_visible(False)
    pos=outer[row,0].get_position(fig);pos2=outer[row,1].get_position(fig)
    fig.text(pos.x0,pos.y1+.064,f'({chr(97+2*row)}) {suite}: selection membership',weight='bold',fontsize=9)
    fig.text(pos2.x0,pos2.y1+.064,f'({chr(98+2*row)}) {suite}: trajectory allocation',weight='bold',fontsize=9)
    unknown={m:ex['counts'][m].get('AMBIGUOUS',0) for m in ['Random','K-Center','Anchors','RA-Coreset']}
    note=f'Seed 41, Task 6; {ex["selection"]["swap_count"]} exchanges; all {ex["selection"]["anchor_count"]} anchors retained.'
    fig.text(.07,pos.y0-.077,note,fontsize=7.5)
    if suite=='OBJECT':fig.text(.07,pos.y0-.102,'Unknown identity: 4 candidates; Random / K-Center / anchors / RA = '+ ' / '.join(str(x) for x in unknown.values())+'.',fontsize=7,color='#475569')
cax=fig.add_axes([.66,.045,.27,.014]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,50,100]);cb.ax.tick_params(labelsize=7,length=2)
fig.legend(handles=[Line2D([],[],marker='x',ls='',color='#B54141',ms=4,label='Removed K-Center window'),Line2D([],[],marker='|',ls='',color=C['Anchors'],ms=5,label='Inserted anchor')],loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.0))
fig.text(.07,.035,'Gray: all candidate windows. Colored: selected windows.\nWindow multiplicity is retained in counts; overlapping points may coincide.',fontsize=7,color='#475569')
export(fig,'fig04_ra_selection_coverage')

# All 24 cases, original high-dimensional distance recorded during selection.
fig,axs=plt.subplots(2,2,figsize=(7.2,4.5),sharex=True)
for i,suite in enumerate(['OBJECT','GOAL']):
    rr=[r for r in records if r['suite']==suite]
    for j,(key,title) in enumerate([('mean_ratio','Mean nearest-center distance'),('radius_ratio','Maximum nearest-center distance')]):
        ax=axs[i,j];ax.axhline(1,color='#657080',ls='--',lw=.9)
        for seed,off,col,mark in [(41,-.14,'#4777AC','o'),(42,0,'#D97706','s'),(43,.14,'#007F75','^')]:
            ss=[r for r in rr if r['seed']==seed];ax.plot([r['task']+off for r in ss],[r[key] for r in ss],marker=mark,color=col,lw=.8,ms=4,label=f'Seed {seed}')
        ax.set_title(f'({chr(97+i*2+j)}) {suite}: {title}',loc='left',fontsize=8)
        ax.set_ylabel('RA / K-Center');ax.set_xticks([6,7,8,9]);ax.grid(axis='y',alpha=.16)
        if i==1:ax.set_xlabel('Task stored')
        vals=[r[key] for r in rr];span=max(max(vals),1)-min(min(vals),1);ax.set_ylim(min(min(vals),1)-max(.008,span*.25),max(max(vals),1)+max(.008,span*.25))
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1))
fig.subplots_adjust(left=.085,right=.985,top=.87,bottom=.17,hspace=.58,wspace=.26)
fig.text(.085,.035,'Archived original visual–action metric; lower is better. Dashed line: unchanged from K-Center.\nTask 9 is stored after the final training stage and is not replayed later in this experiment.',fontsize=7)
export(fig,'figS03_ra_original_distance_tradeoff')

summary={}
for suite in ['OBJECT','GOAL']:
    rr=[r for r in records if r['suite']==suite]
    summary[suite]=dict(cases=len(rr),radius_increased=sum(r['radius_ratio']>1+1e-10 for r in rr),mean_decreased=sum(r['mean_ratio']<1-1e-10 for r in rr),radius_percent_range=[100*(min(r['radius_ratio'] for r in rr)-1),100*(max(r['radius_ratio'] for r in rr)-1)],mean_percent_range=[100*(min(r['mean_ratio'] for r in rr)-1),100*(max(r['mean_ratio'] for r in rr)-1)],all_anchors_retained=True,example_all_methods_represented_demos={m:sum(examples[suite]['counts'][m].get(f'demo_{d}',0)>0 for d in range(10)) for m in ['Random','K-Center','RA-Coreset']})
save('SOURCE_MANIFEST.json',HASHES)
save('EXPORT_AUDIT.json',dict(figures=exports))
save('AUDIT.json',dict(status='PASS_ARCHIVED_SELECTION_AND_BINDING_RECOMPUTATION',cases=len(records),illustrated_cases=['OBJECT seed41 Task6','GOAL seed41 Task6'],optimizer_calls=0,environment_steps=0,new_selection_calls=0,model_forward_calls=0,raw_distance_recomputed=False,summary=summary,checks=['source hashes','equal quotas','seeded Random draw','protected anchor prefix','all exchange identities and order','all selected payload hashes','preflight equals production','original K-Center radius consistency','illustrated complete binding hashes and demonstration counts']))
assert all(sha(W/p)==h for p,h in HASHES.items())
print(json.dumps(summary,indent=2))

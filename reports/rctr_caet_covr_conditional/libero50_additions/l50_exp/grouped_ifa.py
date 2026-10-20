"""Declared five-new-task extension of the original pair-selection and mean IFA."""
import itertools
import math
from pathlib import Path
import torch
import torch.nn.functional as F

def select(current,feature_root,buffer_root):
    from libero.lifelong.algos.mlr import (
        collect_replay_task_features,collect_current_task_features,
        _concat_task_features,_pairwise_cosine_mean)
    current=list(current)
    assert len(current)==5 and current==list(range(current[0],current[0]+5))
    available=collect_replay_task_features(str(buffer_root),max_old_task_id=current[0])
    for t in current:
        language,agent=collect_current_task_features(t,str(feature_root))
        available[t]=dict(language=language,agent=agent)
    ids=sorted(available)
    features={t:{k:_concat_task_features(v) for k,v in available[t].items()} for t in ids}
    language=torch.zeros(len(ids),len(ids));visual=torch.zeros_like(language)
    for i,t in enumerate(ids):
        for j,u in enumerate(ids):
            language[i,j]=_pairwise_cosine_mean(features[t]["language"],features[u]["language"])
            visual[i,j]=_pairwise_cosine_mean(features[t]["agent"],features[u]["agent"])
    pairs=list(itertools.combinations(ids,2))
    ls=[float(language[ids.index(a),ids.index(b)]) for a,b in pairs]
    vs=[float(visual[ids.index(a),ids.index(b)]) for a,b in pairs]
    k=max(1,math.ceil(.5*len(pairs)))
    li=torch.topk(torch.tensor(ls),k=k,largest=True).indices.tolist()
    vi=torch.topk(torch.tensor(vs),k=k,largest=True).indices.tolist()
    chosen=[pairs[i] for i in sorted(set(li)&set(vi)) if set(pairs[i])&set(current)]
    return dict(current_tasks=current,task_ids=ids,selected_pairs=chosen,
                candidate_pairs=pairs,sim_language=language.tolist(),sim_agent=visual.tolist(),
                lang_top_indices=li,agent_top_indices=vi)

def references(current,feature_root,buffer_root):
    refs={};deviation={}
    for t in range(current[-1]+1):
        paths=sorted((Path(feature_root)/str(t)).glob("*.pt")) if t in current else sorted(Path(buffer_root).glob("t%d__*.pt"%t))
        assert paths
        ref=None;maximum=0.
        for path in paths:
            v=torch.load(path,map_location="cpu")["feature"][0,:,0,:]
            if ref is None:ref=v[0].clone()
            maximum=max(maximum,float((v-ref).abs().max()))
        assert maximum<1e-4,(t,maximum)
        refs[t]=ref;deviation[t]=maximum
    return refs,deviation

def loss(last,task_ids,info,refs,gamma=.1):
    current=set(info["current_tasks"])
    selected=info["selected_pairs"]
    all_terms=[]
    for t in info["current_tasks"]:
        mask=task_ids==t
        if not mask.any():continue
        others=[b if a==t else a for a,b in selected if t in (a,b)]
        g=F.normalize(last[mask],dim=-1,eps=1e-8)
        own=F.normalize(refs[t].to(last.device),dim=-1,eps=1e-8)
        pos=torch.acos((g*own).sum(-1).clamp(-1+1e-6,1-1e-6))
        terms=[]
        for other in others:
            ref=F.normalize(refs[other].to(last.device),dim=-1,eps=1e-8)
            neg=torch.acos((g*ref).sum(-1).clamp(-1+1e-6,1-1e-6))
            angle=torch.acos((own*ref).sum(-1).clamp(-1+1e-6,1-1e-6))
            terms.append(torch.relu(pos-neg+gamma*angle))
        all_terms.append(torch.stack(terms).mean(0) if terms else last[mask].sum(-1)*0)
    return torch.cat(all_terms).mean() if all_terms else last.sum()*0

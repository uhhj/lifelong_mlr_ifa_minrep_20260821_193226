"""Outcome-blind confirmation panel proposal; no training, no endpoint access."""
import re
import hashlib
import numpy as np
from .common import *

def canonical_bddl(path):
    text=Path(path).read_text()
    return hashlib.sha256(re.sub(r'\s+',' ',re.sub(r';[^\n]*','',text)).strip().encode()).hexdigest()

def maximin(names,embedding,count=3):
    normalized=embedding/np.linalg.norm(embedding,axis=1,keepdims=True)
    assert np.isfinite(normalized).all()
    selected=[min(range(len(names)),key=lambda i:names[i])]
    while len(selected)<count:
        pool=[i for i in range(len(names)) if i not in selected]
        selected.append(min(pool,key=lambda i:(-min(1-float(normalized[i]@normalized[j]) for j in selected),names[i])))
    return selected

def main():
    import torch
    from cb_ifa_experiment import gpu_runtime as env
    from libero.libero.benchmark import get_benchmark
    from libero.lifelong.models.complete_policy import CompletePolicy
    from libero.lifelong.utils import get_task_embs
    cfg_path=ROOT/'runs/fixed_kappa_confirmation_v2/feature_seed45/run_001/config.json'
    cfg=env.remap_config(read(cfg_path));cfg.use_wandb=False
    cfg.experiment_dir=str(STORE/'panel_bootstrap');cfg.feature_dir=str(STORE/'panel_bootstrap')
    torch.set_num_threads(2);env.seed_all(2026092601)
    model=CompletePolicy(cfg,cfg.shape_meta).cpu().eval()
    checkpoint=torch.load(cfg.pretrain,map_location='cpu')['state_dict']
    checkpoint={k[7:] if k.startswith('module.') else k:v for k,v in checkpoint.items()}
    language=model.language_encoder
    assert all(torch.equal(v,checkpoint['language_encoder.'+k]) for k,v in language.state_dict().items())
    language_hash=env.digest(language.state_dict());del checkpoint
    bddl_root=Path(cfg.bddl_folder)
    old=get_benchmark('libero_object')(0)
    excluded={}
    for i in range(old.n_tasks):
        task=old.get_task(i);path=bddl_root/task.problem_folder/task.bddl_file
        excluded[canonical_bddl(path)]=dict(canonical_id='libero_object/'+task.name,index=i,exposure='robot_base' if i<6 else 'development')
    records=[];selection=[];embeddings=[]
    for suite in ('libero_goal','libero_spatial'):
        benchmark=get_benchmark(suite)(0);eligible=[]
        for i in range(benchmark.n_tasks):
            task=benchmark.get_task(i);path=bddl_root/task.problem_folder/task.bddl_file
            bd=canonical_bddl(path);entry=dict(canonical_id=suite+'/'+task.name,suite=suite,index=i,language=task.language,bddl=str(path),bddl_sha256=sha(path),normalized_bddl_sha256=bd,dataset_relative_path=benchmark.get_task_demonstration(i),duplicate_definition=excluded.get(bd))
            records.append(entry)
            if bd not in excluded:eligible.append(entry)
        assert len(eligible)>=3
        tokens=get_task_embs(cfg,[t['language'] for t in eligible])
        with torch.no_grad():z=language(tokens['input_ids'].cpu(),tokens['attention_mask'].cpu()).double().numpy()
        chosen=maximin([r['canonical_id'] for r in eligible],z)
        selection.extend(eligible[i] for i in chosen)
        embeddings.extend(dict(canonical_id=e['canonical_id'],embedding=z[i].tolist()) for i,e in enumerate(eligible))
    freeze(OUT/'TASK_PANEL_PROPOSAL.json',dict(status='TEXT_SELECTION_COMPLETE_DATA_VALIDATION_PENDING',selected=selection,candidates=records,language_encoder_sha256=language_hash,language_config_sha256=sha(cfg_path),selection_rule='per suite lexicographically smallest canonical ID then maximin signed-cosine distance, lexicographic tie break',seeds=[48,49,50],seed_usage_audit='PENDING',dataset_duplicate_and_interface_audit='PENDING; no datasets substituted and no training launched',base_exposure_audit=read(OLD/'BASE_EXPOSURE_AUDIT.json'),formal_training_resumed=False))
    freeze(OUT/'PANEL_LANGUAGE_EMBEDDINGS.json',dict(encoder_sha256=language_hash,items=embeddings))
    print('TEXT_PANEL_SELECTED', [s['canonical_id'] for s in selection],flush=True)

if __name__=='__main__':main()

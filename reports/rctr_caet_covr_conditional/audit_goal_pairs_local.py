"""Independent metadata/graph/batch recomputation from read-only physical probe."""
import argparse,collections,hashlib,json,math
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);a=ap.parse_args();p=a.directory
 raw=(p/'PAIR_EVIDENCE.json').read_bytes();d=json.loads(raw);receipt=json.loads((p/'RECEIPT.json').read_text())
 assert hashlib.sha256(raw).hexdigest()==receipt['sha256']
 assert d['status']=='PASS' and d['optimizer_calls']==d['environment_steps']==d['server_writes']==0 and not d['cuda_initialized']
 summary={}
 for seed,sd in d['seeds'].items():
  summary[seed]={}
  for stage,st in sd['stages'].items():
   stage=int(stage);tr=st['train_metadata'];payload=st['payload_metadata'];train_ids=st['train_payload_indices']
   assert [payload[i] for i in train_ids]==tr and all(not x['holdout'] for x in tr)
   edges={tuple(e) for e in st['positive_edges_train_indices']};assert len(edges)*2==st['global_positive_directed']
   edge_payload=set();taskpairs=collections.Counter();demopairs=collections.Counter();cur=set();old=set();cur_edges=[]
   for i,j in edges:
    assert i<j and tr[i]['task']!=tr[j]['task'] and tr[i]['moving'] and tr[j]['moving']
    edge_payload.add(tuple(sorted((train_ids[i],train_ids[j]))))
    x,y=sorted((tr[i],tr[j]),key=lambda r:r['task']);taskpairs[f"{x['task']}-{y['task']}"]+=1
    if stage in (tr[i]['task'],tr[j]['task']):
     ci,oi=(i,j) if tr[i]['task']==stage else (j,i);cur.add(ci);old.add(oi);cur_edges.append((ci,oi))
     demopairs[f"{tr[ci]['demo']}|{tr[oi]['task']}:{tr[oi]['demo']}"]+=1
   assert dict(taskpairs)==st['support']['by_task_pair']
   assert len(cur_edges)==st['support']['current_old_pairs']
   batches=used=current_used=old_used=anybatch=currentbatch=0;exposed=set();epochcounts=[];seen_cur=set();seenpairs=set()
   for ep in st['epochs']:
    order=[];ec=0
    for b in ep['batches']:
     ids=b['payload_indices'];order+=ids;assert len(ids)==b['size']<=64
     neg=set(b['negative_supported_payload_indices']);assert neg<=set(ids)
     within={(i,j) for i,j in edge_payload if i in ids and j in ids}
     directed={(i,j) for edge in within for i,j in (edge,edge[::-1])}
     actual={(i,j) for i,j in directed if i in neg}
     assert len(directed)==b['positive_directed'] and actual=={tuple(x) for x in b['used_pairs']}
     assert len(actual)==b['used_directed']
     c=sum(payload[i]['task']==stage or payload[j]['task']==stage for i,j in actual)
     assert c==b['current_old_used_directed'] and len(actual)-c==b['old_old_used_directed']
     ca={i for i,j in actual if payload[i]['task']==stage};assert len(ca)==b['current_eligible_anchors']
     seen_cur.update(ca);seenpairs.update(tuple(sorted(x)) for x in actual)
     batches+=1;used+=len(actual);current_used+=c;old_used+=len(actual)-c;anybatch+=bool(actual);currentbatch+=c>0;ec+=c
    assert sorted(order)==sorted(train_ids) and len(order)==len(set(order))
    epochcounts.append(ec)
   bs=st['batch_summary'];assert (batches,used,current_used,old_used,anybatch,currentbatch)==tuple(bs[k] for k in ('total_batches','used_directed','current_old_used_directed','old_old_used_directed','batches_with_any','batches_with_current_old'))
   assert len(seen_cur)==bs['unique_current_anchors'] and len(seenpairs)==bs['unique_seen_pairs']
   moving=[i for i,x in enumerate(tr) if x['task']==stage and x['moving']]
   current_demo_total=len({x['demo'] for x in tr if x['task']==stage})
   summary[seed][str(stage)]=dict(current_old_unordered=len(cur_edges),old_old_unordered=len(edges)-len(cur_edges),task_pair_counts=dict(taskpairs),
    current_windows_supported=len(cur),current_moving_windows=len(moving),current_moving_coverage_pct=100*len(cur)/len(moving),
    current_unique_targets=len({(tr[i]['demo'],tr[i]['target']) for i in cur}),
    current_demos_supported=len({tr[i]['demo'] for i in cur}),current_demos_total=current_demo_total,
    old_demos_supported=len({(tr[i]['task'],tr[i]['demo']) for i in old}),
    demo_pair_counts=dict(demopairs),largest_demo_pair_share_pct=100*max(demopairs.values())/len(cur_edges) if cur_edges else None,
    effect_batches=batches,current_old_effective_batches=currentbatch,current_old_effective_batch_pct=100*currentbatch/batches,
    current_old_used_directed=current_used,old_old_used_directed=old_used,used_current_anchors=len(seen_cur),
    used_current_demos=len({payload[i]['demo'] for i in seen_cur}),used_current_demo_ids=sorted({payload[i]['demo'] for i in seen_cur}),
    used_directed_by_epoch=epochcounts,current_holdout_supported_windows=len(st['heldout_current_support']),
    minimum_threshold_margin=st['minimum_positive_threshold_margin'])
 out=dict(status='LOCAL_INDEPENDENT_PASS',summary=summary,optimizer_calls=0,environment_steps=0,
  limitations=['CPU replay of frozen batch indices and label masks, not training or GPU forward verification',
   'No learned model or closed-loop benefit measured; current holdout support is label overlap, not learned generalization'])
 (p/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()

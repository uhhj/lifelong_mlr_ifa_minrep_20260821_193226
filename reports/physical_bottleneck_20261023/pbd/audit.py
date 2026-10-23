"""Complete-case gate from saved predictions; no model forwards or scientific changes."""
from .common import *
from physical_effect_probe.train_probe import scalar_metrics
from physical_effect_probe.audit import bootstrap_difference

def main():
 verify();summaries={};allrows=[]
 for phase,n in [('READOUT',30),('SPATIAL',20)]:
  cost=js(OUT/(phase+'_COST.json'));assert cost['status']=='COMPLETE' and len(cost['fits'])==n;allrows+=cost['fits']
 assert sum(x['epochs'] for x in allrows)==10000
 for task in (6,7):
  d,_=data(task);meta=d['metadata'];valid=torch.where(d['mask'][:,0])[0]
  for kind in ('P0','M','Q','G','S'):
   for budget in ([50,100,200] if kind in ('P0','M') else [200]):
    ids=[];predictions={m:[] for m in ('normal','latest_visual_previous','other_demo_visual')}
    for fold in range(5):
     key='%s_task%d_fold%d_b%d'%(kind,task,fold,budget);case=js(OUT/'cases'/(key+'.json'));pf=OUT/'private/predictions'/(key+'.pth');assert sha(pf)==case['predictions_sha256'];assert sha(case['selection']['path'])==case['selection']['sha256'];p=torch.load(pf,map_location='cpu')['test'];ids.append(p['ids'])
     for mode in predictions:predictions[mode].append(p['predictions'][mode])
    ids=torch.cat(ids);assert len(ids)==len(valid) and set(ids.tolist())==set(valid.tolist())
    summary={m:scalar_metrics(torch.cat(pp),d['q'][ids],d['previous_q'][ids],meta,ids,d['dynamic'][ids]) for m,pp in predictions.items()};summaries['%s_b%d_task%d'%(kind,budget,task)]=summary
 report={}
 for kind in ('M','Q','G','S'):
  checks=[]
  for task in (6,7):
   cand=summaries['%s_b200_task%d'%(kind,task)];base=summaries['P0_b200_task%d'%task]
   for field in ('object_eef','object_goal'):
    normal=cand['normal'][field];ci=bootstrap_difference(base['normal'][field]['all']['by_demo'],normal['all']['by_demo']);dyn=normal['dynamic']['by_demo'];alt=cand['latest_visual_previous'][field]['dynamic']['by_demo'];lag=float(np.mean([v['previous_target_median_mm']-v['median_mm'] for v in dyn.values()]));vis=float(np.mean([alt[k]['median_mm']-v['median_mm'] for k,v in dyn.items()]))
    passed=bool(ci['ci95_mm'][0]>0 and normal['all']['median_mm']<=10 and normal['all']['p90_mm']<=25 and dyn and lag>0 and vis>0)
    checks.append(dict(task=task,target=field,median_mm=normal['all']['median_mm'],p90_mm=normal['all']['p90_mm'],bootstrap=ci,dynamic_lag_advantage_mm=lag,dynamic_visual_advantage_mm=vis,passed=passed))
  report[kind]=dict(passed=all(v['passed'] for v in checks),checks=checks)
 selected=next((k for k in ('M','Q','G','S') if report[k]['passed']),None)
 write(OUT/'RESULTS.json',dict(status='COMPLETE_NOT_LOCAL_ACCEPTED',summaries=summaries))
 write(OUT/'A_GATE.json',dict(status='PASS_A_PENDING_LOCAL_AUDIT' if selected else 'STOP_STATE_GATE',selected=selected,candidates=report,source_results_sha256=sha(OUT/'RESULTS.json'),contract_sha256=sha(ROOT/'CONTRACT.json'),fits=50,probe_epochs=10000,probe_updates=sum(x['updates'] for x in allrows),policy_updates=0,environment_episodes=0))
 log('A_gate',selected=selected)
if __name__=='__main__':main()

"""Local scheduler control-flow verification with fake children; no GPU/environment calls."""
import ast,json
from pathlib import Path
E=Path(__file__).resolve().parent
tree=ast.parse((E/'controller.py').read_text(encoding='utf8'))
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_pool')
code=compile(ast.Module(body=[fn],type_ignores=[]),'<isolated scheduler control flow>','exec')
results=[]
for failure_case in [None,'L020_41']:
 live={};events=[];pending_ticks={};peak=[0]
 def launch(case,phase):
  assert case not in pending_ticks;pending_ticks[case]=0;live[case]=phase
  events.append(('launch',case));peak[0]=max(peak[0],len(live))
 def finish(case,block=False):
  pending_ticks[case]+=1
  if not block and pending_ticks[case]<({'L020_41':1,'L020_42':3,'L020_43':2}[case]):return None
  del live[case];events.append(('finish',case));return 1 if case==failure_case else 0
 scope=dict(live=live,launch=launch,finish=finish,state=lambda x:None,time=type('Clock',(),{'sleep':staticmethod(lambda _:None)}))
 exec(code,scope)
 raised=False
 try:scope['train_pool'](['L020_41','L020_42','L020_43'])
 except RuntimeError:raised=True
 assert not live and peak[0]==2 and raised==(failure_case is not None)
 starts=[c for event,c in events if event=='launch']
 assert starts==(['L020_41','L020_42','L020_43'] if failure_case is None else ['L020_41','L020_42'])
 results.append(dict(failure_case=failure_case,events=events,max_active=peak[0],remaining_active=0))
final=ast.parse((E/'final_readonly.py').read_text(encoding='utf8'));values={}
for n in final.body:
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['CHECK_SCRIPT','SCRIPTS']:values[n.targets[0].id]=ast.literal_eval(n.value)
for case,s in values['SCRIPTS'].items():
 assert "Q.parent/'engineering_parallel_3090_20261102_v1'/'controller.exit'" in s
 assert "Q.parent/'engineering_resume_3090_20261102_v1'/'controller.exit'" not in s
 ast.parse(s)
template=ast.parse((E/'dispatch_handoff.py').read_text(encoding='utf8'))
sources=next(ast.literal_eval(n.value) for n in template.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCES' for t in n.targets))
assert all(s==(E/n).read_text(encoding='utf8') for n,s in sources.items())
out=dict(status='PASS_LOCAL_SCHEDULER_CONTROL_FLOW_AND_WRAPPER_BINDING',scenarios=results,training_calls=0,environment_calls=0,source_embedding_exact=True,final_CPU_completion_sources=6,scope='Local control-flow checks only; production handoff and full GPU diagnostics require actual evidence')
(E/'SCHEDULER_LOCAL_REVIEW.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out))

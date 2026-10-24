import os,json,hashlib,sys,io
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
Q=Path('/data/mlr_ifa/experiment_workspaces/RA_ALIGN_PAIRED_GOAL_20261024_v1_engineering02')
P=Path('/root/mlr_ifa/experiments/RA_ALIGN_PAIRED_GOAL_20261024_v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ct=json.loads((Q/'CONTRACT.json').read_text());assert all(sha(Q/n)==v for n,v in ct['files'].items())
out=json.loads((Q/'OUTCOMES.json').read_text())
if REQUEST['kind']=='meta':
 p=P/out[0]['run']/'00_terminal.npz'
 z=np.load(p,allow_pickle=False)
 spec=json.loads((Q/'INPUT_SPEC.json').read_text())
 cfg=json.loads((Path(spec['seeds'][0]['output'])/'resolved_eval.json').read_text())
 result={'source_contract_sha256':sha(Q/'CONTRACT.json'),'source_unchanged':True,'supervisor_exit':(Q/'supervisor.exit').read_text(),'worker_exit':(Q/'worker.exit').read_text(),'sample_npz_keys_shapes':{k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype)} for k in z.files},'cfg':{k:v for k,v in cfg.items() if k in ['benchmark_name','task_order_index','bddl_folder','init_states_folder','data']},'private_files':len(list(P.rglob('*.npz'))),'physical_state_saved':False,'physical_state_note':'Worker saves policy frames, GMM parameters, low observations and actions, but no full simulator/controller state. Future intervention would require verified replay.'}
 import contextlib
 os.environ['LIBERO_CONFIG_PATH']='/data/mlr_ifa/config';sys.path.insert(0,spec['seeds'][0]['source'])
 with contextlib.redirect_stdout(io.StringIO()):
  from libero.libero.benchmark import get_benchmark
  task=get_benchmark(cfg['benchmark_name'])(cfg['data']['task_order_index']).get_task(8)
 bp=Path(cfg['bddl_folder'])/task.problem_folder/task.bddl_file
 result['task']={'language':task.language,'name':task.name,'bddl':bp.read_text(),'bddl_sha256':sha(bp)}
 coverage=[]
 for o in out:
  if o['seed']!=41:continue
  run=o['run'];man=json.loads((Q/'light'/run/'PRIVATE_MANIFEST.json').read_text());windows=[]
  for window in ['first_mean_delta','first_mode_switch','first_gripper_flip','terminal']:
   fn='%02d_%s.npz'%(o['initial_index'],window);p=P/run/fn
   if not p.exists():continue
   assert sha(p)==man[fn]['sha256'];z=np.load(p,allow_pickle=False);n=len(z['agentview_image'])
   end=o['policy_steps'] if window=='terminal' else o['first_events'][window[6:]]
   windows.append({'window':window,'start':end-n+1,'end':end,'frames_per_camera':n,'sha256':man[fn]['sha256'],'keys':z.files})
  coverage.append({'run':run,'initial_index':o['initial_index'],'success':o['success'],'steps':o['policy_steps'],'windows':windows})
 result['discovery_window_coverage']=coverage
 pairs_equal=[]
 for index in range(20):
  zs=[np.load(P/('%s_s41_driver%d_g%d'%(m,d,index//10))/('%02d_first_mean_delta.npz'%index),allow_pickle=False) for m in ['RA','ALIGN'] for d in [8,9]]
  assert all(all(np.array_equal(z[k],zs[0][k]) for k in z.files) for z in zs)
  pairs_equal.append(index)
 result['initial_raw_windows_equal_across_four_drivers']=pairs_equal
 print(json.dumps(result));sys.exit(0)
assert REQUEST['kind']=='sheet'
seed=int(REQUEST['seed']);method=REQUEST['method'];indices=REQUEST['indices'];window=REQUEST.get('window','terminal');assert seed in [41,42,43] and method in ['RA','ALIGN'] and len(indices)<=5 and all(0<=i<20 for i in indices)
assert window in ['terminal','first_mode_switch','first_gripper_flip','first_mean_delta']
if window=='first_mean_delta':
 canvas=Image.new('RGB',(len(indices)*192,410),(238,238,238));draw=ImageDraw.Draw(canvas)
 for j,index in enumerate(indices):
  run=f'{method}_s{seed}_driver9_g{index//10}';fn=f'{index:02d}_first_mean_delta.npz';p=P/run/fn
  man=json.loads((Q/'light'/run/'PRIVATE_MANIFEST.json').read_text());assert sha(p)==man[fn]['sha256'];z=np.load(p,allow_pickle=False)
  assert len(z['agentview_image'])==1
  for ci,k in enumerate(['agentview_image','robot0_eye_in_hand_image']):canvas.paste(Image.fromarray(z[k][0]).resize((192,192)),(j*192,20+ci*192))
  draw.text((j*192,0),f'i{index} initial t0 (equal all4)',fill='black')
 b=io.BytesIO();canvas.save(b,format='JPEG',quality=90);sys.stdout.buffer.write(b.getvalue());sys.exit(0)
# 2 drivers x 2 cameras, 3 time points per pair. Raw RGB stays in remote memory.
W=158;H=158;canvas=Image.new('RGB',(6*W, len(indices)*(2*H+42)),(238,238,238));draw=ImageDraw.Draw(canvas)
for ri,index in enumerate(indices):
 for dj,driver in enumerate([8,9]):
  run=f'{method}_s{seed}_driver{driver}_g{index//10}';o=next(o for o in out if o['run']==run and o['initial_index']==index)
  fn=f'{index:02d}_{window}.npz';p=P/run/fn;y=ri*(2*H+42)+22
  draw.text((dj*3*W,y-18),f'{method} s{seed} i{index} S{driver} {window}',fill='black')
  if not p.exists():draw.text((dj*3*W,y+10),'NO SAVED EVENT',fill='black');continue
  manifest=json.loads((Q/'light'/run/'PRIVATE_MANIFEST.json').read_text());assert sha(p)==manifest[fn]['sha256']
  z=np.load(p,allow_pickle=False);keys=[k for k in z.files if 'image' in k];assert len(keys)==2
  n=len(z[keys[0]]);sel=[0,n//2,n-1]
  end=o['policy_steps'] if window=='terminal' else o['first_events'][window[6:]]
  for ti,t in enumerate(sel):
   for ci,k in enumerate(keys):
    a=z[k][t];assert a.dtype==np.uint8 and a.ndim==3
    im=Image.fromarray(a).resize((W,H));canvas.paste(im,(dj*3*W+ti*W,y+ci*H));draw.text((dj*3*W+ti*W+2,y+ci*H+2),f'{k} t{end-n+1+t}',fill='red')
b=io.BytesIO();canvas.save(b,format='JPEG',quality=90);sys.stdout.buffer.write(b.getvalue())

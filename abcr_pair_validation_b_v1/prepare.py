"""Prepare blinded BASE-only human annotation, without calibrating or screening.

Raw HDF5 anchors do not depend on exported feature record numbering. This does
not certify any incremental .pt/image alignment. No model or simulator imports.
"""
import argparse
import ast
import csv
import hashlib
import itertools
import json
import os
import platform
import time
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from .original_experiment_b import sha256, dump_json, save_csv


PACK_PROTOCOL = dict(schema='abcr_B_annotation_preparation_v1', tasks=list(range(6)),
    calibration_demos=list(range(25)), validation_demos=list(range(25,50)),
    anchor_start=7, anchor_stride=8, future_q=3,
    pairs_per_task_pair={'calibration':20,'validation':10}, random_seed=20261014,
    sampling='uniform random permutation of physically admissible pairs; one per unordered demo pair',
    physical={'position_m':.03,'rotation_deg':15.,'gripper_fraction':.10,'tau_s':1.},
    out_of_travel='EXCLUDE_AND_REPORT; no clipping or empirical travel rescaling',
    visual_score_used_for_selection=False, labels_auto_filled=False,
    human_images='raw HDF5 RGB flipped vertically for OpenGL display ONLY; no model preprocessing implied',
    velocity='POSE_ONLY_NOT_FULL_DYNAMIC_STATE',
    calibration='NOT_RUN_WITHOUT_INDEPENDENT_HUMAN_LABELS',
    incremental_screen='NOT_RUN', formal_training_resumed=False)


def atomic_csv(path, rows, fields):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    save_csv(tmp,rows,fields);os.replace(tmp,path)


def task_names(root):
    p=root/'libero/libero/benchmark/libero_suite_task_map.py'
    for n in ast.parse(p.read_text()).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='libero_task_map' for t in n.targets):
            return ast.literal_eval(n.value)['libero_object']
    raise ValueError('TASK_MAP_NOT_FOUND')


def physical_distances(a,b):
    pos=np.linalg.norm(a['p'][:,None,:]-b['p'][None,:,:],axis=2)
    qa=a['q']/np.linalg.norm(a['q'],axis=1,keepdims=True)
    qb=b['q']/np.linalg.norm(b['q'],axis=1,keepdims=True)
    angle=2*np.arccos(np.clip(np.abs(qa@qb.T),0,1))
    grip=np.abs(a['g'][:,None]-b['g'][None,:])
    return np.maximum.reduce([pos/.03,angle/np.deg2rad(15),grip/.10]),pos,np.rad2deg(angle),grip


def orientation_crosscheck(quat,rotvec):
    # Axis-angle representations can differ by 2*pi while encoding the same
    # rotation. Compare physical rotations, not coordinate-vector subtraction.
    return float((Rotation.from_quat(quat).inv()*Rotation.from_rotvec(rotvec)).magnitude().max())


def pick_pairs(a,b,limit,seed):
    ds,pos,rot,grip=physical_distances(a,b)
    ii,jj=np.where(ds<=1)
    order=np.random.RandomState(seed).permutation(len(ii))
    selected=[];seen=set()
    for k in order:
        i,j=int(ii[k]),int(jj[k]);key=(a['rows'][i]['traj_id'],b['rows'][j]['traj_id'])
        if key in seen:continue
        seen.add(key)
        selected.append((a['rows'][i],b['rows'][j],float(ds[i,j]),float(pos[i,j]),float(rot[i,j]),float(grip[i,j])))
        if len(selected)==limit:break
    return selected,len(ii)


def collect(root,out):
    names=task_names(root);pool={};audit=[];assets=[];frame_signatures=set()
    for task in range(6):
        p=root/'data/libero_object'/(names[task]+'_demo.hdf5')
        with h5py.File(p,'r') as f:
            env=json.loads(f['data'].attrs['env_args'])
            assert env['env_kwargs']['robots']==['Panda']
            assert f['data'].attrs['macros_image_convention']=='opengl'
            assets.append(dict(task=task,path=str(p),bytes=p.stat().st_size,sha256=sha256(p),
                env_args=env,raw_image_convention='opengl'))
            buckets={'calibration':[],'validation':[]}
            for demo_index in range(50):
                d=f['data']['demo_'+str(demo_index)];obs=d['obs'];state=d['robot_states'][:]
                actions=d['actions'][:].astype(np.float32)
                assert state.shape==(len(actions),9)
                assert np.array_equal(state[:,:2],obs['gripper_states'][:])
                assert np.array_equal(state[:,2:5],obs['ee_pos'][:])
                assert np.max(abs(np.linalg.norm(state[:,5:9],axis=1)-1))<1e-10
                # Corroborate native xyzw quaternion against exported rotation vector.
                ori=obs['ee_ori'][:]
                orientation_error=orientation_crosscheck(state[:,5:9],ori)
                assert orientation_error<1e-7, 'ORIENTATION_ENCODING_NOT_VERIFIED'
                xml=ET.fromstring(d.attrs['model_file'])
                base=next(b for b in xml.findall('.//body') if b.attrib.get('name')=='robot0_base')
                frame_signatures.add(json.dumps(base.attrib,sort_keys=True))
                joints={j.attrib['name']:j.attrib for j in xml.findall('.//joint') if 'finger_joint' in j.attrib.get('name','')}
                left=next(j for k,j in joints.items() if k.endswith('finger_joint1'))
                right=next(j for k,j in joints.items() if k.endswith('finger_joint2'))
                assert np.allclose(np.fromstring(left['range'],sep=' '),[0,.04])
                assert np.allclose(np.fromstring(right['range'],sep=' '),[-.04,0])
                g=(state[:,0]-state[:,1])/.08
                role='calibration' if demo_index<25 else 'validation'
                anchors=list(range(7,len(actions)-2,8));bad=0
                for t in anchors:
                    if not 0<=g[t]<=1:
                        bad+=1;continue
                    assert actions[t:t+3].shape==(3,7)
                    row=dict(id='hdf5:libero_object/task%d/demo_%d/t%d'%(task,demo_index,t),task=task,
                        traj_id='object/task%d/demo_%d'%(task,demo_index),demo='demo_'+str(demo_index),
                        t=t,hdf5_file=str(p),role=role,p=state[t,2:5],q=state[t,5:9],g=float(g[t]))
                    buckets[role].append(row)
                audit.append(dict(task=task,demo=demo_index,role=role,frames=len(actions),
                    sampled_anchors=len(anchors),excluded_gripper_out_of_travel=bad,
                    gripper_fraction_min=float(g.min()),gripper_fraction_max=float(g.max()),
                    orientation_max_error=orientation_error))
            for role,rows in buckets.items():
                assert rows
                pool[(task,role)]=dict(rows=rows,p=np.stack([r['p'] for r in rows]),
                    q=np.stack([r['q'] for r in rows]),g=np.array([r['g'] for r in rows]))
        print('BASE_ASSET_READY',task,flush=True)
    assert len(frame_signatures)==1,'ROBOT_BASE_FRAME_MISMATCH'
    dump_json(out/'BASE_ASSET_MANIFEST.json',dict(assets=assets,robot_base_frame=list(frame_signatures),
        frame='shared recorded world frame and identical robot base transform',
        quaternion_source='robot_states[:,5:9], get_robot_state_vector -> robot0_eef_quat xyzw',
        gripper_source='(left_joint_qpos-right_joint_qpos)/0.08 from recorded XML travel',
        velocity='not provided by HDF5; no estimated velocities introduced'))
    atomic_csv(out/'PHYSICAL_FIELD_AUDIT.csv',audit,list(audit[0]))
    return pool,names


def export_images(out,pairs,names):
    image_dir=out/'annotation/images';image_dir.mkdir(parents=True)
    files={};record_rows={}
    try:
        for row in pairs:
            canvas=Image.new('RGB',(800,650),'white');draw=ImageDraw.Draw(canvas)
            draw.text((10,8),row['pair_id']+' | LABEL BLANK - independent human review',fill='black')
            for col,r in enumerate((row['_a'],row['_b'])):
                x=col*400;path=r['hdf5_file']
                if path not in files:files[path]=h5py.File(path,'r')
                d=files[path]['data/'+r['demo']];digests={}
                for v,key in enumerate(('agentview_rgb','eye_in_hand_rgb')):
                    rgb=d['obs'][key][r['t']]
                    assert rgb.shape==(128,128,3) and rgb.dtype==np.uint8
                    digests[key]=hashlib.sha256(rgb.tobytes()).hexdigest()
                    # Display orientation only; model inputs must follow native preprocessing.
                    im=Image.fromarray(rgb[::-1].copy()).resize((288,288),Image.Resampling.NEAREST)
                    canvas.paste(im,(x+55,65+v*290))
                draw.text((x+8,26),'Task%d %s t%d'%(r['task'],r['demo'],r['t']),fill='black')
                obj=names[r['task']].replace('pick_up_the_','').replace('_and_place_it_in_the_basket','')
                draw.text((x+8,43),obj,fill='black')
                record_rows[r['id']]=dict(id=r['id'],task_id=r['task'],traj_id=r['traj_id'],
                    demo=r['demo'],anchor_t=r['t'],role=r['role'],hdf5_file=path,
                    eef_pos_m=r['p'].tolist(),eef_quat_xyzw=r['q'].tolist(),gripper_open_fraction=r['g'],
                    rgb_sha256=digests,future_actions=d['actions'][r['t']:r['t']+3].astype(np.float32).tolist(),
                    future_valid=[True]*3,source='DIRECT_HDF5_ANCHOR',
                    pt_alignment_verified=False,unmodulated_feature_export='PENDING')
            canvas.save(image_dir/(row['pair_id']+'.png'))
    finally:
        for f in files.values():f.close()
    dump_json(out/'BASE_RECORD_MANIFEST.json',dict(records=list(record_rows.values()),
        note='Direct HDF5 anchors for human pack, not a records.npz ready for calibration; encoder features pending'))
    return record_rows


def html(out,pairs):
    # No automatic predicted labels or similarity scores displayed to annotators.
    data=[dict(pair_id=r['pair_id'],id_a=r['id_a'],id_b=r['id_b'],role=r['role']) for r in pairs]
    template='''<!doctype html><meta charset="utf-8"><title>ABCR B 人工标注</title>
<style>body{font:17px sans-serif;max-width:950px;margin:auto;padding:20px}img{width:800px;max-width:100%}button,select,input{font:inherit;margin:8px}p{line-height:1.6}</style>
<h1>基础任务跨任务状态可比性标注</h1>
<p>左右是两条真实演示；每侧上方为外部相机，下方为腕部相机。结合末端—目标物体关系、抓取/放置阶段和相关物体配置判断局部状态是否足够可比。不能只因桌面、篮子或机械臂姿态相似就标正例；也不能只因目标物体类别不同就标负例。1=可比，0=不可比，U=证据不足。不要根据想要的标签配额作判断。</p>
<p>所有候选已通过预注册位姿门，但这不是真值标签。未显示视觉余弦、专家动作类别或实验 A 判定。U 不会导出成 0。校准与验证分属不相交轨迹。</p>
<label>标注者<input id="who"></label><button onclick="move(-1)">上一对</button><button onclick="move(1)">下一对</button><button onclick="download()">导出标注 CSV</button><br>
<b id="title"></b><br><img id="image"><br><select id="lab" onchange="save()"><option value="">未标注</option><option value="1">1 可比</option><option value="0">0 不可比</option><option value="U">U 不确定</option></select><input id="note" placeholder="理由／局部关系" oninput="save()">
<script>const rows=__DATA__;let n=0;const key='abcr-B-v1-human-20261014';let values=JSON.parse(localStorage.getItem(key)||'{}');
function show(){let r=rows[n],v=values[r.pair_id]||{};document.getElementById('title').textContent=(n+1)+' / '+rows.length+' — '+r.pair_id+' — '+r.role;document.getElementById('image').src='images/'+r.pair_id+'.png';document.getElementById('lab').value=v.label||'';document.getElementById('note').value=v.note||'';}
function save(){values[rows[n].pair_id]={label:document.getElementById('lab').value,note:document.getElementById('note').value,annotator:document.getElementById('who').value};localStorage.setItem(key,JSON.stringify(values));}
function move(d){save();n=Math.max(0,Math.min(rows.length-1,n+d));show();}
function download(){save();let q=x=>'"'+String(x??'').replaceAll('"','""')+'"';let a=[['pair_id','role','id_a','id_b','label','uncertain','annotator','note']];for(let r of rows){let v=values[r.pair_id]||{};a.push([r.pair_id,r.role,r.id_a,r.id_b,v.label==='U'?'':(v.label||''),v.label==='U'?1:0,v.annotator||'',v.note||'']);}let el=document.createElement('a');el.href=URL.createObjectURL(new Blob(['\\ufeff'+a.map(x=>x.map(q).join(',')).join('\\n')],{type:'text/csv;charset=utf-8'}));el.download='human_labels.csv';el.click();URL.revokeObjectURL(el.href);}
show();</script>'''
    (out/'annotation/index.html').write_text(template.replace('__DATA__',json.dumps(data)),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    root=a.root.resolve();out=root/'reports/abcr_pair_validation_b_v1'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'PACK_COMPLETE.json').exists():
        record=json.loads((out/'PACK_COMPLETE.json').read_text())
        assert all(sha256(out/k)==v for k,v in record['outputs'].items()),'PACK_CHANGED'
        print('PACK_COMPLETE_ALREADY_EXISTS_NO_REMEASUREMENT');return
    assert not (out/'ANNOTATION_MANIFEST.csv').exists(),'Partial pack preserved: inspect before retry'
    start=time.time()
    if (out/'ANNOTATION_PROTOCOL.json').exists():
        assert json.loads((out/'ANNOTATION_PROTOCOL.json').read_text())==PACK_PROTOCOL,'PREPARATION_PROTOCOL_CHANGED'
    else:dump_json(out/'ANNOTATION_PROTOCOL.json',PACK_PROTOCOL)
    pool,names=collect(root,out);pairs=[];strata=[]
    for role in ('calibration','validation'):
        for ta,tb in itertools.combinations(range(6),2):
            limit=PACK_PROTOCOL['pairs_per_task_pair'][role]
            got,count=pick_pairs(pool[ta,role],pool[tb,role],limit,
                PACK_PROTOCOL['random_seed']+1000*(role=='validation')+ta*10+tb)
            strata.append(dict(role=role,task_a=ta,task_b=tb,eligible_physically_gated_pairs=count,
                requested=limit,selected=len(got),unique_trajectory_pairs=len(got)))
            for k,(ra,rb,ds,p,r,g) in enumerate(got):
                pairs.append(dict(pair_id='%s_t%d_t%d_%03d'%(role,ta,tb,k),role=role,
                    id_a=ra['id'],id_b=rb['id'],label='',uncertain='',annotator='',note='',
                    physical_distance=ds,position_delta_m=p,rotation_delta_deg=r,gripper_fraction_delta=g,
                    _a=ra,_b=rb))
    assert pairs,'NO_PHYSICAL_PAIRS'
    records=export_images(out,pairs,names);html(out,pairs)
    cal={r['traj_id'] for r in records.values() if r['role']=='calibration'}
    val={r['traj_id'] for r in records.values() if r['role']=='validation'}
    assert not cal & val
    clean=[{k:v for k,v in r.items() if not k.startswith('_')} for r in pairs]
    atomic_csv(out/'ANNOTATION_MANIFEST.csv',clean,list(clean[0]))
    fields=['pair_id','role','id_a','id_b','label','uncertain','annotator','note']
    for role in ('calibration','validation'):
        atomic_csv(out/('annotation/'+role+'_labels_blank.csv'),
            [{k:r[k] for k in fields} for r in clean if r['role']==role],fields)
    atomic_csv(out/'ANNOTATION_STRATA.csv',strata,list(strata[0]))
    dump_json(out/'PREPARATION_SUMMARY.json',dict(status='WAITING_FOR_INDEPENDENT_HUMAN_LABELS',
        pairs_by_role={role:sum(r['role']==role for r in pairs) for role in ('calibration','validation')},
        unique_records=len(records),calibration_trajectories=len(cal),validation_trajectories=len(val),
        trajectory_overlap=0,seconds=time.time()-start,python=platform.python_version(),
        cuda_used=False,training=False,formal_training_resumed=False,
        visual_calibration='NOT_RUN',incremental_screen='NOT_RUN',
        physical_matching_accuracy='NOT_ESTABLISHED_WITHOUT_LABELS',
        unmodulated_features='PENDING_EXPORT',
        limitations=['Pose only, not dynamic equivalence','Gripper out-of-travel anchors excluded and counted',
            'All selected pairs pass pose gate; negatives must come from independent scene/relative-state judgment',
            'No post-label sample replenishment or threshold retuning authorized automatically']))
    # A live stdout log changes after publication; bind artifacts, not that log.
    outputs={str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*'))
        if p.is_file() and p.name!='PACK_COMPLETE.json' and p.suffix!='.log'}
    dump_json(out/'PACK_COMPLETE.json',dict(status='ANNOTATION_PACK_READY_NOT_EXPERIMENT_B_PASS',outputs=outputs))
    print('ANNOTATION_PACK_READY',len(pairs),'CUDA_NOT_IMPORTED',flush=True)


if __name__=='__main__':main()

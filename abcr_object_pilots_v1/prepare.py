"""Freeze raw-frame identities and verify simulator-state time alignment."""
import argparse
import ast
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import h5py
import numpy as np
from .common import PROTOCOL,sha,write,npz

def names(root):
    tree=ast.parse((root/'libero/libero/benchmark/libero_suite_task_map.py').read_text())
    return next(ast.literal_eval(n.value)['libero_object'] for n in tree.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='libero_task_map' for t in n.targets))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(PROTOCOL,checkpoint_sha256=sha(a.repo/'checkpoints'/PROTOCOL['checkpoint']),
                  native_complete_policy_config_sha256=sha(a.repo/'libero/configs/policy/complete_policy.yaml'),
                  state_time_evidence_url='https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/scripts/create_dataset.py')
    write(a.out/'PROTOCOL.json',protocol,immutable=True)
    demos=[];assets=[];offset=0
    all_state=[];all_actions=[]
    for task,name in enumerate(names(a.repo)):
        path=a.repo/'data/libero_object'/(name+'_demo.hdf5')
        assets.append(dict(task=task,path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
        with h5py.File(path,'r') as h:
            assert h['data'].attrs['macros_image_convention']=='opengl'
            for demo in range(5 if task<6 else 10):
                d=h['data']['demo_'+str(demo)];n=len(d['actions'])
                state=np.concatenate([d['obs/joint_states'][:],d['obs/gripper_states'][:]],axis=1)
                sim=d['states'][:];robot=d['robot_states'][:]
                assert state.shape==(n,9)
                # Native XML free-joint coordinates in saved MjSimState, time + qpos + qvel.
                xml=ET.fromstring(d.attrs['model_file']);qadr=0;nv=0;joints={}
                for joint in xml.find('worldbody').iter('joint'):
                    typ=joint.get('type','hinge');joints[joint.get('name')]=(qadr,typ)
                    qadr+={'free':7,'ball':4}.get(typ,1);nv+={'free':6,'ball':3}.get(typ,1)
                assert sim.shape==(n,1+qadr+nv)
                assert np.array_equal(sim[1:,1:10],state[:-1]),'OBS_SIM_TIME_ALIGNMENT_MISMATCH'
                assert np.array_equal(robot[:,:2],state[:,7:])
                assert np.array_equal(robot[:,2:5],d['obs/ee_pos'][:])
                target=name[len('pick_up_the_'):-len('_and_place_it_in_the_basket')]+'_1_joint0'
                assert target in joints and joints[target][1]=='free'
                target_adr=joints[target][0]
                target_xyz=np.zeros((n,3));target_xyz[:-1]=sim[1:,1+target_adr:4+target_adr]
                target_valid=np.arange(n)<n-1
                future=np.zeros((n,6,7),np.float32);valid=np.zeros((n,6),bool)
                actions=d['actions'][:].astype(np.float32)
                for horizon in range(6):
                    count=max(n-horizon-1,0)
                    future[:count,horizon]=actions[horizon+1:];valid[:count,horizon]=True
                ident='task%d_demo%d'%(task,demo)
                npz(a.out/(ident+'_raw.npz'),state=state,actions=actions,future=future,future_valid=valid,
                    eef=robot[:,2:5],quat=robot[:,5:9],target_xyz=target_xyz,target_valid=target_valid,
                    target_relative=target_xyz-robot[:,2:5])
                demos.append(dict(id=ident,task=task,demo=demo,n=n,offset=offset,hdf5=str(path),
                    language=name.replace('_',' '),target_joint=target,target_qpos_adr=target_adr,
                    raw_sha256=sha(a.out/(ident+'_raw.npz')),
                    same_index_qpos_max_error=float(np.max(abs(sim[:,1:10]-state))),
                    next_index_qpos_max_error=0.,future_missing_by_q={str(q):int((~valid[:,:q].all(1)).sum()) for q in range(1,7)}))
                offset+=n
                if task<6:
                    all_state.append(state);all_actions.append(actions)
    state=np.concatenate(all_state);act=np.concatenate(all_actions)
    scale=np.maximum(np.quantile(act[:,:6],.75,axis=0)-np.quantile(act[:,:6],.25,axis=0),.1)
    write(a.out/'NORMALIZATION.json',dict(state_mean=state.mean(0).tolist(),state_std=np.maximum(state.std(0),1e-6).tolist(),
        action_scale=scale.tolist()+[2.],fit_tasks=list(range(6)),fit_demos=list(range(5)),n_frames=len(state)),immutable=True)
    write(a.out/'DATA_MANIFEST.json',dict(demos=demos,frames=offset,assets=assets,protocol_sha256=sha(a.out/'PROTOCOL.json')),immutable=True)
    write(a.out/'TIME_ALIGNMENT_AUDIT.json',dict(status='PASS',demos=len(demos),exact_obs_t_equals_sim_t_plus_1=True,
        object_qpos_from='states[t+1], last observation has no saved corresponding simulator state',
        future_action_start='actions[t+1]',native_training_alignment_modified=False,
        past_B_q3_comparison='B used actions[t:t+3] per its frozen protocol. This new genuine-future pilot is not numerically identical; historical results untouched.',
        source=protocol['state_time_evidence_url'],formal_training_resumed=False))
    print(json.dumps(dict(status='PREPARED',demos=len(demos),frames=offset)),flush=True)

if __name__=='__main__': main()

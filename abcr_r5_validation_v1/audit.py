"""Multi-frame blinded review pack plus exact saved-state contact reconstruction."""
import argparse,gc,html,importlib.util,xml.etree.ElementTree as ET
from pathlib import Path
import h5py,numpy as np
from PIL import Image,ImageDraw
from .common import *

def remap_xml(xml,repo):
    root=ET.fromstring(xml);maps=[];rs=Path(importlib.util.find_spec('robosuite').origin).parent
    for e in root.iter():
        p=e.get('file')
        if p is None:continue
        if '/chiliocosm/assets/' in p:q=repo/'libero/libero/assets'/p.split('/chiliocosm/assets/',1)[1]
        elif '/robosuite/' in p:q=rs/p.split('/robosuite/',1)[1]
        else:q=Path(p)
        q=q.resolve();assert q.is_file(),('MISSING_XML_ASSET',str(q));maps.append(dict(original=p,resolved=str(q)));e.set('file',str(q))
    return ET.tostring(root,encoding='unicode'),maps

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();dest=a.out/'audit';dest.mkdir(exist_ok=True)
    import mujoco
    rows=read(a.out/'AUDIT_SELECTION_PRIVATE.json')['pairs'];dm=read(a.repo/'reports/abcr_object_pilots_v1/DATA_MANIFEST.json')['demos'];metadata={(r['task'],r['demo']):r for r in dm};cards=[];physics=[];assets={}
    for r in rows:
        complete=dest/(r['audit_id']+'.json')
        if complete.exists():
            cached=read(complete);assert cached['image_sha256']==sha(dest/(r['audit_id']+'.png'));physics.append(cached);cards.append(Image.open(dest/(r['audit_id']+'.png')).convert('RGB'));continue
        im=Image.new('RGB',(1152,590),'white');d=ImageDraw.Draw(im);d.text((8,2),r['audit_id']+' | frames relative to anchor: -4 / 0 / +4 | each: agent + wrist',fill='black');record={'audit_id':r['audit_id'],'sides':{}}
        for side,y,task in [('new',25,r['task']),('old',305,r['old_task'])]:
            m=metadata[task,r[side+'_demo']];d.text((4,y),side.upper()+': '+r[side+'_language'],fill='black');timeline=[]
            with h5py.File(r[side+'_hdf5'],'r') as h:
                demo=h['data/demo_%d'%r[side+'_demo']];cache=(r[side+'_hdf5'],r[side+'_demo'])
                xml,maps=remap_xml(demo.attrs['model_file'],a.repo);model=mujoco.MjModel.from_xml_string(xml);assets[str(cache)]=maps
                data=mujoco.MjData(model);n=len(demo['actions']);target=m['target_joint'].replace('_joint0','')
                for col,off in enumerate([-4,0,4]):
                    t=r[side+'_t']+off
                    if t<0 or t>=n:timeline.append(dict(offset=off,status='OUT_OF_RANGE'));continue
                    for k,v in enumerate(['agentview_rgb','eye_in_hand_rgb']):
                        pic=Image.fromarray(np.flipud(demo['obs/'+v][t]).copy()).resize((192,192));im.paste(pic,(col*384+k*192,y+22))
                    d.text((col*384+3,y+216),'frame %d'%t,fill='black')
                    if t+1>=n:timeline.append(dict(offset=off,status='NO_SAVED_POST_STATE'));continue
                    state=demo['states'][t+1];assert len(state)==1+model.nq+model.nv;data.time=state[0];data.qpos[:]=state[1:1+model.nq];data.qvel[:]=state[1+model.nq:];mujoco.mj_forward(model,data)
                    obsstate=np.r_[demo['obs/joint_states'][t],demo['obs/gripper_states'][t]];assert np.array_equal(data.qpos[:9],obsstate)
                    contacts=[]
                    for c in data.contact[:data.ncon]:
                        n1=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(c.geom1)) or '';n2=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(c.geom2)) or ''
                        if target in n1 or target in n2:contacts.append(dict(geom1=n1,geom2=n2,distance=float(c.dist)))
                    finger=any('gripper' in c['geom1'] or 'gripper' in c['geom2'] for c in contacts)
                    xyz=data.qpos[m['target_qpos_adr']:m['target_qpos_adr']+3].copy();eef=demo['obs/ee_pos'][t]
                    rr=dict(offset=off,frame=t,status='RECONSTRUCTED_FORWARD_ONLY',target_xyz=xyz.tolist(),eef_xyz=eef.tolist(),target_relative=(xyz-eef).tolist(),target_contacts=contacts,target_gripper_contact=finger,n_contacts=int(data.ncon),state_sha=hashlib.sha256(state.tobytes()).hexdigest());timeline.append(rr)
                    text='grip contact: '+str(finger)+' | rel '+','.join('%.2f'%z for z in xyz-eef);d.text((col*384+3,y+232),text,fill='black')
                record['sides'][side]=dict(language=r[side+'_language'],timeline=timeline)
                del data,model;gc.collect()
        p=dest/(r['audit_id']+'.png');im.save(p);record['image_sha256']=sha(p);write(complete,record);physics.append(record);cards.append(im)
        print('AUDIT',r['audit_id'],flush=True)
    for i in range(0,len(cards),3):
        board=Image.new('RGB',(1152,590*len(cards[i:i+3])),'white')
        for j,im in enumerate(cards[i:i+3]):board.paste(im,(0,j*590))
        board.save(dest/('board_%02d.png'%(i//3+1)))
    write(a.out/'SIMULATOR_RECONSTRUCTION.json',dict(mujoco=mujoco.__version__,records=physics,xml_asset_remaps=assets,simulation_steps=0,blinding='No action labels or previous AI verdicts. Geometric/contact evidence visible.'))
    write(dest/'BLINDED_MANIFEST.json',dict(records=physics,reviewer='AI_NOT_HUMAN',protocol_sha=sha(a.out/'PROTOCOL.json')))
    atomic_text(dest/'index.html','<!doctype html><meta charset="utf-8"><title>R5 AI audit</title><h1>R5 multi-frame audit</h1><p>COMPARABLE / NOT_COMPARABLE / UNCERTAIN; target interaction phase, motion and contact. No action labels.</p>'+''.join('<section><h2>'+r['audit_id']+'</h2><img width="1152" src="'+r['audit_id']+'.png"></section>' for r in rows))
    table(dest/'LABEL_TEMPLATE.csv',[dict(audit_id=r['audit_id'],label='',new_phase='',old_phase='',reason='',reviewer_type='AI',image_sha256=sha(dest/(r['audit_id']+'.png'))) for r in rows])

if __name__=='__main__':main()

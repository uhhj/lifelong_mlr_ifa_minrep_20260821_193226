"""Paired, teacher-anchored inference on disk recordings. Only the raw six-axis ODE output is scored."""
from inputs import *
from collections import Counter
MODES=('train_zero','component_a','component_b','aug_a','aug_b')
def timing(rows):
    value={k:[] for k in ('skew','fb_age','cmd_age','target_joints_age','fb_joints_age','overhead_to_fb','wrist_to_fb','overhead_to_cmd','wrist_to_cmd')}
    for r in rows:
        now=r['sample_monotonic_ns'];a=r['cameras']['overhead']['monotonic_ns'];b=r['cameras']['wrist']['monotonic_ns']
        fb=r['feedback'];cmd=r['target']
        f=fb['sample_monotonic_ns'];c=cmd['sample_monotonic_ns']
        for k,x in zip(value,(abs(a-b),now-f,now-c,now-min(cmd['joint_bus_monotonic_ns']),now-min(fb['component_monotonic_ns'][:3]),a-f,b-f,a-c,b-c)):
            value[k].append(x/1e6)
    return {k:max(v) for k,v in value.items()}
def evaluate(train,batch,grid):
    obs=batch['obs'];anchor=batch['action'][:,7]
    def pred(**kwargs):return train.predict_action(obs,a0=anchor,**kwargs)['action'][0]
    z=pred(eps=torch.zeros_like(anchor))
    torch.manual_seed(2026);a=pred(eps_center=True)
    result={'train_zero':z,'component_a':a}
    if grid:
        torch.manual_seed(2027);result['component_b']=pred(eps_center=True)
        crops=[train.obs_encoder.key_transform_map[k][1] for k in ('overhead','wrist')]
        for mod in crops:mod.train()
        try:
            for seed,name in ((2026,'aug_a'),(2027,'aug_b')):
                torch.manual_seed(seed);result[name]=pred(eps=torch.zeros_like(anchor))
        finally:
            for mod in crops:mod.eval()
    return result
def end_vec(v):return np.asarray(v).round(6).tolist()
@torch.inference_mode()
def run(group):
    train,station,stats=init();ref=read(last/('windows_'+group+'.json'));entries={}
    real_ids={r['id'] for r in read(cache/'metadata.json')['episodes']}
    for ep,rec in subjects(group):
        src=ref[ep];wanted={r['t']:r for r in src['windows']};hi=max(wanted)
        loader,seq,good,file_err=episode_arrays(rec,hi+9)
        checks={};result=[];rp.reset(station)
        # fresh() reuses a graph only when the plan really started on this timestamp.
        for t in range(hi+1):
            if not good[t]:
                rp.reset(station);continue
            fb=np.array(rp.at(rec['rows'][t],'feedback'),np.float32)
            anchor=np.array(rp.at(rec['rows'][t-1],'target'),np.float32) if t and good[t-1] else fb
            stamp=rec['rows'][t]['sample_monotonic_ns']
            views=[rp.imaging(rec,t,k) for k in ('overhead','wrist')]
            visual_step(station,views,stamp,fb.tolist(),anchor.tolist())
            p,s=station.history[-1];assert s.shape==(7,)
            for i,(role,key) in enumerate(zip(roles,('head_camera','wrist_camera'))):
                actual=torch.from_numpy(p[i]).cuda().permute(2,0,1).contiguous()
                expected=loader.x[key][t]
                maxes(checks,**{role+'_u8':layer_delta(actual.float(),expected.float())})
                fv=actual[None].float()/255;tv=expected[None].float()/255
                act=station.model.obs_encoder.key_transform_map[role](fv)
                exp=train.obs_encoder.key_transform_map[role](tv)
                mean=tv.new_tensor([.485,.456,.406])[None,:,None,None];std=tv.new_tensor([.229,.224,.225])[None,:,None,None]
                assert act.shape==(1,3,112,112)
                maxes(checks,**{role+'_transformed':layer_delta(act,exp),role+'_manual_center':layer_delta(exp,(tv[:,:,8:120,8:120]-mean)/std)})
            if t not in wanted:continue
            w=wanted[t]
            assert all(good[t-7:t+9]) and station.trace['ready']
            batch=loader.batch(torch.tensor([t],device='cuda'))
            state0=torch.from_numpy(np.stack([x for _,x in station.history])).cuda()
            control=torch.tensor([rp.at(rec['rows'][i],'target') for i in range(t,t+8)],device='cuda')
            first=torch.from_numpy(anchor).cuda().unsqueeze(0)
            maxes(checks,state=layer_delta(batch['obs']['agent_pos'][0],state0),anchor=layer_delta(batch['action'][:,7],first),labels=layer_delta(batch['action'][0,8:],control))
            for key,v,u in [('action',batch['action'][:,7:],torch.cat((first[:,None],control[None]),1)),('agent_pos',batch['obs']['agent_pos'],state0[None])]:
                norm=train.normalizer[key];observed=norm.normalize(v);other=station.model.normalizer[key].normalize(u)
                low=v.new_tensor(stats[key]['min']);high=v.new_tensor(stats[key]['max']);rng=high-low
                scale=torch.where(rng<1e-4,1.,2/rng.clamp_min(1e-4));off=torch.where(rng<1e-4,-low,-1-scale*low)
                maxes(checks,**{key+'_norm':layer_delta(observed,other),key+'_manual':layer_delta(observed,v*scale+off),key+'_roundtrip':layer_delta(norm.unnormalize(observed),v)})
            pred=station.base_forecast(anchor)
            px=xyz(station,pred);origin=xyz(station,np.asarray(rp.at(rec['rows'][t],'feedback')))
            forecast=evaluate(train,batch,w['grid']);delta={}
            for name,v in forecast.items():
                full=output_xyz(station,v);ends=full[-1]-origin;diff=full-px
                dq=v.cpu().numpy()-pred
                delta[name]=dict(end=end_vec(ends),tip_max=float(np.linalg.norm(diff,axis=1).max()),tip8=float(np.linalg.norm(diff[-1])),
                     joint_deg=float(np.rad2deg(abs(dq[:,:6])).max()),jaw_mm=float(abs(dq[:,6]).max()))
            maxes(checks,prior_ref=float(np.linalg.norm(px[-1]-origin-w['raw'])))
            relative=timing(seq[t-7:t+8])
            result.append(dict(t=t,phase=w['phase'],grid=w['grid'],station=end_vec(px[-1]-origin),variants=delta,timing=relative))
        entries[ep]=dict(in_train=ep in real_ids,windows=result,checks=checks,source_file_stamp_ns=file_err)
        if len(entries)%4==0:print(group,len(entries),flush=True)
    write('paired_'+group+'.json',entries);print('complete',group,len(entries),sum(len(e['windows']) for e in entries.values()),flush=True)
if __name__=='__main__':run(sys.argv[7])

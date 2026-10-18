"""Read-only supplement using saved isolation update; zero optimizer/environment calls."""
from .model import *
def main():
    assert not (O/'COST_ISOLATION.json').exists();setup();p,t0=new_public();load_stage(p,7)
    baseline,_=make_bank(7);trained=copy.deepcopy(baseline)
    s=torch.load(O/'integration_recovery.pth',map_location='cpu');trained.load_state_dict(s['bank'])
    assert all(torch.equal(a,b) for a,b in zip(baseline.branches['6'].parameters(),trained.branches['6'].parameters()))
    assert any(not torch.equal(a,b) for a,b in zip(baseline.branches['7'].parameters(),trained.branches['7'].parameters()))
    h=torch.cat([torch.load(f,map_location='cpu')['feature'] for f in sorted((P/'coreset/buffer').glob('t6__*.pt'))[:10]]).cuda()
    with torch.no_grad():
        before=decode(p,t0,baseline,h);after=decode(p,t0,trained,h)
        diff=float((before['means']-after['means']).abs().max());dd=float((before['delta']-after['delta']).abs().max())
        assert max(diff,dd)<=1e-6
        latencies=[]
        for length in [1,4,8]:
            hh=h[:,:length].contiguous()
            for _ in range(3):decode(p,t0,trained,hh)
            def measure(fn):
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(10):fn()
                end.record();torch.cuda.synchronize();return begin.elapsed_time(end)/10
            a=measure(lambda:p.policy_head(p.temporal_encode(hh)[:,-1],train=False))
            b=measure(lambda:selected_dist(decode(p,t0,trained,hh)))
            latencies.append(dict(batch=10,history=length,parent_temporal_head_ms=a,zrv2_temporal_head_ms=b,overhead_ms=b-a))
    write(O/'COST_ISOLATION.json',dict(status='PASS',actual_task7_only_update_other_task6_mean_max=diff,cross_delta_max=dd,latency=latencies,note='Real retained latent, no spatial encoder or environment in timing; ten repeated forwards, zero optimizer calls.',T0_parameters=sum(p.numel() for p in t0.parameters()),T0_bytes=sum(v.numel()*v.element_size() for v in t0.state_dict().values()),gpu_peak_allocated=torch.cuda.max_memory_allocated(),optimizer_calls=0,environment_steps=0))
    print('COST_ISOLATION PASS; zero updates/rollouts')
if __name__=='__main__':main()

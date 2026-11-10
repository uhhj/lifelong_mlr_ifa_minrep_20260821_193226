"""Reuse accepted S's single-frame encoding + 8-frame latent history.
Keep this run's original Normal sampling, global RNG and done handling.
Numerical admission uses the accepted S tolerance, not bitwise equality.
"""
import torch,collections,copy

def encode(front,d):
    x={'obs':{k:v.clone() for k,v in d['obs'].items()},'task_emb':d['task_emb'].clone(),'masks':d['masks'].clone()}
    return front.spatial_encode(front.preprocess_input(x,train_mode=False))

def cached_get_action(self,data):
    self.eval()
    with torch.no_grad():
        n=next(iter(data['obs'].values())).shape[1]
        assert 1<=n<=8
        # Original evaluator supplies the complete raw history. Encode only its
        # newest frame; preserve original temporal/head/sample operation order.
        newest=dict(obs={k:v[:,-1:].contiguous() for k,v in data['obs'].items()},task_emb=data['task_emb'],masks=data['masks'])
        self.latent_queue.append(encode(self,newest))
        self.latent_queue=self.latent_queue[-8:]
        assert len(self.latent_queue)==n,'History/reset mismatch'
        x=torch.cat(self.latent_queue,dim=1)
        x_temp=self.temporal_encode(x)
        dist=self.policy_head(x_temp[:,-1,:],train=False)
        action=dist.sample().detach().cpu()
        return action.view(action.shape[0],-1).numpy()

def admission(p,shape,instr,mask,rng,restore_rng):
    """Same 1..8 history, feature/GMM-parameter 1e-4 tests as accepted S."""
    saved=rng();was_training=p.training;p.eval();before={k:v.detach().clone() for k,v in p.state_dict().items()}
    try:
        gen=torch.Generator().manual_seed(20261024)
        obs={k:torch.rand((2,8,*v),generator=gen).cuda() for k,v in shape['all_shapes'].items()}
        base=dict(task_emb=instr.repeat(2,1).cuda(),masks=mask.repeat(2,1).cuda())
        with torch.no_grad():
            frames=[encode(p,dict(obs={k:v[:,i:i+1] for k,v in obs.items()},**base)) for i in range(8)]
            rows=[]
            for n in range(1,9):
                whole=encode(p,dict(obs={k:v[:,:n] for k,v in obs.items()},**base));cached=torch.cat(frames[:n],1)
                assert torch.allclose(whole,cached,atol=1e-4,rtol=1e-4)
                q0=p.policy_head.forward_fn(p.temporal_encode(whole)[:,-1]);q1=p.policy_head.forward_fn(p.temporal_encode(cached)[:,-1])
                assert all(torch.allclose(x,y,atol=1e-4,rtol=1e-4) for x,y in zip(q0,q1))
                rows.append(dict(length=n,max_feature_abs=float((whole-cached).abs().max()),max_param_abs=max(float((x-y).abs().max()) for x,y in zip(q0,q1))))
        assert all(torch.equal(before[k],v) for k,v in p.state_dict().items())
        return dict(status='PASS_ACCEPTED_S_CACHE_TOLERANCE',atol=1e-4,rtol=1e-4,cache_parity=rows,frontend_calls=16,policy_parameter_calls=16,optimizer_calls=0,environments=0,model_unchanged=True,bitwise_equivalence_claim=False)
    finally:
        p.latent_queue=[];p.train(was_training);restore_rng(saved)

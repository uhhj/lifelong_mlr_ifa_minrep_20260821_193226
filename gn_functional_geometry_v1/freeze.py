from .common import *
import subprocess
p=OUT/'PROTOCOL.json'
assert not p.exists()
z=dict(name='gn_functional_geometry_v1',git_HEAD=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),old_task=8,new_task=9,seed=45,
 rank=32,primary_directions=16,H1_top_directions=8,scales=[.25,.5,1.],primary_scale=1.,
 epsilon=read(OUT/'UPDATE_SCALE.json')['median_update_norm'],dtype='frozen float32 decoder latents; float64 head for all diagnostics',
 geometry='terminal GMM expected action; variance branch is invisible to this geometry; not full-distribution Fisher',
 eigensolver=dict(method='matrix-free scipy eigsh',k=32,ncv=72,tol=1e-6,maxiter=500,seeds=[2026092303,2026092304]),
 numerics=dict(orthogonal_max=2e-5,eigen_relative_residual_max=.002,repeat_min_principal_cosine=.99,trace_excess_relative=1e-5),
 action_metric=dict(groups=['translation 0:3','rotation 3:6','gripper 6'],source='old_build terminal expert actions',scale_floor=.05,equal_group_weight=True),
 split_sha256=sha(OUT/'DATA_SPLIT.json'),scale_sha256=sha(OUT/'UPDATE_SCALE.json'),
 H1=dict(spearman_min=.6,top_random_median_ratio_min=2.,top_complement_median_ratio_min=2.,epsilon_rank_correlation_min=.9,signs=[-1,1]),
 H2=dict(trace='exact streamed scalar output gradients; no full J stored',rank90_max=32,adjacent_minus_cross_median_min=.01,all_build_demo_adjacent_minus_cross_positive=True,far_fraction=.2),
 H3=dict(gradient_roles=['old_gradient','new_gradient'],evaluation_roles=['old_probe','new_probe'],weak_relative_gradient_norm=1e-6,loss_effect_tolerance=1e-9,
  perturbation='delta=-epsilon*sign(z_new)*unit_direction; eta_k=epsilon/abs(z_new); weak directions not amplified',
  numerical_weak_directions_remain_in_denominator=True,minimum_nonweak=8,minimum_each_class=3,sign_accuracy_min=.75,spearman_min=.6,
  shared_joint_fraction_min=.7,conflict_joint_fraction_min=.7,accuracy_advantage_over_random_min=.1,spearman_advantage_over_random_min=.1,
  random_baseline='unconditioned orthonormal rank32, first16 directions at same norm',raw_baseline='whole new-gradient descent only; one checkpoint cannot establish statistical predictive superiority',
  matched_probe_derivative='secondary Taylor calibration only, never used to label heldout records'),
 threshold_status='prospective engineering screening rules, not theoretical guarantees; frozen before outcome measurement',
 H2_fingerprint_limit='nonnegative squared responses omit direction signs and within-action orientation',
 support_scope='one development seed, one task pair, head only; no independent confirmation or closed-loop evidence',
 training_active=False,formal_solver_modified=False,model_writeback=False,optimizer_steps_executed=0)
write(p,z)
print('PROTOCOL_FROZEN',sha(p))

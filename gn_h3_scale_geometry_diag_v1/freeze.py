from .common import *
import shutil,subprocess
assert not (OUT/'PROTOCOL.json').exists()
for name in ['EXPERIMENT_CONTEXT.json','DATA_SPLIT.json','ACTION_METRIC.json','UPDATE_SCALE.json']:
 shutil.copy2(OLD/name,OUT/name)
z=dict(name='gn_h3_scale_geometry_diag_v1',parent_HEAD=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
 parent_protocol_sha256=sha(OLD/'PROTOCOL.json'),checkpoint_sha256=read(OLD/'EXPERIMENT_CONTEXT.json')['checkpoint_sha256'],
 mean_basis_path=str(OLD/'OLD_TASK_GN_BASIS.npz'),mean_basis_sha256=sha(OLD/'OLD_TASK_GN_BASIS.npz'),split_sha256=sha(OUT/'DATA_SPLIT.json'),
 geometry=['MEAN_GN_REUSED','MODEL_FISHER_MC8'],rank=16,scales=[1.,.25,.0625,.015625],epsilon0=read(OLD/'PROTOCOL.json')['epsilon'],
 perturbation='Identical previous rule: delta=-epsilon0*scale*sign(u.T@g_new_gradient)*u, eta_k=epsilon/abs(z_new); no optimizer updates.',
 primary_predictor='g_old_probe.T@delta; exact same held-out windows as measured NLL, derivative calibration only',
 secondary_predictor='g_old_gradient.T@delta; trajectory-disjoint transfer prediction',
 fisher=dict(definition='E_H E_a~p_theta0(a|H)[score(a,H) score(a,H).T]',approximation='8 frozen independent model action samples per old_build window; 128 windows; no gradient through sampled actions',
  samples_per_window=8,sampling_seed=2026092306,action_metric='full official GMM log density; no Wa rescaling, label replacement or empirical expert-score approximation',
  covariance_centering=False,interpretation='Model Fisher MC estimate, NOT claimed equal to expert NLL Hessian or exact GGN; includes mixture means, weights and scales'),
 eigensolver=dict(k=16,ncv=40,tol=1e-6,maxiter=500,seeds=[2026092307,2026092308]),
 numerical=dict(max_orthogonal_error=2e-5,max_eigen_relative_residual=.002,min_repeat_principal_cosine=.99),
 weak_new_projection_relative=1e-6,loss_effect_tolerance=1e-9,
 rules=dict(good_scale='16 numerically evaluable directions, conservative sign accuracy >=.9375 (15/16), Spearman>=.9, Pearson>=.9',
  local_recovery='smallest scale good, sign accuracy improves >=.125, relative linearization error drops >=4x',
  material_geometry_advantage='at least two pre-fixed scales improve accuracy >=.125 and Spearman >=.15, without lower evaluable coverage',
  limits='Descriptive engineering criteria; not statistical significance. If both smallest-scale calibrations fail: unresolved Taylor regime/measurement, NOT evidence that gradient directional derivatives are invalid.'),
 no_rank_or_step_tuning=True,no_new_sampling_of_training_windows=True,no_formal_changes=True,training_active=False,formal_training_resumed=False,
 sources=['https://www.jmlr.org/papers/v21/17-678.html'],
 limitations=['Fisher Monte Carlo approximation is one fixed draw set, not repeated reference estimation.','The two eigenspaces change direction identity and functional displacement at equal parameter norm; this comparison cannot uniquely prove previous functional definition was wrong.','Same-probe Taylor agreement is not cross-trajectory relation generalization.','No control, optimizer or rollout benefit tested.'])
write(OUT/'PROTOCOL.json',z);print('FROZEN',sha(OUT/'PROTOCOL.json'))

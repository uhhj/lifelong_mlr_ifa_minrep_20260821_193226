from .common import *
import shutil,subprocess
assert not (OUT/'PROTOCOL.json').exists()
for name in ['EXPERIMENT_CONTEXT.json','DATA_SPLIT.json','ACTION_METRIC.json','UPDATE_SCALE.json']:shutil.copy2(OLD/name,OUT/name)
assets={name:dict(path=str(path),sha256=sha(path)) for name,path in [('mean_basis',OLD/'OLD_TASK_GN_BASIS.npz'),('fisher_basis16',FISH/'NLL_MODEL_FISHER_BASIS.npz'),('fisher_draws',FISH/'FISHER_MODEL_DRAWS.npz')]}
p=dict(name='gn_h3_subspace_closure_v1',parent_HEAD=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),assets=assets,split_sha256=sha(OUT/'DATA_SPLIT.json'),epsilon0=read(OLD/'PROTOCOL.json')['epsilon'],
 A=dict(scope='Same true checkpoint, official GMM NLL, head float64 and fixed float32 decoder features, both old_probe and new_probe',
  directions='Existing mean top16, existing Fisher top16, normalized old_probe gradient, normalized new_gradient gradient, one fixed random unit direction: 35 total',
  h_factors=[1.,.1,.01,.001,.0001,.00001,.000001,.0000001],formula='mean(per-window L(theta+h*v)-L(theta-h*v))/(2*h)',
  tolerance='relative error <=1e-4, denominator max(abs(g.T@v),1e-8*norm(g)); report weak derivatives separately',
  pass_rule='Every one of 70 direction/task groups closes at two adjacent pre-fixed scales with h_factor<=1e-3. No new h grid after outcomes.',
  failure_action='Stop before B; preserve all measurements, no optimization or training'),
 B=dict(geometries=['MEAN_GN','MODEL_FISHER_MC8','RANDOM'],ranks=[4,8,16,21],
  mean_basis='Reuse existing top32; do not rebuild',fisher_basis='Same saved 1024 model draws and operator; extend top16 to top21, keep original 16 columns, save five new columns only',
  fisher_extension_solver=dict(k=21,ncv=48,tol=1e-6,maxiter=500,seed=2026092309),
  random_basis_seed=2026092310,rotation_seed=2026092311,
  projector='P=U @ solve(U.T@U,U.T@v); correct finite-precision non-orthogonality without changing span',
  score='g_old_gradient.T @ P @ g_new_gradient; matched old_probe score recorded separately',
  perturbation='delta=-epsilon*(P@g_new)/norm(P@g_new); common norm across all spaces, epsilon=epsilon0*scale. Record unnormalized score and projection energy too.',
  scales=[1.,.015625],scale_reason='Reuse only previous endpoints, normal scale and 1/64 local check; no additional step search',
  predicted_change='-epsilon*score/norm(P@g_new), not score itself',
  invariance='For every r/space rotate by frozen orthogonal r x r matrix; projector vector, score and perturbation must agree to relative 1e-10',
  weak_projection_relative=1e-8,loss_effect_tolerance=1e-9,
  aggregation='Four nested ranks are correlated; report exact per-rank outcomes, not independent trials or significance. One random nested basis only.',
  comparison='At each scale, compare conservative sign counts, joint old/new outcomes, Spearman and relative Taylor error; ties or mixed tradeoffs mean no established advantage. No controller gate.'),
 training_active=False,formal_training_resumed=False,optimizer_steps_executed=0,formal_solver_modified=False,
 limits=['B is a normalized projected-gradient direction test, not real AdamW and not a raw unconstrained gradient step.','Basis invariance holds within each fixed span, not when truncation changes which span is chosen.','Fisher is a fixed model-score MC estimate, not expert NLL Hessian.','A validates local derivatives only; does not guarantee a given B step is within the linear regime.','No sample, task, seed, reference or rank tuning based on outcomes.'])
write(OUT/'PROTOCOL.json',p);print('FROZEN',sha(OUT/'PROTOCOL.json'))

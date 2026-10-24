from pathlib import Path
import json,hashlib,ast
R=Path(__file__).resolve().parent;P=R.parent/'20261024_align_behavior_drift'
ins=json.loads(next(R.glob('inspect_readonly_*/inspect.json')).read_text())
spec=json.loads((R/'INPUT_SPEC.json').read_text())
for e,x in zip(spec['seeds'],ins['entries']):
 assert e['seed']==x['seed']
 e['ra_checkpoints']=x['ra_cp_sha256']
 # Original model implementations are byte-identical across these contracts.
 for n in ['libero/lifelong/models/brief.py','libero/lifelong/models/policy_head.py','libero/lifelong/models/modules/transformer_modules.py']:
  assert x['contract']['source_sha256'][n]==hashlib.sha256(json.loads(next(P.glob('preflight_readonly_*/preflight.json')).read_text())['source'][n].encode()).hexdigest()
(R/'INPUT_SPEC.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
t=(P/'drift_readonly_v2.py').read_text()
t=t.replace("S=Path(entry['source']);tic", "S=Path(entry['source']);RA=Path(entry['ra_output']);tic")
t=t.replace("cp=O/'ra_coreset/models'", "cp=RA/'ra_coreset/models'")
t=t.replace("entry['checkpoints'][str(stage)]", "entry['ra_checkpoints'][str(stage)]")
t=t.replace("sha(O/'ra_coreset/models'/f'mlrifa_{stage}_model_ep99.pth')", "sha(RA/'ra_coreset/models'/f'mlrifa_{stage}_model_ep99.pth')")
t=t.replace("verify_source()\n assert (O/", "verify_source()\n assert sha(RA/'CONTRACT.json')==entry['ra_contract_sha256']\n assert js(RA/'PROCESS_EXIT_ACCEPTANCE.json')['status']=='PASS'\n assert js(RA/'ra_coreset/storage/task8.json')['selected']==js(O/'ra_coreset/storage/task8.json')['selected']\n assert (O/")
# source/destination path fields differ; compare exact identity/hash list only.
t=t.replace("assert js(RA/'ra_coreset/storage/task8.json')['selected']==js(O/'ra_coreset/storage/task8.json')['selected']", "assert [(r['destination'],r['sha256']) for r in js(RA/'ra_coreset/storage/task8.json')['selected']]==[(r['destination'],r['sha256']) for r in js(O/'ra_coreset/storage/task8.json')['selected']]")
t=t.replace("assert sha(fp)==member['sha256']", "assert sha(RA/'ra_coreset/buffer'/name)==sha(fp)==member['sha256']")
t=t.replace("'contract_sha256':entry['contract_sha256']", "'contract_sha256':entry['ra_contract_sha256'],'common_align_contract_sha256':entry['contract_sha256']")
ast.parse(t);(R/'ra_drift_readonly.py').write_text(t,encoding='utf8')
(R/'PROTOCOL.md').write_text('''# RA/ALIGN 共同窗口与配对闭环诊断 v1

用户授权：先给RA补同样漂移诊断，再对已有Stage8/9进行固定初态、逐轨迹动作噪声、同历史双检查点闭环诊断。不得因切换或平均误差自行启动保护训练。

## 离线口径（推理前冻结）
GOAL seeds41/42/43，每seed同一452个Task8保留窗口；SHA/顺序与已接受ALIGN完全相同。加载原RA Stage8/9严格100键，复用原模型代码，CPU FP32 eval/no_grad单线程。全部漂移、分量切换、夹爪翻转及合法a[u+1]混合NLL/所选Normal NLL/分组Huber严格复用已接受ALIGN定义。RA原训练目标a[u]，共同a[u+1]误差用于相同比较，不称RA自身训练损失。无需重复ALIGN前向。预计174 batch前向/2712模型窗口，0优化器/反向/环境/CUDA/远程写入。独立本地复算，按窗口SHA配对。

## 闭环注册原则
另行冻结实施细节后启动有限新诊断，现有实验与检查点只读。研究Task8 Stage8→9保持；RA与ALIGN均作对照，所有三seed，初态0–19。两驱动策略各自生成轨迹，两个检查点在每条轨迹的同一实际历史上比较。逐轨迹独立标准正态噪声与固定环境seed，共享跨stage/method；不将不同驱动轨迹同时间步动作差当条件漂移。保留实际输入历史及失败前原始观测窗口在服务器私有位置，只发布标量/hash/outcomes。

成功定义仍是环境原成功信号；600步未成功称超时失败，不凭空定义第一次不可逆失败状态。分量编号变化非行为错误；有害变化须关联可重复成功→失败配对及同历史决策证据，仍不能仅靠相关证明单步因果。保护三组训练仅在完成分析后另行登记，不自动扫描阈值。
''',encoding='utf8')
(R/'RUN_STATUS.json').write_text(json.dumps({'status':'PARTIAL_IMPLEMENTATION','offline':'REGISTERED_NOT_RUN','closed_loop':'NOT_STARTED','training_updates':0},indent=2))
freeze={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['PROTOCOL.md','INPUT_SPEC.json','ra_drift_readonly.py']}
(R/'OFFLINE_FROZEN.json').write_text(json.dumps(freeze,indent=2))
for n,t in ins['sources'].items():
 p=R/'reference_source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(t,encoding='utf8')

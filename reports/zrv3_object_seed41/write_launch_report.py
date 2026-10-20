"""Render admitted run progress; cannot declare scientific success."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parent
D=sorted(R.glob('progress_*'))[-1];E=D/'evidence/v3'
def js(p):return json.loads(p.read_text())
m=js(D/'evidence/PACKAGE_MANIFEST.json')
assert all(hashlib.sha256((D/'evidence'/n).read_bytes()).hexdigest()==h['sha256'] for n,h in m.items())
ep=js(E/'epochs.json');st=[json.loads(l) for l in (E/'STRUCTURE.jsonl').read_text().splitlines()]
assert len(ep)==40 and sum(x['batches'] for x in ep)==10550 and sum(x['samples'] for x in ep)==105240
assert len(st)==16 and all(x['passed'] and x['unselected_means_exact'] and x['zero_residual_parent_exact'] for x in st)
assert max(x['actual_kl_max'] for x in st)<=.5+1e-12
assert (E/'calibrate.exit').read_text().strip()=='0'
text=f"""# ZR-v3 OBJECT seed41执行报告（PARTIAL）

证据时间：{datetime.datetime.fromtimestamp(js(D/'SNAPSHOT.json')['time']).astimezone().isoformat()}。
当前实际阶段：40校准epoch、10,550正式optimizer调用、105,240样本曝光全部完成，calibrate.exit=0；独立原协议评估进程已启动，当前下载快照正式记录为0/200。尚无新AUC或效果结论，不能标记实验COMPLETED。

复用已完成RA＋IFA seed41四阶段模型及完全相同的记忆；无公共策略重训、无新latent、无基线或ZR-v2复评。新UV从零输出起步，未加载旧ZR-v2 UV。原模型和数据源哈希验收通过。

实现变化仅为执行Gaussian NLL、仅argmax分量均值修正、固定epsilon=0.5 nat动作分布投影。训练与真实CompletePolicy.get_action调用同一函数。实际FP32分布而非投影前中间值接受KL核验，必要浮点收缩不放宽预算。SOURCE.patch记录相对v2模板的修改。

预检：25项继承软件；9项新核心CPU、9项新核心CUDA测试；真实语言门与缓存准入；恰好4次隔离GPU更新含恢复重放；真实图像1/4/8历史分布/选中分量/配对采样动作完全一致；0环境步骤。正式过程重新初始化RNG与UV。

16次全dataset结构检查全部PASS，零残差精确还原父均值，非选中分量保持，父参数/T0不变，自身/非自身门约束通过。最大实际KL={max(x['actual_kl_max'] for x in st):.15f}。四阶段末投影样本数：

| 阶段 | 投影样本 | 当前阶段实际dataset |
|---|---:|---:|
"""
for x in st:
    if x['label']=='stage_end':text+=f"| {x['stage']} | {x['projected_samples']} | {x['samples']} |\n"
text+=f"""
投影确实被触发，不是无效开关。但这只证明分布约束在已有缓存输入上成立，不证明闭环收益。各阶段执行分量NLL数值远高于旧混合NLL且保持有限，两个不同目标不能直接按数值比较；没有因此修改loss、阈值、学习率、梯度裁剪或预算。

校准epoch内实测总计{sum(x['seconds'] for x in ep):.2f}秒，包含额外全量结构/保存的正式校准总墙钟约4分钟。耗时短源于只训练小型UV残差且重用冻结latent，不能视为200条闭环已经完成。同GPU旧ZR-v2评估耗时86.25分钟，当前预计剩余约1.5–2小时，后续按真实cell吞吐修正。

固定门槛相对RA父：AUC>73.02083333%、FWT≥75%、OldFinal≥78.33333333%，并工程全PASS。通过才执行v3-A同10550step/200条的无投影消融；未通过停止。最多两个新变体，不扩其他seed。服务器有限队列脱离SSH运行；每小时自动检查，遇无法安全恢复的故障留证停止，完成验收及发布后删除定时任务。

真实执行命令与退出：

```text
/data/mlr_ifa/env/bin/python /data/mlr_ifa/experiment_workspaces/ZRv3_OBJECT_RA_seed41_20261020/server_runner.py prepare v3
/data/mlr_ifa/env/bin/python /data/mlr_ifa/experiment_workspaces/ZRv3_OBJECT_RA_seed41_20261020/server_runner.py preflight v3
/data/mlr_ifa/env/bin/python -u /data/mlr_ifa/experiment_workspaces/ZRv3_OBJECT_RA_seed41_20261020/supervisor.py
python execution/20261020_zrv3_object_seed41/check_progress.py
```

prepare/preflight均0；正式calibrate=0，evaluate及controller/supervisor尚未退出；本地轻量包哈希与历史控制指标独立复算PASS_PARTIAL。预检逐条命令与退出在证据preflight/commands.jsonl及*.exit，正式逐阶段命令在commands.jsonl。服务器产物/data/mlr_ifa/experiments/ZRv3_OBJECT_RA_seed41_20261020_v3。当前轻量证据：{D.name}。权重、RNG二进制、latent和buffer载荷不下载或发布。

解释边界：seed41是因已知v2负结果选取；单seed探索性；原轨迹动作RNG未隔离；新增校准计算不等预算；官方预训练state元数据unknown。完整v3达到门槛也须v3-A才能讨论信任域的额外作用，不能直接声称因果或跨seed有效。
"""
(R/'LAUNCH_REPORT.md').write_text(text,encoding='utf-8')
(R/'LOCAL_CALIBRATION_ACCEPTANCE.json').write_text(json.dumps(dict(status='PASS_CALIBRATION_ONLY',epochs=40,optimizer_steps=10550,exposures=105240,structure_checks=16,max_actual_KL=max(x['actual_kl_max'] for x in st),input_package_hashes_pass=True,final_experiment_complete=False,evidence=D.name),indent=2)+'\n')
print('PASS: calibration only; evaluation pending')

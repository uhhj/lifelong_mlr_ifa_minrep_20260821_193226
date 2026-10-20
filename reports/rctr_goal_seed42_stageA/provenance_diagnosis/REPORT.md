# RCTR 原采集资产与物理版本诊断

**结论：确认当前MuJoCo版本造成了实质模型/转移差异，修复了固定窗口的历史一步与8帧低维观察恢复，但完整历史续走仍未通过。保持BLOCKED；54条筛查0/54、训练0。**

## 来源搜索

- 服务器`/data`和`/root`搜索到20份HDF5及19个NPZ所在目录；没有找到原采集`state_*.npz`、未裁剪demo文件或这三条演示的完整控制器/积分快照。现有其他实验NPZ不能替代原演示状态。
- 本地历史MLR目录的10个匹配文件均是后续实验轨迹；检查170个相关归档未发现可用原采集资产。另行检查`Downloads/data.tar.gz`，它在438个目录项后截断，已读部分是PushL实验，不能宣称已完整搜索损坏部分。
- 当前Task6 HDF5的SHA256与官方HF LFS条目一致：`ee58a7d72c143a71bf1448674e70940ac7781c1d1bb10a1b0c7a5c69b26abbef`。有50演示、5349步；三个demo首时间均为0.25，未包含前五步。
- [官方采集代码](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/4f0dd188960c25b608babc7de34e1e20a9d1306f/scripts/collect_demonstration.py)最早公开于2023-06-07。原采集HDF5本可记录date/time及robosuite版本，但公开转换文件没有保留这些属性；即使该字段存在，它也不是MuJoCo版本。
- [转换程序](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/scripts/create_dataset.py)先step再存观察，裁前5步，states仍来自原记录。[robosuite1.4采集wrapper](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/wrappers/data_collection_wrapper.py)保存flat状态/动作，不保存完整控制器和积分状态。不能把“有states”说成“有完整快照”。
- 初始LIBERO requirements甚至未固定robosuite；robosuite1.4仅要求MuJoCo>=2.3.0。**尚未找到精确采集版本的可信锁文件。**

## 固定版本对照

结果前固定比较现用3.2.3与2.3.5。后者是LIBERO首次公开时已发布的候选，不是已证实的采集版本；没有尝试多个版本后挑最低误差。2.3.5单独安装在诊断目录，用独立进程加载；原环境仍3.2.3。

同一XML、同一动作、同一重建方式：

|演示|3.2.3目标一步误差|2.3.5目标一步误差|2.3.5历史8帧最大转移误差|2.3.5原生继续下一步误差|
|---|---:|---:|---:|---:|
|demo_0|0.00020087426|2.3120394e-14|1.0041273e-13|2.8657632e-14|
|demo_1|0.00020087426|5.9246878e-14|1.0540457e-12|7.5481288e-14|
|demo_2|0.060333349|2.409184e-14|3.350431e-12|0.033132738|

物理阈值仍1e-5，低维观察仍1e-6，重复恢复仍1e-10。误差为混合单位state坐标最大绝对值，不是米。三个窗口8帧低维观察最大误差均在约1e-15内；两次同完整快照输出误差均0。但demo_2紧接着的下一步失败，不能选择性只报告已通过的目标一步。

为排除动作控制器改变的混淆，还做了纯MuJoCo对照：相同XML、相同物理坐标和零初始隐状态、相同逐子步ctrl、相同mj_forward/mj_step调用，没有执行robot/controller。三个窗口版本间最终状态差分别为：
0.002854053233, 0.0002008742566, 0.1759129883。
这证明版本及其模型编译是实质差异来源；不能据此把所有剩余误差都归于某一个引擎bug。

同一XML编译出的质量也不同（kg）：

|body|3.2.3|2.3.5|
|---|---:|---:|
|akita_black_bowl_1_main|0.045899782|0.031340122|
|cream_cheese_1_main|0.012395263|0.012337579|
|wine_bottle_1_main|0.050916294|0.027561319|
|plate_1_main|0.03048326|0.024034048|
|wooden_cabinet_1_base|2.2409196|1.0141676|
|flat_stove_1_burner_plate|0.18497829|0.048327235|
|wine_rack_1_main|0.61639816|0.85165562|

最大质量差来自柜体；不能把它误写成碗质量差或断言它主导动态误差。碗、酒瓶、盘子、奶酪等也有差异；惯量、部分geom尺寸/姿态及mesh顶点形状亦不同。没有手工修改密度/质量/接触参数去拟合轨迹。

## 观察时序与剩余错误

在2.3.5中，用真正step返回的观察，三个目标EEF误差约0；随后仅调用forward并刷新observable，物理坐标不变，却使EEF对历史观察的差异变成约5.37e-5、3.14e-4、5.04e-5。历史观察包含原step返回时的派生量缓存时序，不能用“恢复qpos后刷新出的全部观察”替代它。这解释了上一轮关节精确而EEF仍有偏差的现象。原数据不重标，图像差异也没有以翻转、阈值放宽方式掩盖。

进一步区分本次保存问题与历史状态问题：

- demo_2不经过任何恢复、直接从重建目标状态继续原动作，仍误差0.033132738158055。
- 复制整个MuJoCo MjData并保留控制器/环境状态，得到同一误差和观察；因此**该失败不能归因于本次快照序列化漏掉某字段**。
- 显式积分状态恢复、强制控制器更新也未消除它；写入精确记录坐标仍误差0.03574。没有采用某种修补作为“通过”。
- 最大偏差在碗的角速度，随后有奶酪和机器人关节；demo_2更早的39/40步接触附近也有约0.0136/0.0144误差。前三条演示前4步共同的较大误差主要在夹爪，和前五步及累积夹爪状态缺失相容，但不构成唯一归因。
- 已知前缀注入记录坐标是诊断控制器的办法，不是已验证的无干预全轨迹重放。完整原始控制器/solver状态、真实采集构建和未裁剪前缀仍缺失，无法唯一分离它们与接触数值敏感性；不能声称全部原因已经查清。

这次没有加载父checkpoint或执行原生策略GMM准入，因为完整历史状态门槛仍失败；更没有启动54条、训练或B/C。保留原3窗口，不换简单窗口凑通过。

## 真实命令、退出与账目

- `bash provenance_diagnosis/run_versions.sh`：安装2.3.5退出0；3.2.3诊断退出0；2.3.5因缺少新版Python `mjtState`接口退出1，原controller.exit=1保留。
- 外置`PYTHONPATH=.../mujoco235:$PYTHONPATH python legacy_snapshot_adapter.py`：退出0。只用旧版公开字段保存积分输入，保留warmstart/ctrl/外力/mocap/userdata和控制器/环境/RNG；阈值未改。
- 同隔离环境`python controller_phase_probe.py`：退出0。
- 两个版本各`python physics_only.py`：退出均0；不用controller，共6控制步/150物理子步。
- `python finalize_diagnostic.py`及隔离环境`python physics_addendum.py`：退出0；输入及旧失败合同哈希不变，无残留worker。
- 本轮164+47（失败接口前）+164+170+6=**551工程控制步**；加既有644共1195。工程步骤不是episode。4次env构建/关闭、16次reset、另6次纯MuJoCo模型构建；策略续跑仍0/54，optimizer0。
- `python provenance_diagnosis/audit_local.py`：独立复算包哈希、账目、版本控制条件及未通过阈值。PASS只表示这份BLOCKED审计可信，不是实验准入PASS。

若要保持原标准继续，仍需要可核验原采集运行时/资产及这三条演示的未裁剪前缀或完整状态，或找到可验证的剩余工程根因。现有证据不足时不拟合参数、不改阈值、不用新采集轨迹代替原样本。

远端新证据目录：`/data/mlr_ifa/experiments/RCTR_GOAL_seed42_stageA_20261020/provenance_diagnosis`。
发布仅代码、报告、版本/资产哈希、计数及误差元数据，不含ctrl、物理state、模型、演示或噪声载荷。

本目录是独立的预注册确认实验包。协议见PROTOCOL.md；不修改历史算法文件。

服务器根目录：`/media/dm/Elements/wzp2026/lifelong_mlr_ifa_minrep`。

正式运行前依次完成seed审计、缓存准备和preflight。不要直接调用训练子命令绕过闸门。

```bash
cd /media/dm/Elements/wzp2026/lifelong_mlr_ifa_minrep
PYTHON=/home/dm/miniconda3/envs/wzp_mlr_tmp/bin/python
"$PYTHON" -m confirmation_experiment.progress
```

正式队列模块为 `confirmation_experiment.queue`。它要求SEED_AUDIT、FROZEN_INPUTS、PREFLIGHT_COMPLETE、IMPLEMENTATION均有效，且正式输出未启动；拒绝覆盖或自动重跑。主运行目录为runs/fixed_kappa_confirmation_v2，报告为reports/fixed_kappa_confirmation_v2。

`common.environment()`仅为子进程选择与当前内核匹配的已验证NVIDIA CUDA/EGL库，不安装或修改系统驱动。具体官方软件包哈希及实际EGL测试结果见GL_RUNTIME.json。

训练包装器仍直接调用历史MLR/MLRFixedKappa observe。前后指标、快照与等待闸门均不得改变随机数序列或优化更新；真实GPU严格parity是启动条件。eval_entry通过runpy执行原test.py，仅在第一格完成后等待，其20次trial不会重跑。

事件记录是算法本身原有g/r标量及代数推导的n信号；没有额外反向传播。恢复机制快照时，必须先核对stage plan内容哈希，再恢复model/optimizer/scheduler/计数/模块模式和pre-forward RNG；当前batch已保存。当前epoch的剩余batch直接按已保存索引取数，不能创建新的DataLoader iterator额外消耗RNG；到下一epoch才恢复原来的iterator创建步骤。

本轮只采集最多48个机制状态，不运行反事实机制实验。最终自动汇总先报告3个新seed，再历史3seed，最后6seed描述；结果决策只使用新seed。

服务器未配置GitHub凭证。实现提交与最终报告通过本机已登录的GitHub账号推送，
不向服务器复制令牌。`publish_relay.ps1`等待服务器完成报告commit，以Git bundle
取回该提交、核对SHA、推送并验证远端分支，再回执服务器。它不会修改训练或启动重跑。
本机必须保持运行和联网，最终自动推送才能完成；训练与评估在服务器独立运行。

The `frozen_runtime/` directory contains byte-for-byte audit copies of the actual
historical implementation, configuration and endpoint tests used here, including
legacy extensions that were not tracked at the starting commit. They are not
imported by the confirmation experiment. `MANIFEST.json` records their hashes;
preflight verifies every active runtime file against its preserved copy. To
reconstruct this exact runtime, apply those relative paths to a separate checkout
of the starting SHA, keeping the original experiment checkout untouched. Input
datasets/checkpoints/caches are referenced by hashes and are not bundled in Git.

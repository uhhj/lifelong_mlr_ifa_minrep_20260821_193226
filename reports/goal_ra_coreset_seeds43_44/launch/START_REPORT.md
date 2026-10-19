# RA-Coreset GOAL 43/44：启动与进度

状态：PARTIAL，证据时间2026-10-19T12:24:41.976604+08:00。不是最终验收。

本轮依据用户明确授权补完43、44，无论43性能是否提升都执行44；工程失败则停止留证，不盲重启。旧42和各自Random/Coreset基线只读，不重训重评。服务器RTX3090空闲时启动，系统盘约18.56GiB可用，沿用GOAL系统盘授权，无新付费资源。

科学代码与已完成42的selector/storage/model/train/evaluate保持字节一致；仅四个prepare/offline/freeze/audit文件参数化seed和来源。按真实8×7专家动作窗口为各seed重新绑定身份，禁止将42缓存改标签复用。原距离标定42、sampler0、训练/锚点/评估seed43或44、env[s,s+1]。详见SOURCE_DIFF.json和source.patch。

| seed | 状态 | 完整epoch | 累计step | 正式闭环 |
|---:|---|---:|---:|---:|
| 43 | TRAINING | 15/400 | 1500/76100 | 0/200 |
| 44 | QUEUED | 0/400 | 0/76100 | 0/200 |

每seed4ep99最终CP、200条正式/20vector；新增共800epoch、152200step、8CP、400条。三seed最终将对全部任务复算AUC/FWT/NBT/OldFinal/FinalAll、逐seed差、均值及样本SD(ddof1)，不省略负结果。

## 准入证据

- seed43：ADMISSION_PASS；4次隔离GPU原生更新与恢复、0环境；真实入库离线计划交换次数{'6': 121, '7': 106, '8': 100, '9': 249}。
- seed44：PENDING_ADMISSION_OR_COLLECTION；等待该组准入或完整证据收集，不能声称GPU通过。

初次环境探测两个父基线INDEPENDENT_ACCEPTANCE均PASS、各4最终CP存在、无科学进程运行。远端部署/启动命令及退出码在DEPLOY_COMMAND.json；队列启动PID2156，真实后续子进程以CURRENT_SNAPSHOT为准。部署退出0；当前queue/wrapper未退出时不宣称全实验完成。

后台流程：43准入→训练→评估→验收→44同流程→三seed汇总→退出验收；不依赖本地电脑持续连接。两小时自动化ra-coreset-goal-seed43-44-seed已创建，工程堵塞按冻结恢复合同处理，完整验收发布后删除。

耗时参照已完成42：训练40.56分钟、评估68.73分钟/seed，两个seed从正式启动起约3.5–4.5小时，尚非本轮完成承诺。后续按实际吞吐修正。

边界：本扩展在看到42结果后授权，属于探索性重复验证。三seed不自动等于显著；全局动作RNG未逐轨迹隔离，原路径排序会改变batch，IFA实际参考内容随记忆改变，官方pretrain state元数据unknown，原调度补全均保留披露。

发布仅代码、协议、身份/哈希/指标与episode轻量证据，不包含权重、原动作、latent、buffer载荷、RNG二进制或凭据。

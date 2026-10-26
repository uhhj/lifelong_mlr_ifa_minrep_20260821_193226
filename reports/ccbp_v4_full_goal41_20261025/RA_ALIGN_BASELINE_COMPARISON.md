# 以历史RA+ALIGN为基线的补充比较

同一GOAL seed41；只读复用已验收结果，0新训练/前向/环境。原C是RA+ALIGN上再加持续专家监督和margin路由的强对照，不能把C叫作原RA+ALIGN。

|方法|AUC|相对RA+ALIGN|OldFinal|FinalAll|相对RA+ALIGN|
|---|---:|---:|---:|---:|---:|
|RA_ALIGN|83.4375|+0.0000|75.0000|78.75|+0.00|
|C|87.5000|+4.0625|80.0000|82.50|+3.75|
|D|78.3333|-5.1042|71.6667|72.50|-6.25|
|O|87.5000|+4.0625|76.6667|80.00|+1.25|
|S|86.8750|+3.4375|81.6667|85.00|+6.25|

|方法|最终Task6|Task7|Task8|Task9|早期7/6|
|---|---:|---:|---:|---:|---:|
|RA_ALIGN|55.0|100.0|70.0|90.0|60.0|
|C|40.0|100.0|100.0|90.0|80.0|
|D|55.0|100.0|60.0|75.0|60.0|
|O|45.0|100.0|85.0|90.0|70.0|
|S|45.0|100.0|100.0|95.0|70.0|

换成历史RA+ALIGN参考，O和S的AUC及最终平均都更高，D更低。S最终Task8/9为100/95，对历史70/90；但最终Task6为45，对历史55，仍有10pp损失。O最终Task6也低10pp，早期Stage8的Task8低5pp；S的Stage8 Task6/Task8低5/15pp。全部负单元在JSON保留。

这是补充描述，不改写预登记判定：相对更强C，O和S没有通过整体门。S最终平均较C高2.5pp，但早期7/6低10pp。

已经逐条验证历史200条初态记录与DOS的文件/逐状态SHA相同；历史记录没有逐轨迹noise SHA，不能声称完整随机数配对。Task6没有新增辅助项，历史RA+ALIGN为80%，当前C/D/O/S却均90%，进一步说明跨运行差值不能全归于模块。未为本补充比较重评历史检查点。

Historical RA+ALIGN is not the auxiliary-expert C arm.
Initial-state file and per-state hashes match, but historical records do not certify matching per-trajectory action noise.
C and DOS Task6 have no new auxiliary loss, yet historical RA+ALIGN scores 80 versus 90; cross-run differences cannot all be assigned to new modules.
DOS use additional training interaction; C/RA+ALIGN are external references, not equal-resource controls.
Post-hoc descriptive baseline comparison does not replace the frozen C comparator or its stopping rule.

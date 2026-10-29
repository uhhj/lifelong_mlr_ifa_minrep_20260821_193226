# 最新完整S OBJECT三seed：独立验收与报告完成

## 用户纠正后的主对照口径

主对照为已验收完整C OBJECT41/42/43（execution/20261027_c_goal42_43_object41_42_43/final_readonly_20261027_222119），不是原S、内部C候选或取消的单阶段机制C。本轮报告此前漏掉已有完整C，现已更正；历史原S仅作版本参考。只复算已有证据，600条C/S初态字节、noise、环境seed和唯一attempt逐条配对通过，无新增训练/模型前向/采集/评估。新S对C均值ΔAUC+1.4931pp、ΔOldFinal+3.8889pp、ΔFinalAll+2.50pp、ΔFWT−0.8333pp。保留全部负cell及配对SF/FS，seed43最终Task6两者均30%。S多120训练采集和269400生产候选optimizer，不能单独归因门控或宣称等算力/等交互。详见FINAL_REPORT.md首节与PRIMARY_C_COMPARISON.json。

实验仍COMPLETE_ACCEPTED，同一监控已删除，不恢复旧实验或开启新监控。本次仅更正分析与轻量发布，验收结论和原始结果不变。

三个seed均400轮/40采集/200正式，合计1200轮、120采集、600正式；2026-10-29 19:57:28 CST自然结束。final_readonly_20261029_201235真实CPU核验与engineering_copy_20261029_v1/audit_final_local.py本地最终验收实际PASS_COMPLETE_LOCAL_ACCEPTANCE；12模型、3恢复、12缓存teacher、初态noise、27300门控及预算验收通过。CPU0环境/0optimizer，没有重复训练评估。

FINAL_REPORT.md与REPORT.md已完成解释，RESULT_SUMMARY和INTERPRETATION_AUDIT保留矩阵、SF/FS、历史原S描述性差值、训练探针与梯度限制。FinalAll均值85%、AUC86.4931%、OldFinal81.1111%；历史原S分别82.5%、85.0347%、82.2222%。旧任务均值未改善，seed43 Task6最终30%且Stage8到9为9 SF/2 FS，不能宣布全面有效或因果优势。

科学合同8594c0b609b8bf948e7914fbf071567ed991d1698a9d8357b90fcb03dec3573d及工程合同befceb11089903813ba65c839e604a44ab1d0c18d7f2dc8282436c581c5ed3cf保持。完成来源仅engineering_copy_20261029_v1/controller.exit=0且无进程；原root controller.exit不存在。不再运行build、guard、validation、dispatch、controller、训练或评估。旧机制永久停止，旧UNKNOWN成本保留。

待收尾仅本轮publish.py轻量发布核remoteSHA/localreceipt，然后删除同一监控s-object-seed。如果已有本地最新PUSH_VERIFIED收据及监控关闭收据，则全部完成，不再远程检查或追加科学工作。禁止发布CP/模型/cache/state/RNG/凭据。已完成的最终CPU不重复运行。

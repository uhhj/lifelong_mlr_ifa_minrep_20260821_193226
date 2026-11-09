# 临时文件用途与清理

2026-11-09；清理完成，实验重跑启动状态另见 RUN_STATUS。未向本地归档任何模型。

已删除服务器 `diagnostics/lambda_recovery_3090_20261102_v2/state_{0,10,20,30,40,50,60}.pth` 七个逐块诊断临时文件。原 validate.py 显示这些文件仅用于 CPU/device 复制路径的完整状态比较；对应验证已完成 PASS，309 次 optimizer started/completed 一致，11 次 autograd，0 环境。后续恢复/验收源未引用这些路径；132 份冻结合同无目标文件依赖且清理前后 SHA 不变。删除前逐项校验实际路径、文件类型、大小和 SHA。

释放 2,424,152,064 bytes（2.26 GiB），清理后服务器 data 可用 8,753,704,960 bytes；随后只读复核为 8,753,700,864 bytes。保留 VALIDATION、LEDGER、验证源码、日志、比较报告与全部哈希；原 interrupted_epoch260.pth 恢复备份 SHA 仍为 65b8d3b4c9f589fb543ba5b7a41991a8c1dbc50daae309c21c386471178b4070。生产模型、缓存和实验结果删除数为0。

READOUT_PATCH 的 patch6/7 文件仍有旧工程引用，未清理。没有递归删除旧实验目录。

此前开始的本地备份下载已停止，295,763,968 bytes 未完成文件已移除，归档目录为空；相关下载入口已禁用。D盘复核仍有约52.9 GiB可用。本地仅保存代码、哈希、报告和轻量证据，不用本地磁盘替服务器保存大体积 state。

证据：capacity_readonly_20261109_152708（用途与依赖）、152759（合同与原备份）、152833（执行回执）、153402（删除后只读核验）、CLEANUP_RECEIPT.json、ARCHIVE_STOPPED.json。

初始10 GiB空间估算与修订6 GiB输出上界/2 GiB保留的推导保留于 PREPARATION_STATUS历史记录和 PROTOCOL；清理不改变科学预算。六个新run仍须按真实启动与自身准入记录判断状态。

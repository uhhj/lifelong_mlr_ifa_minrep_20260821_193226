# IFA关闭：LIBERO-OBJECT三seed Random / Coreset

**COMPLETED**：seed42/43/44，六组2400epoch、633000optimizer steps、24个最终ep99模型、1200条新正式闭环。旧IFA开启1200条只读对照在evidence/on，不能算作新增记录。

- [最终报告、全部12张任务矩阵、三seed均值/样本SD及解释边界](FINAL_REPORT.md)
- [服务器最终验收](FINAL_ACCEPTANCE.json) / [本地原始记录复算](LOCAL_FINAL_RECOMPUTATION.json)
- [完整轻量证据](evidence/off) / [旧对照证据](evidence/on) / [队列与退出证据](evidence/queue)
- [冻结源码与协议](../../experiments/object_coreset_ifaoff_seeds42_43_44/README.md)

AUC均值±样本SD：Random75.486111±13.706660，Coreset81.979167±2.738217。
配对C−R均值+6.493056pp，样本SD12.503616；逐seed为+20.208333、+3.541667、−4.270833pp。
IFA开启时平均优势+6.562500pp，开关交互均值−0.069444pp，SD16.103888。
均值接近不能证明IFA没有影响；仅3个探索性seed，原全局动作RNG未隔离，不能声称显著性或普遍因果。seed44负结果完整保留。

## 离线只读复核

从仓库根目录执行（仅Python标准库，无SSH/GPU/环境执行）：

```sh
python reports/object_coreset_ifaoff_seeds42_43_44/verify_publication.py
```

验证所有归档文本字节、三seed冻结源码、2400epoch/633000step、新旧共2400条记录、120次新vector attempts、全部矩阵、精确指标、均值/SD与交互。不会改变记录或重新评估。

所有证据以可审查文本展开，Git中不含压缩包、模型权重、演示数据、latent特征、buffer载荷或凭据。FEATURES/storage等JSON仅是身份、哈希和选样/覆盖审计元数据。原始绝对路径保留作历史来源；本地报告提到的tar是工作区下载归档，Git版本使用此展开目录代替。COMPARISON.json保留生成时AWAITING状态，最终状态以FINAL_ACCEPTANCE.json为准。

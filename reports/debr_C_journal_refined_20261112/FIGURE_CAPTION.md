**中文图注**

图 X. DEBR 的双经验协同框架。CADL 将当前专家示范与示范记忆 $M_D$ 用于行为克隆、IFA 和执行监督；SHBR 利用成功历史记忆 $M_S$ 及固定行为参考提供支持度加权保持。基础候选 $\Omega_C$ 与保持增强候选 $\Omega_Q$ 从相同完整训练状态和主批次序列出发，各更新 $B$ 步。仅当当前任务的行为克隆损失与执行误差均不高于基础候选、且旧行为保持误差严格更低时，提交 $\Omega_Q$；否则提交 $\Omega_C$，并保留所选分支的完整训练状态。阶段末分别更新两类记忆，供下一任务使用；部署沿用原策略推理结构。图中的专家示范与自主成功执行帧取自不同任务的存档，用于展示两类经验来源。

**English caption**

Fig. X. Dual-experience coordination in DEBR. CADL combines current expert demonstrations and demonstration memory $M_D$ for behavior cloning, IFA, and execution supervision. SHBR uses successful-history memory $M_S$ and fixed behavioral references for support-weighted retention. The base candidate $\Omega_C$ and retention-enhanced candidate $\Omega_Q$ start from the same complete training state and main-batch sequence and each take $B$ updates. The gate commits $\Omega_Q$ only when its current-task behavior-cloning loss and execution error are no greater than those of $\Omega_C$, while its old-behavior retention error is strictly lower; otherwise, it commits $\Omega_C$. The selected branch's full training state is preserved. Both memories are updated at task end for subsequent learning, while deployment retains the original policy inference. Expert-demonstration and autonomous-success frames are archival illustrations from different tasks.

图号由论文最终排版确定。本文件提供独立正文图注，不嵌入SVG或PNG。

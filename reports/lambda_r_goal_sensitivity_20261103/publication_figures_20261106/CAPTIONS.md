# Publication captions

## Figure 1 — Continual closed-loop performance

Closed-loop success rates on LIBERO-OBJECT and LIBERO-GOAL after each incremental training stage. Columns show the mean over all seen tasks, previously learned incremental tasks, and the newly introduced task, respectively. Lines denote the mean and shaded regions denote one sample standard deviation across seeds 41, 42, and 43; each task-stage cell uses the same 20 formal initial configurations per seed. S combines support-weighted retention with candidate gating, whereas Q retains the retention mechanism without gating. C is the matched-optimizer-budget baseline. C and S match physical production optimizer calls; Q and S match retained-update budgets and training-collection counts. The old-task average is undefined at stage 6, and the set of tasks entering each average changes with the training stage. The plotted task IDs are 6–9; pretraining tasks 0–5 are outside this evaluation matrix.

中文说明：展示全部已见任务、旧增量任务、新任务三个维度的闭环成绩。阴影为三个seed的样本标准差，不是置信区间；横轴为真实训练阶段，而非epoch。等预算C与Q/S的预算口径不同，图例中已明确标记。

## Figure 2 — Task-level performance throughout continual learning

Stage-by-task success matrices for C, Q, and S on OBJECT and GOAL. Each cell reports the mean success rate across three seeds, each evaluated on 20 paired initial configurations. Rows correspond to the completed training stage and columns to the evaluated task. Gray cells denote tasks not yet introduced and are excluded from all averages. All panels share a 0–100% color scale. Insets report AUC, FWT, and NBT using the project protocol: AUC averages each task's available post-introduction scores before averaging tasks; FWT is the diagonal mean; NBT is the mean subsequent decrease relative to each task's diagonal score, with lower values indicating less forgetting. C denotes the matched-optimizer-budget baseline.

中文说明：统一色标、保留全部10个有效单元，直接观察方法差异来自哪一个任务和阶段。灰色不是0分；负NBT代表后续相对初学表现的平均改善。

## Figure 3 — Candidate tradeoffs and gate activity

Training-time candidate diagnostics for the default S configuration (lambda_R=0.1), using all 27,300 OBJECT and 20,100 GOAL blocks across three seeds. Each block compares C and Q candidates from the same complete training state and batch sequence; these candidates are distinct from the separately trained C and Q methods in Figures 1–2. Left: execution-error and retention-error changes, expressed as the symmetric relative difference 200(Q−C)/(|Q|+|C|), or zero when both values are zero. Both axes use a symmetric-log scale with a 0.01% linear threshold; all blocks are rendered without sampling or outlier removal. Middle: mutually exclusive patterns of violated gate conditions, as a percentage of all blocks. BC and execution conditions require Q≤C, whereas retention requires Q<C. A candidate can therefore be rejected in the lower-left region if BC worsens. Right: stage-wise Q acceptance, with thin lines/markers for the three seeds and a bold diamond-marked mean. Gate acceptance is a training-probe decision, not a closed-loop success probability.

中文说明：门控图把同状态候选的真实取舍、三个条件各自和共同拒绝哪些候选，以及接受比例随阶段的变化放在一起。中间八类互斥且合计100%；不是三个会重复计数的拒绝比例。散点层以600dpi嵌入PDF，坐标轴、文字和其余图元保持矢量。

## Figure 4 — Retention-weight sensitivity

Sensitivity of S to lambda_R∈{0.05,0.1,0.2} on OBJECT and GOAL. Columns show AUC, the final mean over all four incremental tasks, the final mean over old tasks 6–8, and final new-task (task 9) performance. Colored markers show every seed, with marker shape indicating seed identity; light lines connect the same seed across coefficients, and black diamonds show the three-seed mean. Small fixed horizontal offsets distinguish coincident seed markers; the mean remains at the nominal coefficient. Coefficients are displayed at equal doubling intervals. The pre-existing 0.1 runs are reused once per seed, rather than counted twice as independent baselines. These are fixed-grid sensitivity results; the historical baseline and added runs have the hardware provenance described in the experimental protocol.

中文说明：突出默认权重的整体表现及新旧任务取舍，完整显示种子分歧，不用误差带掩盖GOAL在0.2下的分化。0.1为已有基线复用，新增两档与其存在硬件来源差异。

## Figure S1 — Paired task-level gains and losses

Success-rate differences S−C and S−Q at each observed task-stage cell, averaged across three seeds with paired formal initial configurations. Positive values favor S; negative values favor the comparator. All panels use the same zero-centered symmetric color scale, and all positive, zero, and negative cells are retained. C denotes the matched-optimizer-budget baseline. Gray cells are not evaluated because the task has not yet been introduced.

## Figure S2 — Forward learning and backward transfer sensitivity

FWT and NBT across the same three retention coefficients and seeds as Figure 4. FWT is the mean first-learned task performance; NBT measures subsequent decreases relative to that first-learned score, so lower values are better and negative values indicate average backward improvement. Seed identities, mean markers, and horizontal offsets match Figure 4. The dashed line marks zero NBT. Read NBT together with absolute old/new final success rather than interpreting its sign as an absolute performance guarantee.

## Shared reporting context

All plots use accepted existing experiment evidence. No model updates, additional environment episodes, reruns, feature extraction, seed expansion, or parameter selection were performed to produce these figures. The three-seed summaries describe the existing development protocol; initial configurations repeat across stages and are not treated as independent new scenes. Cross-hardware reuse in the lambda analysis and the existing development-set/pretraining-provenance limits remain documented in the source reports. These context details belong in the experimental protocol; they do not need to be repeated in each plot panel.

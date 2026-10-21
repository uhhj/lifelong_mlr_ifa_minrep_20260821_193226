CAET-v2 的整体结构
RA-Coreset：固定历史经验

原有潜在特征、专家动作、任务配额与选样成员


模块 A：Causal Effect Binding

将原缓存与真实的下一动作及其物理结果对齐，消除目标动作后的信息泄漏


模块 B：Supported Effect Transfer

根据物理效果、操作条件和历史经验支持度，决定哪些跨任务知识能够迁移


模块 C：Functional Retention

以旧任务历史策略的行为表现为参考，约束新任务训练引起的功能退化


原有 MLR＋IFA 策略

仍然只更新 Temporal Transformer＋GMM Head，使用原协议进行闭环推理


这里的核心变化不再是“额外预测一下物体位移”，而是：

只有经过验证的跨任务物理联系才可以促进新任务学习，同时对历史任务的实际动作输出设置独立的保护条件。
模块 A：修复因果效果的时间定义
这一部分可以利用已有缓存，不需要立即重新导出所有潜在特征。

记原演示数组的索引为 \(u\)。

当前缓存最后一个观测是 \(\operatorname{obs}_u\)，对应完成动作 \(a_u\) 之后的状态。它已经不能用于预测 \(a_u\) 的结果，但可以合法地用于预测下一个动作 \(a_{u+1}\) 的效果。

所以新版本改为：
\[
\boxed{
x_u,\ a_{u+1}
\longrightarrow
e_{u+1}
}
\]

其中：
\[
x_u=\operatorname{Encode}
(\operatorname{obs}_{u-7:u})
\]

\[
e_{u+1}
=
\operatorname{Effect}(s_{u+1},s_{u+2})
\]

注意，\(x_u\) 的全部观测都发生在动作 \(a_{u+1}\) 之前；执行该动作后的状态只用于构造标签。

这样有三个好处。

第一，不改变 RA-Coreset 的潜在特征和样本成员。第二，新的辅助预测目标符合当前机器人执行“看到状态，再决定下一动作”的顺序。第三，可以直接利用真实演示验证效果，而不需要恢复历史 MuJoCo 状态。

需要明确，这仍然改变了 CAET 的辅助监督目标：原 BC 损失继续使用原来的专家动作标签 \(a_u\)，但新增的 CAET 使用 \(a_{u+1}\)。两者存在时间监督差异，必须检验其梯度冲突，并通过专门消融排除“仅仅修正动作滞后一拍”带来的收益。

代码修改： 在 caet_exp/labels.py 基础上新建 labels_v2.py，保留原始标签审计记录。通过现有 bindings/task*.json 中唯一的 demo_id 与 target 索引，读取 HDF5 的下一动作，并计算下一转移效果。
# 示意：u 为原缓存最后一帧对应的动作索引# 需要严格排除有歧义、填充和轨迹终点样本if identity_unique and u + 2 < len(states):    x = cached_feature              # [8, 4, 512]    next_action = actions[u + 1]     # [7]    next_effect = physical_effect(states[u + 1], states[u + 2])else:    effect_valid = False



其中 states 与 actions 的对齐关系须按新审计确认的真实 HDF5 契约执行，不能仅依据示意下标猜测。还需要核验视觉 token 和状态 token 的时间来源、时间窗填充和有效边界。

这会比 v1 少一部分有效效果标签，但不会删除任何 BC/RA 训练样本。
模块 B：从“效果相似”改为“有证据的可迁移效果”
v1 的匹配规则本质上是：
\[
\lVert e_i-e_j\rVert\le 0.5
\]

只要不同任务中的六维效果接近，就尝试将其作为正样本。

这个条件过于粗糙，因为相同的物体位移不一定来自相同的控制机制。

新设计使用两类条件共同判断可迁移性。
判断层次	主要信息	目的
物理效果兼容	物体局部位移方向、旋转、朝目标的进展、关节自由度类型	识别结果是否具有可比较的物理意义
执行条件兼容	动作前的夹爪状态、末端与目标物体相对位置、物体是否处于运动阶段	避免把不同操作条件硬对齐
对于 OBJECT 的四个抓取放置任务，我建议优先在篮子/目标的局部坐标系下比较物体运动，而不是直接比较世界坐标下的位移。

例如，两个物体可能从不同位置沿不同世界方向运动，但都在被抬升并移向各自的目标放置区域。这种共同的操作效果比“世界坐标移动方向相同”更值得迁移。

对于 GOAL 中自由物体与旋钮等不同自由度结构，不能强行使用相同的六维位姿差。应显式识别物理自由度类型；只有建立了可验证的任务目标进展对应关系才允许匹配，否则拒绝迁移。
新的效果关联图
令当前任务样本为 \(i\)，历史 RA 记忆为 \(j\)，定义：
\[
W_{ij} =
\mathbf 1[\text{物理类型兼容}]
\cdot
\mathbf 1[\text{执行条件兼容}]
\cdot
\mathbf 1[\text{效果一致且有双向支持}]
\]

这里的双向支持意味着：当前样本不仅能找到旧任务近邻，旧任务近邻也应能够在当前数据分布中找到相容的操作，而不是依赖单个偶然匹配。

进一步要求跨任务关联覆盖多个独立演示和有效操作时刻，不允许大量重叠窗口重复计数。

最关键的改变是：关联图直接决定哪些当前任务样本能够获得跨任务效果迁移监督。

原 transfer.py 主要在效果预测器内部进行对比学习；v2 则把关联关系传递到策略训练。
\[
\mathcal L_{\mathrm{transfer}}
=
\frac{\sum_{i\in D_{\mathrm{current}}}
w_i
\,\ell\!\left(
F_\phi(x_i,\hat a_i),
e_i^*
\right)}
{\sum_i w_i}
\]

其中 \(w_i\) 只对具有可信旧任务支持的当前样本取正值，其余样本不接受跨任务迁移损失。

效果预测器仍从当前任务与合法历史 RA 记忆学习，跨任务配对则为共享效果表征提供额外监督。

与 v1 不同，旧任务样本不再无条件与当前样本接受完全相同的效果辅助梯度。旧任务的保护由模块 C 单独处理。

如果某个阶段没有合格的当前—旧任务关联，或效果预测器在整演示留出集上不能表现出可靠的动作条件预测能力，则该阶段完全退回原 RA＋IFA 训练，不仅关闭对比项，也关闭策略侧效果辅助损失。

这能够针对 GOAL Task 7 正样本为零的已知问题建立明确的退出条件，而不是靠放大匹配阈值强行制造共享知识。

三、最关键的新增创新：旧任务功能保护
仅仅有选择性迁移还不够。CAET-v1 在 seed41 上的退化说明，即使新任务物理效果监督有效，也可能改变旧任务的动作输出。

因此 v2 不应继续单纯叠加一个固定权重的正则项，而应让旧任务功能退化本身成为训练约束。
1. 保护什么？
不是冻结历史网络参数，也不是要求所有旧特征完全不变。

我们要保护的是：
对于 RA-Coreset 保存的历史状态，当前策略对真实下一动作的预测能力，不应无约束地变差。

对于历史记忆样本 \(i\)，定义下一动作负对数似然：
\[
r_i(\theta)
=
-\log p_\theta(a_{i}^{+}\mid x_i)
\]

其中 \(a_i^+\) 是经过模块 A 重新对齐后的专家下一动作。

同时利用经过验证的效果预测器，考察当前策略的动作预计是否还能产生正确的历史物理效果。

这两类信息分别对应：
- 行为功能：还能否预测旧专家应执行的下一动作。
- 效果功能：预计还能否产生旧任务要求的物理变化。
前者必须作为主要保护信号，后者只有在效果预测器可靠的操作区域内才能启用。
2. 不只保护上一阶段，而是保护历史最佳功能状态
设学习当前任务 \(t\) 之前，已有阶段检查点：
\[
\theta_6,\theta_7,\ldots,\theta_{t-1}
\]

对于旧任务 \(j\) 的一个物理效果组 \(c\)，计算这些历史策略在相同合法 RA 样本上的专家下一动作误差。

选择其中误差最低的历史阶段作为该组的功能参考：
\[
s^*_{j,c}
=
\arg\min_{s\in\{j,\ldots,t-1\}}
R_{j,c}^{\mathrm{action}}(\theta_s)
\]

由此得到功能参考边界：
\[
b_{j,c}
=
R_{j,c}^{\mathrm{action}}(\theta_{s^*_{j,c}})
\]

这样做是为了避免一种渐进退化：每个新阶段只比上一个阶段稍差一点，连续学习多个任务后仍可能严重遗忘。

但历史“最佳”指的是同一 RA 记忆上的专家动作似然最好，不代表已证明其闭环成功率最高。
3. 允许有利更新，约束有害退化
定义效果组级保护：
\[
\mathcal L_{\mathrm{retain}}
=
\sum_{j,c}\mu_{j,c}
\left[
R_{j,c}^{\mathrm{action}}(\theta)
-b_{j,c}-\epsilon_{j,c}
\right]_+
\]

其中：
- \(R_{j,c}^{\mathrm{action}}(\theta)\)：当前策略在旧任务效果组上的专家下一动作误差。
- \(b_{j,c}\)：历史参考策略在同一组样本上的误差。
- \(\epsilon_{j,c}\)：预先冻结的容差。
- \(\mu_{j,c}\)：该旧任务效果组的保护强度。
当当前模型没有超过允许退化范围时，不额外施加该组保护；一旦退化超出容差，保护约束才产生额外梯度。

进一步可以加入可靠区域内的预测效果退化约束，但必须用同一个当前阶段冻结的效果预测器评估当前模型和参考模型，避免辅助网络自身的变化被误判为策略遗忘。
4. 自动响应不同旧任务的退化
保护权重不必对所有旧任务固定相同。

建议采用受限的原始—对偶更新：每个完整 epoch 结束后，在当前合法 RA 记忆上计算各组约束违背量，更新 \(\mu_{j,c}\)。

如果某组历史操作持续退化，其保护权重增加；如果没有退化，则保护权重可以回落。

这不是数学意义上的闭环成功率保证。它只是针对历史专家行为误差的可计算约束，仍然可能因为离线与闭环分布不一致而失效。

但比 CAET-v1 的固定 0.05 × effect_loss 更明确地回答了：

为什么新任务训练时，旧任务某些功能应该被保护，而另一些功能可以继续变化？
四、CAET-v2 的最终优化目标
将三个部分结合：
\[
\boxed{
\mathcal L_{\mathrm{CAET-v2}}
=
\underbrace{\mathcal L_{\mathrm{BC}}+
0.1\mathcal L_{\mathrm{IFA}}}_{\text{原论文训练目标}}
+
\underbrace{\lambda_T\mathcal L_{\mathrm{transfer}}}_{\text{选择性正迁移}}
+
\underbrace{\mathcal L_{\mathrm{retain}}}_{\text{历史功能保护}}
}
\]

与原版最大的区别：
	CAET-v1	CAET-v2
效果预测	存在动作后信息泄漏	使用合法的下一动作因果配对
跨任务关系	六维效果距离	物理及操作条件兼容的支持图
迁移启用	有效果标签就参与策略辅助训练	仅可信且有旧任务支持的当前样本参与
历史经验	对旧样本也施加统一效果损失	独立进行历史行为功能保护
遗忘控制	没有显式退化边界	历史最佳功能参考＋效果组约束
负迁移	可能无条件发生	无支持时回退RA＋IFA，并检查梯度冲突
还建议加入一个塑性保护规则：当跨任务效果损失的梯度与当前任务 BC 梯度明显冲突时，不允许该辅助梯度直接抵消当前任务的专家行为学习。

在实现中可先采取确定性的冲突门控：
\[
g_T^{\mathrm{safe}}=
\begin{cases}
g_T,&\langle g_T,g_{\mathrm{BC,current}}\rangle\ge0\\
0,&\text{否则}
\end{cases}
\]

这里仅控制额外迁移梯度，不修改原有 BC 和 IFA 梯度。需要单独审计它的真实触发比例与计算成本；梯度非冲突也不保证闭环成功率改善。

五、对照现有代码，具体应该如何改造？
建议不要覆盖原 CAET-v1，而是在其已审计源码上建立独立版本 caet_v2_exp。
现有文件	重设计内容
caet_exp/labels.py	新建因果下一动作标签版本，保持HDF5原始状态、动作、缓存哈希严格绑定
caet_exp/dataset.py	为原样本附加 next_action、next_effect、效果组、迁移支持权重
caet_exp/model.py	保留轻量效果预测器结构，使用新的因果标签重新训练
caet_exp/transfer.py	替换简单 pair_masks，建立当前—旧任务支持图，去除重复时间窗造成的虚高计数
caet_exp/fit.py	在当前任务和合法RA记忆上拟合效果模型，加入按真实匹配样本构造的对比训练
caet_exp/observe.py	将原统一效果损失改为选择性迁移损失＋旧任务功能保护
caet_exp/train.py	增加阶段支持准入、历史参考评分、保护权重更新和完整恢复
caet_v2_exp/functional_reference.py	新增：计算每个旧任务效果组的历史最佳功能参考
caet_v2_exp/admission.py	新增：因果时序、有效配对、动作敏感性与约束有效性验收
原 ra_exp/storage.py、ra_exp/selector_reference.py、coreset_exp/eval_metric.py 原则上不改。

代码参考：原 CAET 实现目录；上述新增文件是设计要求，并非现有仓库已经包含的代码。
实际训练时序
以 OBJECT seed42 为例：
阶段	CAET-v2 应执行的行为
Task 6	原RA＋IFA训练100epoch；模型与既有RA Task6逐参数一致；保存选中样本对应的因果效果标签
Task 7	根据Task6记忆与Task7数据构建支持图；验证效果模型；有可信支持才启用选择性迁移和Task6功能保护
Task 8	在Task6、7记忆与Task8当前数据之间建立关联；分别保护旧任务中的相关效果组
Task 9	同样训练；历史功能参考不仅来自Task8结束时，还可来自各旧任务过去表现更好的阶段
注意，历史参考只能使用当前阶段之前已经完成的模型和合法记忆，不能访问未来任务，也不能把历史已丢弃的完整示范重新引入训练。
训练入口示意
下面是算法层面的执行结构，而非可直接运行的完整代码：
for task in [6, 7, 8, 9]:    native_data = build_original_ra_dataset(task)    if task == 6:        train_original_ra_ifa(native_data)    else:        causal_data = attach_causal_effect_labels(native_data)        support = build_effect_compatibility_graph(            current_task=task,            old_buffer=ra_buffer,            causal_data=causal_data,        )        if not sufficient_transfer_support(support):            # 整阶段严格退回原RA+IFA            train_original_ra_ifa(native_data)        else:            effect_model = fit_and_validate_effect_model(                causal_data, support            )            if not reliable_action_conditioned_prediction(effect_model):                train_original_ra_ifa(native_data)            else:                references = build_historical_function_envelope(                    previous_checkpoints,                    old_buffer=ra_buffer,                    effect_model=effect_model,                )                train_ra_with_safe_transfer_and_retention(                    native_data,                    effect_model,                    support,                    references,                )    save_epoch99_checkpoint()    verify_and_store_exact_ra_members()    retain_effect_sidecar_for_selected_members()



其中 build_original_ra_dataset() 保持原来的样本拼接、排序、DistributedSampler及梯度更新预算。

CAET辅助网络的随机数流必须独立于原策略训练，不能因为构建支持图或效果预测器而扰动原有随机数据顺序。
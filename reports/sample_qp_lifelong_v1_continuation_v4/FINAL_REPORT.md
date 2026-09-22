# batch258 等价第二求解器验证

正式训练未恢复。QP、normalization、lambda映射、1e-8门限及AdamW语义均未修改。

## 数值结论

原v3 L-BFGS-B在1次迭代后以目标相对变化停止，KKT=1.547045713756745e-8。指定的ftol=gtol=0候选运行39次仍失败，最小callback KKT=1.1223141533529099e-8，final=2.694640956587779e-8；该失败候选未纳入v4。

trust-constr按固定配置从同一30k标准解出发，第40次首次通过：float64=3.3966850332638125e-9，80位十进制=3.3966852472411387e-9。最终第50次以gtol停止，final KKT=1.944270892788285e-11；按照预定规则接受第40次，而不是事后挑更优最终解。

## 门禁

16项单元测试通过；F1/F2/B185/B258均通过两种精度验收。F1不调用fallback，F2/B185沿用原v3第一fallback，仅B258调用trust-constr。

B258真实单步的physical residual=0，actual_positive=0，optimizer与scheduler均3657→3658；模型/校正/位移均finite。B185结果与历史v3完整状态精确相同，physical residual=3.7383576701034376e-10。两项post-step状态保存后restore全部一致。每个hard batch只执行一次native更新，没有前序batch重放。

32/64/96普通路径在非零LR、非零位移上均与v3的模型、moments、scheduler、RNG逐项完全相同，两套fallback均未调用。保留warmup首步零位移检查，但以补充的非零位移结果作为充分证据。

正式v4来自数值代码逐字节相同的候选包，step仅改import路径；已验证函数identity、train/step/core的显式捕获参数传递，并在真实v4 solver中强制失败验证数组/轨迹/上下文先落盘再raise。

## 资产审计更正

旧qp_solver_recovery_v2/F2_before_extended.npz已被历史流程覆写为B185内容，首次使用它的回归作废并保留在invalid_F2_source_attempt。最终F2取v3冻结fixtures/F2_dual_qp.npz，并核验其原始残差1.0038755857838533e-8，不能将两个B185误当四个独立实例。

## 当前边界

V4_PREFLIGHT_PASS=true。未启动train或评估，也未生成正式RECOVERY_INDEX。保存的post-step文件是可直接恢复的诊断状态，不冒充完整epoch恢复点。后续正式启动仍需使用既有parent-child恢复机制迁移合法边界。

正式v3源码和旧资产保持不变；未做全仓hash。四个hard实例的通过不保证所有未来QP都会成功，最终失败仍保留完整现场并fail-stop。

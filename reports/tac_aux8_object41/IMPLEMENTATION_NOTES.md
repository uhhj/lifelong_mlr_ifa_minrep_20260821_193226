# 实现与范围

新实验名CAET-v2+TAC_AUX8；不是原七步TAC恢复。旧worker、source、合同、297epoch中断产物均保留。

`aux_head.py`持有独立头/AdamW；`observe.py`在效果与保护路径只调用aux_head；`functional_reference.py`绑定历史主policy及辅助头；`train.py`完整epoch保存主+辅助状态、ep99分开导出。`science.py/variant.py/model.py`统一八步和下一动作物理标签。原coreset_exp及libero主策略/BC/evaluate源码不改。

辅助教师用detach共享表示，效果与保护反传共享Temporal但不直接进正式GMM。全局clip可能改变主梯度缩放，因此“GMM无辅助直接梯度”不等于其参数轨迹永远等于RA；Task6/零迁移阶段须保持原生精确一致。H_aux首初始化深拷贝不消耗随机数，后来独立加载先前头状态。

原RSET与EFTR仍用旧worker源和入口，新dispatcher只替换首个A入口，双队列锁防重复。任何独立工程失败都保留原退出证据，只在新版本修复后重启；不静默重贴冻结源码hash。最终本地验收脚本需按辅助头新元数据独立适配，不套用旧7步检查。

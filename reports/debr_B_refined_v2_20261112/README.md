# DEBR B版：双经验协同精修

用户已选定B版。本次保留对称双翼构图，强化“专家经验引导动作学习—自主成功经验保持已获得行为—协同选择策略更新”的视觉主线。

主要调整：两翼采用同一网格与字体层级；角色标题改为Guide action learning / Retain self-acquired behavior；蓝青路径通过双色圆环汇入Dual-experience coordination；三项训练探针分别标示当前任务模仿、执行与成功行为保持；部署侧突出One evolving policy。双色圆环表示协同，不表示候选平均或始终叠加保持。各项科学关系与原版相同，原图稿及Word未改动。

文件：
- DEBR_B_refined.svg：1800×940原生可编辑矢量母版。
- DEBR_B_refined.png：4800×2507高清PNG，直接从上述SVG导出。
- DEBR_B_preview.png：原尺寸预览。
- DEBR_B_SVG_PNG_comparison.png：左SVG渲染、右高清PNG缩览的并排对照。

PNG与SVG使用唯一母版，几何、文案和颜色相同，区别仅为分辨率与显示时的栅格化。图内统一Times New Roman，所有文字、线条、框体和图标均为原生SVG，无嵌入位图。其他电脑编辑需安装该字体。

实际核对：两条经验路径及阶段末记忆更新；相同完整状态和主批次；B=10双候选；BC_Q≤BC_C、Exec_Q≤Exec_C、Ret_Q<Ret_C；全过选Q否则C；提交所选完整状态；首任务仅基础更新；部署推理结构不变。没有新增实验、训练或评估。

重建：python build.py，然后用Node执行render.cjs。RENDER_RECEIPT记录实际SVG和PNG哈希，REVIEW记录结构及视觉复核，发布使用白名单，不包含实验数据或旧图。

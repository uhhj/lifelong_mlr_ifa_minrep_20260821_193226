# 实现说明

直接导入已冻结ALIGN原生build、masked BC、IFA、optimizer/scheduler，使用原Stage9 manifest而非扩大的buffer。仅B/C/D增加训练时eval模式旧Task8前向；保存并恢复modes/RNG，不改部署结构。训练恢复态以完整epoch为提交边界，当前实现不自动重启。准入序列先全部四组通过再开始A→B→C→D主训练；随后一次有限160episode评估。新科学参数在PROTOCOL明确登记，未根据结果选择。

准入实际结果尚无；最终4CP/完整账目/配对初态噪声/160回合独立验收也尚无。旧任务均不恢复。

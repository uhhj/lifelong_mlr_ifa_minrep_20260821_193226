# S GOAL42/43 实施与验收边界

已接受的CCBP-v4 engineering02是方法来源。losses.py、legacy_losses.py、memory.py、updates.py、synthetic_checks.py保持原文本；UPLOAD_REVIEW绑定原SHA和本次SHA。review目录的文本差异逐项审查：唯一科学运行变量为seed及其原有RA/ALIGN输入，S单组无需D比较。删去D相等断言不削弱原官方初始化/原ALIGN更新/全部状态恢复准入；历史CUDA RNG与Task6仅记录硬件差异。

预检实际校验两seed父合同/源/完整输入文件SHA/RA身份mask及初态分割，没有新forward、optimizer或环境。实际卡3090只记录，不以旧4080S张量差异拒绝。全部新数据在/data；4GiB全队列峰值按唯一inode抵扣，不降低1/2GiB保留。

单controller有限派发：train42→train43→evaluate42→evaluate43。每seed独立完整100×4epoch，绝不共享seed41最终模型或两个新seed的教师。部署前实际Python3.8导入与合成兼容检查通过后才派发；自己的GPU准入必须逐阶段实际通过。没有只跑Task9分叉。

check_readonly核顶层33冻结项（包括两子合同）与各父源，audit_progress按独立seed账本复算每轮人口/步数/锚槽/准入/缓存/初态记录和noise。有效进程与exit另核，单独账目PASS不当完成。容量脚本只stat，动态元数据快照不一致应保留原记录后精确说明，不能误判为科学失败。

最终remote CPU wrapper与audit_final_local/describe_final尚未真实调用；静态解析不算最终PASS。需要controller.exit0和无自身进程后才实际8个100键CP/2完整恢复/8源教师缓存与抽样/原RA/全80训练400正式及真实初态字节noise重算。外置wrapper错误留证修复，不修改冻结实验，不重训重评。最终报告须解释负单元、离线代理与闭环一致或不一致及额外交互；没有C/O42/43，不能伪造筛选因果。

凭据在用户授权Windows Credential Manager和本机内存worker中，.session_control与DEPLOY_TEXT_BUNDLE不发布。每轮本目录publish.py精确审核仅代码/报告/hash/scalars/outcomes，非force推MLR并留本地receipt。

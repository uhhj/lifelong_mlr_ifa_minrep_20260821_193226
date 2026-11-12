# 本轮符号规范与现有Word的差异

本轮图稿严格采用用户最新消息指定的记号：$M_D,M_S$；$\Omega_C,\Omega_Q$；$\pi_{\theta_k}$；以及不带帽号的三个完整探针函数 $L_{\mathrm{BC,new}}(\theta)$、$E_{\mathrm{new}}(\theta)$、$R_{\mathrm{old}}(\theta)$。没有擅自修改Word正文。

只读核对现有文档 `C:/Users/Lenovo/Desktop/论文/论文写作/DEBR_Complete_Manuscript_Method_ABC_20261110_消融表更新 - 副本.docx` 时发现：该文件中的记忆写法为花体M的上标D/S，而不是本轮指定的下标D/S；式(20)六个L/E/R基字母使用了OMML重音结构。若继续采用这一旧稿，需在正文端统一记忆上下标和探针帽号，才能达到图文完全一致；当前图稿不能被描述为已自动完成Word同步。

只读证据：DOCX SHA256 `b6c7809bbc6d47b17c8fdacaf44814a193c10e8c1d22d3ccf06e602af657235e`。内部`word/document.xml`，按直属`w:body/w:p`零起始计数（含空段落、排除表格），p[18]的oMath[3]/[4]为`m:sSup`且基字母M带`m:scr=script`；p[69]的式(20)中有六个`m:acc`，依次作用于L/L/E/E/R/R。本轮用户消息明确提供了目标公式，因此图稿以该消息为准。

## 2026-11-01 18:24 用户追问突然变慢：只读资源诊断

capacity_readonly_20261101_182401及LOCAL_SLOWDOWN_AUDIT实际复算通过，31.25秒0更新/0环境/0写入/0信号。客户机可见CPU配额不限、限流0，主线程排队仅0.0019秒；无swap/major fault增加。训练写入346.55MB、主机IO full pressure增加6.34秒，GPU利用0–52%、34–38°C；有IO等待和间歇GPU空闲，但未定位函数级原因，不可称限流或确认共享资源争用。Task7固定每轮227接受/454物理/23块，前五组十轮均76.51/154.26/213.19/89.74/177.16秒，第51–54轮又均71.91秒。轮内seconds不含其后checkpoint保存及epoch末probe，IO不能单独解释全部变慢。详见SLOWDOWN_REPORT_20261101.md。

100–145小时只是慢速持续情景，不是稳定预计完成日期；四轮回快不足以宣布恢复。没有修改生产运行，最新完整进度审计仍18:19，下方预算/合同/720分钟整run监控保持；以后按实际吞吐更新，不从诊断历史冒称完整进度审计PASS。

## 2026-11-01 18:19 用户要求进度与估时

实际check_readonly_20261101_181931审计PASS，162910全部epoch/门控/诊断/正式前缀保留。L005_41 Task6完整100轮，Task7完整50轮，共150/400轮27050接受38400生产物理，33隔离19准入autograd、10采集，正式0；其余五run未开始。135冻结项及controller2163/GPU2169正常，无退出失败、重训或新基线调用。全部六run150/2400轮、正式0/1200。

Task7前50轮均142.1730秒、近10轮177.1605秒，明显慢于上次8轮均77.01；余50轮纯计算118.48–147.63分钟。Task8/9未实测，按batch及已接受优化S43阶段比例分别外推，当前run余纯计算13.65–19.44小时；计准备保存及剩余30采集，规划15–21小时。若当前速度一直持续，五个未启动run沿用同资源速度情景，全部剩余训练采集加1200正式粗略100–145小时（约11月5日晚至7日晚），不是保证；正式暂留2–4小时历史余量，CPU验收解释发布另计。速度恢复可大幅缩短，不能按时钟断言完成。TIMING_AUDIT保存全部公式和假设；下一720分钟在当前整run结束较早侧复查，覆盖480分钟，不固定12小时。

capacity_readonly_20261101_182043仅3次短资源采样，实际0更新/0环境/0写入：训练进程累计CPU约96.4%单核，GPU利用率2/44/3%，温度33–34度，未报告功率或热降频，compute-app仅自身2169。提示CPU侧/同步开销可能相关，但不足以定位瓶颈或证明持续资源竞争；未改参数/进程/环境，LOCAL_RESOURCE_AUDIT保留边界。

# OBJECT 最新S固定λ_R六run：当前交接

实际观察 2026-11-01T18:19:33.291586+08:00；check_readonly_20261101_181931/LOCAL_AUDIT=PASS_PROGRESS_NOT_FINAL。

唯一root `/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1`，controller2163，合同`6fde2ecbc3953f0af0709e70bbc3aaed3df9be57bf9e6bf59476985caafb135e`，135冻结项。唯一部署deploy_remote_20261101_152959。禁止重复compose/freeze/deploy/controller或已完成epoch。

|case|完整轮/400|接受更新|生产物理|采集/40|正式/200|
|---|---:|---:|---:|---:|---:|
|L005_41|150|27050|38400|10|0|
|L005_42|0|0|0|0|0|
|L005_43|0|0|0|0|0|
|L020_41|0|0|0|0|0|
|L020_42|0|0|0|0|0|
|L020_43|0|0|0|0|0|

累计150/2400轮、27050/633000接受、38400/1171800生产、33/546隔离、19/246准入autograd、10/240采集、0/1200正式。方向诊断最多216autograd另记。

下一720分钟复查，首次核自己的保持阶段准入及吞吐，此后依据整seed或整组正式实际剩余时间调整，不固定高频，不按时钟推断完成。

先完整读PROTOCOL.md、IMPLEMENTATION_NOTES.md、RUN_STATUS.json、ACTIVE_ROOT.json、FINAL_WRAPPER_REVIEW.json及最新EVIDENCE/LOCAL_AUDIT。用户只授权OBJECT六run：0.05/0.2各41/42/43，已有0.1三个完整S只读复用。GOAL不跑，C/Q/旧机制不恢复。每阶段100轮，最新S双候选10步门控、零epsilon、自己成功缓存q和原4槽分母不变；只改变保持合成项λ_R，准入同λ均匀支持参照不是原0.1轨迹。

顺序L005三seed训练→600正式→L020三seed训练→600正式。240采集20–29不补采，1200正式0–19原600步5zero配对Normal。每stage均须自己的GPU准入/COPY_EQ、官方初态及Task6对已有S全tensor实际通过；未来阶段不能套已有PASS。

运输使用旧机制目录现有.session_control worker，先读status并核PID，不重启活跃worker或输出凭据。本目录check复制旧目录queue，按result真实目录复制回来，audit_progress.py NEW PREVIOUS→record_progress.py NEW INTERVAL。任何错误先保留原失败、hash、进程/退出、完整恢复和已耗成本，禁止盲恢复部分块或环境attempt，不改科学协议。

只有自然本root/controller.exit0且无自身进程后，才final_readonly实际CPU24CP/6恢复/24teacher缓存/1200初态noise/54600门控及预算验收→audit_final_local→compare_primary_s。FINAL_WRAPPER_REVIEW目前仅STATIC_ONLY。包装错误保留原证据后外置版本化修复，不重训重评。

随后自主完成FINAL_REPORT/ACCEPTANCE：三档全结果、新旧任务、SF/FS、全部负单元、门控接受与梯度/训练代理、成本边界。不按正式成绩选择最佳λ，不扩seed或负结果加实验。已有0.1的600回合被两次配对而非1200独立基线回合。保留开发集反复使用和预训练来源未闭合，敏感性分析不是独立确认或全局无泄露。

每次检查按根AGENTS经publish.py轻量审查code/report/hash/scalars/outcomes，串行nonforce推MLR核remoteSHA/localreceipt，禁CP/model/state/RNG/cache/凭据/未审bundle。健康进展静默，只完成、新失败或需行动通知。六run全部独立验收、解释及发布后才删除本新监控，不恢复旧监控。

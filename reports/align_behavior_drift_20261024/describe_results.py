import datetime,hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]).resolve()
def read(p):return json.loads(p.read_text(encoding='utf8'))
s=read(D/'SUMMARY.json');audit=read(D/'LOCAL_AUDIT.json');cost=read(D/'complete.json')
assert audit['status']=='PASS_LOCAL_SCALAR_DRIFT_AUDIT_NOT_CAUSAL_CLOSED_LOOP_PROOF'
lines=['# ALIGN GOAL Task8→Task9 行为漂移：实际验证结果','',
'已执行用户附件第六节的第一步：不训练、不跑环境，使用三个已完成GOAL seed的stage8/stage9检查点，对自身相同Task8 RA窗口做配对CPU前向。6CP严格100键、1356窗口、1283合法下一动作标签通过本地独立核验；2712模型窗口、174次batch前向。原样本人口不因标签无效而删减，无效标签仅不计专家误差。','',
'**结论：存在真实部署分量行为漂移，Wasserstein距离可稳定刻画它；但当前证据没有验证“模式切换/夹爪翻转导致遗忘”。在可核验的夹爪翻转窗口，学生反而更接近专家。不能据此直接强制锁住教师分量或把低margin当作有害风险。**','',
'## 同一Task8窗口与既有闭环','',
'|seed|最高分量切换（452窗口）|夹爪严格异号翻转|原Task8闭环stage8→9|原Task9最终|',
'|---|---:|---:|---:|---:|']
for seed,x in s.items():
 a=x['all'];c=x['closed_loop_existing_percent']
 lines.append(f"|{seed}|{a['mode_switch_count']}/452（{a['mode_switch_count']/452*100:.2f}%）|{a['sign_flip_count']}/452|{c['stage8_task8']:g}%→{c['stage9_task8']:g}%|{c['stage9_task9']:g}%|")
lines += ['', '闭环只引用已完整验收的原20回合/cell结果，无新增rollout。seed43切换最多（76），Task8下降却小于另两个seed；三个点不足以统计推断，但已不能把切换数量直接当作性能下降大小。','',
'## 五项诊断的数值','',
'平移/旋转均值漂移是各三维动作子向量的L2距离；单位为原动作控制坐标，**不是mm或角度**，未校准成物理位移。W2²是被选两个对角高斯的七维均值差平方和加标准差差平方和，另分别保存平移/旋转/夹爪分解，不能直接当训练预算epsilon。','',
'|seed|平移动作均值漂移mean / P90|旋转动作均值漂移mean / P90|所选W2² mean / median / P90|标准差变化占W2²比例|','|---|---:|---:|---:|---:|']
for seed,x in s.items():
 n=x['numeric'];t=n['translation_mean_l2_delta'];rot=n['rotation_mean_l2_delta'];w=n['selected_w2_squared']
 lines.append(f"|{seed}|{t['mean']:.6f} / {t['p90']:.6f}|{rot['mean']:.6f} / {rot['p90']:.6f}|{w['mean']:.6f} / {w['median']:.6f} / {w['p90']:.6f}|{100*x['variance_fraction_of_total_w2']:.3f}%|")
lines += ['', 'W2²均值被少数大均值跳变拉高，典型窗口变化远小于均值。夹爪翻转的动作均值差接近2，对应W2²约4；W2不会像除以极小方差的KL那样爆炸。变化主要来自均值，不是标准差噪声增加。但“距离稳定”不等于“距离越大越有害”。','',
'|seed|专家mixture NLL：教师→学生|所选均值分组Huber：教师→学生|合法窗口误差变差数|','|---|---:|---:|---:|']
for seed,x in s.items():
 n=x['expert']['mixture_nll'];h=x['expert']['grouped_huber'];a=x['all']
 lines.append(f"|{seed}|{n['teacher']['mean']:.6f}→{n['student']['mean']:.6f}|{h['teacher']['mean']:.8f}→{h['student']['mean']:.8f}|{a['expert_huber_worsened']}/{a['valid_expert']}|")
lines += ['', '三个seed的旧回放mixture NLL与所选均值Huber平均都改善，闭环却下降。仍有168/162/176个合法窗口误差上升，因此不说所有窗口均改善；总体平均也不能定位关键失败时刻。连续密度的NLL可以为负，其改善不代表闭环成功。','',
'所选Normal NLL原值完整保留：近1e−4夹爪sigma会使少量错误均值导致百万量级平均NLL，未裁剪或修改方差。mixture NLL和选中Normal NLL不是同一指标，不得把后者当训练loss爆炸。','',
'## 最关键反例：翻转可能纠正教师错误','',
'三个seed分别3/5/4个夹爪翻转，共12个。其中9个有合法ALIGN专家下一动作（每seed3个），这9个的所选动作分组Huber全部下降：教师约0.50，学生约5e−6至4.5e−5；其余3个没有合法标签，不评价翻转好坏。它们全是最高分量切换。','',
'因此，直接要求学生维持教师夹爪符号或教师argmax，至少会反对这些已观察到的专家拟合改善。专家演示动作不是唯一可行动作，离线改善也不能证明闭环因果收益；但它足以否定把每次翻转自动标为有害的做法。','',
'## margin能提示切换，尚不能提示损害','',
'按预先固定的教师概率margin四分位描述：seed41/42全部切换位于最低四分位；seed43四组切换数为42/24/10/0。低margin与切换倾向相关，符合argmax边界现象。但最低margin组的平均专家Huber也都改善，其中包含上述有益拟合修正。故低margin是切换敏感性指标，尚不是经验证的损害风险。窗口高度重叠，不做窗口级p值或独立样本显著性宣称。','',
'## 固定分量编号与行为距离必须分开','',
'|seed|实际所选W2²均值|固定教师编号W2²均值|学生最接近教师行为分量W2²均值|','|---|---:|---:|---:|']
for seed,x in s.items():
 n=x['numeric'];lines.append(f"|{seed}|{n['selected_w2_squared']['mean']:.6f}|{n['same_teacher_index_w2_squared']['mean']:.6f}|{n['nearest_student_component_w2_squared']['mean']:.6f}|")
lines += ['', 'seed43固定编号距离明显高于最近行为分量距离，提示固定编号会额外限制模式重排/重新分工。最近分量仅做每窗口到教师所选分量的单向最小距离，不是一一匹配，也不是可直接部署的选择规则。不能据此把全部76次切换称为纯编号交换。','',
'## 对附件下一步的判断','',
'本次验证支持：已有CP可进行原生行为诊断；选中动作确有漂移；W2避免了小方差KL尺度爆炸。它未支持：漂移/翻转可直接标记遗忘原因，或按低margin一律加严保护可改善闭环。','',
'下一阶段应先登记能区分教师正确性与学生纠错的保护规则，并独立校准动作分组尺度与epsilon；纯锁定教师分量应当作为可能有副作用的对照，而非默认正确约束。若继续固定/自适应信赖域训练，必须另固定风险函数、margin/预算、相同训练更新数、比较组和新旧任务闭环目标；不得根据本次三个seed反复选阈值后宣称独立验证。附件尚未量化这些项，且要求先做机制验证，本轮未启动任何信赖域训练或消融。','',
'现有检查仅覆盖保留的Task8演示窗口，不含闭环失败访问状态或关键接触时刻；Task8成功率下降也可能涉及这些未见状态、误差积累或原global动作RNG序列差异。本次CPU原类/原权重验证不等于CPU/GPU逐bit动作一致性，接近logit并列的模式也应保留数值敏感性限制。','',
'## 执行与工程证据','',
f"成功运行CPU计算{cost['cpu_seconds']:.6f}秒、墙钟{cost['wall_seconds']:.6f}秒，单线程；无优化器、反向、环境或CUDA、无服务器产物写入。源/CP/侧车/全部buffer输入前后SHA与模型tensor不变。两模型共享相同窗口与合法标签，6个CP实际strict100键加载。",
'', '首版外置脚本因服务器Python3.8无str.removeprefix在第一前向前失败，新增0前向。独立v2仅兼容前缀处理，失败源与记录保留；另一次本地生成命令引号错误在连接前取消，0远程操作。本地审计首次误用服务器状态名检查已接受本地包，失败源保留，改用真实COMPLETED_LOCAL_INDEPENDENT_PASS后通过，无重复模型计算。',
'', '完整原始标量、统一统计、逐demo和margin四分位分层及SHA绑定见运行目录SUMMARY.json/LOCAL_AUDIT.json。既有ALIGN闭环和所有旧实验结论不改。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
binding={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'REPORT.md',R/'audit_local.py',R/'describe_results.py',D/'SUMMARY.json',D/'LOCAL_AUDIT.json',D/'complete.json')}
(R/'REPORT_BINDING.json').write_text(json.dumps({'status':'PASS_REPORT_BOUND_TO_LOCAL_AUDIT','files':binding},indent=2)+'\n',encoding='utf8')
(R/'RUN_STATUS.json').write_text(json.dumps({'status':'COMPLETE_READONLY_DIAGNOSTIC_ACCEPTED','observed':datetime.datetime.now().astimezone().isoformat(),'result_directory':str(D),'seeds':[41,42,43],'checkpoints':6,'paired_windows':1356,'valid_expert_windows':1283,'model_windows':2712,'forward_calls':174,'optimizer_updates':0,'environment_steps':0,'trust_region_training_started':False,'causal_forgetting_mechanism_proven':False,'next_scheduled_check':None},indent=2)+'\n',encoding='utf8')
(R/'AUDIT_WRAPPER_RESOLUTION.json').write_text(json.dumps({'status':'PASS_AFTER_LOCAL_SCHEMA_FIX','failed_source':'AUDIT_WRAPPER_ATTEMPT_01.py','fix':'Require actual accepted local status COMPLETED_LOCAL_INDEPENDENT_PASS, retaining source hash and data validation','local_audit_sha256':binding[str((D/'LOCAL_AUDIT.json').relative_to(R))],'remote_reruns':0,'extra_forwards':0},indent=2)+'\n',encoding='utf8')
print(R/'REPORT.md')

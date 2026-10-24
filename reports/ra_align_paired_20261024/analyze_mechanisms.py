"""Descriptive, trajectory-level mechanism audit; no threshold fitting or new inference."""
import json,sys,hashlib,datetime,statistics
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]).resolve()
def js(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert js(D/'LOCAL_FINAL_AUDIT.json')['status']=='PASS_LOCAL_PAIRED_DIAGNOSTIC_AUDIT'
pairs=js(D/'PAIRED_TRAJECTORIES.json');outcomes=js(D/'OUTCOMES.json');outmap={(r['seed'],r['method'],r['driver'],r['initial_index']):r for r in outcomes};summaries=[];evidence=[]
for p in pairs:
 for driver in [8,9]:
  key=(p['seed'],p['method'],driver,p['initial_index']);o=outmap[key];path=D/'light'/o['run']/('trajectory_%02d.jsonl'%p['initial_index']);rows=[json.loads(s) for s in path.read_text().splitlines()]
  norms=lambda k,sl:[float(np.sqrt(sum(x[k][sl]))) for x in rows]
  mu=np.asarray([sum(x['mean_delta_squared']) for x in rows]);sd=np.asarray([sum(x['sigma_delta_squared']) for x in rows]);mode=[x['mode_switch'] for x in rows];flip=[x['gripper_flip'] for x in rows]
  first=lambda vals:next((i for i,x in enumerate(vals) if x),None)
  v={'seed':p['seed'],'method':p['method'],'initial_index':p['initial_index'],'driver':driver,'transition':p['transition'],'success':o['success'],'steps':len(rows),'same_initial_observation':p['same_initial_observation'],'first_observation_divergence_step':p['first_observation_divergence_step'],'first_action_difference_on_shared_history':p['first_action_difference_on_shared_history'],'first_mean_difference_gt_1e3':o['first_events'].get('mean_delta'),'first_mode_difference':first(mode),'first_gripper_flip':first(flip),'has_mode_difference':any(mode),'has_gripper_flip':any(flip),'mode_difference_fraction':float(np.mean(mode)),'gripper_flip_fraction':float(np.mean(flip)),'terminal8_has_flip':any(flip[-8:]),'mean_translation_action_l2':float(np.mean(norms('action_delta_squared',slice(0,3)))),'mean_rotation_action_l2':float(np.mean(norms('action_delta_squared',slice(3,6)))),'mean_gripper_action_abs':float(np.mean(norms('action_delta_squared',slice(6,7)))),'mean_selected_w2_squared':float((mu+sd).mean()),'variance_fraction_of_w2':float(sd.sum()/(mu.sum()+sd.sum())) if (mu+sd).sum() else 0,'temporal_mode_changes_stage8':sum(rows[t]['teacher_mode']!=rows[t-1]['teacher_mode'] for t in range(1,len(rows))),'temporal_mode_changes_stage9':sum(rows[t]['student_mode']!=rows[t-1]['student_mode'] for t in range(1,len(rows))),'rows_sha256':sha(path)}
  v['first_mean_difference_gt_1e_minus3']=v.pop('first_mean_difference_gt_1e3')
  prefix=rows[:32]
  v['prefix32_steps']=len(prefix);v['prefix32_mode_difference']=any(x['mode_switch'] for x in prefix);v['prefix32_gripper_flip']=any(x['gripper_flip'] for x in prefix)
  v['prefix32_translation_action_l2']=float(np.mean([np.sqrt(sum(x['action_delta_squared'][:3])) for x in prefix]))
  evidence.append(v)
def aggregate(items):
 if not items:return {'n':0}
 return {'n':len(items),'mode_difference_episodes':sum(x['has_mode_difference'] for x in items),'gripper_flip_episodes':sum(x['has_gripper_flip'] for x in items),'terminal8_flip_episodes':sum(x['terminal8_has_flip'] for x in items),'median_mode_difference_fraction':float(np.median([x['mode_difference_fraction'] for x in items])),'median_gripper_flip_fraction':float(np.median([x['gripper_flip_fraction'] for x in items])),'median_translation_action_l2':float(np.median([x['mean_translation_action_l2'] for x in items])),'median_w2':float(np.median([x['mean_selected_w2_squared'] for x in items])),'median_variance_fraction':float(np.median([x['variance_fraction_of_w2'] for x in items])),'first_flip_steps':[{k:x[k] for k in ['seed','initial_index','first_gripper_flip','steps']} for x in items if x['has_gripper_flip']]}
groups={method:{str(driver):{tr:aggregate([x for x in evidence if x['method']==method and x['driver']==driver and x['transition']==tr]) for tr in ['SS','SF','FS','FF']} for driver in [8,9]} for method in ['RA','ALIGN']}
prefix_groups={}
for method in ['RA','ALIGN']:
 prefix_groups[method]={}
 for tr in ['SS','SF','FS','FF']:
  rr=[x for x in evidence if x['method']==method and x['driver']==9 and x['transition']==tr]
  assert all(x['prefix32_steps']==32 for x in rr)
  prefix_groups[method][tr]={'n':len(rr),'mode_difference':sum(x['prefix32_mode_difference'] for x in rr),'gripper_flip':sum(x['prefix32_gripper_flip'] for x in rr),'median_translation_action_l2':float(np.median([x['prefix32_translation_action_l2'] for x in rr])) if rr else None}
firsts={method:{'pairs':sum(p['method']==method for p in pairs),'same_initial_observations':sum(p['method']==method and p['same_initial_observation'] for p in pairs),'shared_history_first_action_difference_steps':sorted(set(p['first_action_difference_on_shared_history'] for p in pairs if p['method']==method and p['first_action_difference_on_shared_history'] is not None)),'first_obs_divergence_steps':sorted(set(p['first_observation_divergence_step'] for p in pairs if p['method']==method and p['first_observation_divergence_step'] is not None))} for method in ['RA','ALIGN']}
summary=js(D/'PAIRED_OUTCOME_SUMMARY.json');pooled=[]
for method in ['RA','ALIGN']:
 rows=[x for x in summary if x['method']==method];cnt={k:sum(x[k] for x in rows) for k in ['SS','SF','FS','FF']};pooled.append({'method':method,**cnt,'stage8_success':cnt['SS']+cnt['SF'],'stage9_success':cnt['SS']+cnt['FS'],'n':60})
result={'status':'DESCRIPTIVE_MECHANISM_ANALYSIS_COMPLETE','per_trajectory':evidence,'outcome_strata':groups,'first_differences':firsts,'pooled_outcomes':pooled,'training_authorized_by_this_analysis':False,'limitation':'No time of irreversible failure or causal attribution from offline differences. Original RGB windows retained privately but not visually adjudicated in this scalar analysis.'}
result['posthoc_fixed32_exposure_check']=prefix_groups
(D/'MECHANISM_ANALYSIS.json').write_text(json.dumps(result,indent=2),encoding='utf8')
text=['# 同历史决策与失败关联：已完成分析','','本次是新的共同噪声协议，不能与原正式AUC混算。全部240回合、120对Stage8/9、54646个有效决策步；输入history精确重构、执行动作与噪声复核通过。旧模型和科学源码未改变，无新训练。','','|seed|方法|Stage8成功/20|Stage9成功/20|成功→失败|失败→成功|','|---|---|---|---|---|---|']
for x in summary:text.append('|{seed}|{method}|{stage8_success}|{stage9_success}|{SF}|{FS}|'.format(**x))
text+=['','## 是否是ALIGN独有退化机制？','','共同窗口离线RA也有分量切换、夹爪纠错和部分平均Huber改善，不能当成ALIGN独有证据。新配对闭环RA合计37SS/6SF/9FS/8FF，Stage8 43/60→Stage9 46/60；ALIGN32SS/13SF/7FS/8FF，45/60→39/60。这里为描述计数，三个seed并非足够的显著性或保护保证。ALIGN下降主要由seed41贡献；seed42/43净变化为0，但存在相互抵消的SF与FS，不可写成没有个体遗忘。','','## 从哪里开始出现不同决策？','', '以下步号为0起算，差异来自同一实际历史上的两检查点。']
for m,v in firsts.items():text.append(f"{m}：{v['same_initial_observations']}/{v['pairs']}配对首观测完全相同；共同历史上首次动作差异步号集合{v['shared_history_first_action_difference_steps']}，两驱动轨迹首次观测分歧步号集合{v['first_obs_divergence_steps']}。动作数值很早就不同，所以不能把之后不同轨迹的同时间步差异当作策略条件漂移，也不能把首次非零差异直接认定为有害。")
text+=['','## 分量与夹爪事件是否区分失败？','','以Stage9实际驱动历史为例，每步同时查询Stage8/9；跨stage分量不同与单模型沿时间切换是不同统计。下表按轨迹计数，避免把重叠帧视作独立样本。','','|方法|Stage8→9结果|轨迹数|出现分量不同|出现夹爪异号|终止前8步异号|','|---|---|---|---|---|---|']
for m in ['RA','ALIGN']:
 for tr in ['SS','SF','FS','FF']:
  g=groups[m]['9'][tr];text.append('|%s|%s|%d|%d|%d|%d|'%(m,tr,g['n'],g.get('mode_difference_episodes',0),g.get('gripper_flip_episodes',0),g.get('terminal8_flip_episodes',0)))
text+=['','这些事件在SS/FS等非退化配对也会出现；是否有害需要具体历史的抓取/接触/目标进展语义，不能凭事件发生或平均距离排名判定。本文没有把动态动作幅度阈值调到最能解释结果，也没有宣称已确定某个不可逆失败步。终止前与首次事件的原始双相机窗口仍私有保存；本次标量分析没有进行图像人工语义裁决，不能把保存图像写成已看图确认失败原因。','','## 是否进入信赖域三分叉？','','暂不启动。现有证据重复观察到ALIGN的SF，但未识别一个跨seed、能将有害变化与正确纠错区分开的共同机制；尤其seed42/43有相等FS抵消，RA也会SF。此时直接压制分量或夹爪变化可能同时锁住旧错误。用户规定的“发现可重复有害变化”尚未形成可执行的选择性保护规则。保留240回合私有历史与完整标量，后续可用已存关键窗口做语义审查/另登记同状态干预，再决定同Stage8三组保护；不自动扫描训练超参或追加rollout。','','本轮两项诊断已完成并交付，保护训练0。当前结论是证据边界，不是信赖域无效的实验结论。']
text+=['','## 具体反例与时间暴露限制','','ALIGN seed41初态12是成功→失败：Stage9驱动600步中两检查点从未出现夹爪均值异号，首次分量编号差在第12步。说明夹爪翻转不是所有退化的必要条件。seed41初态5却是失败→成功：第48步首次分量差、第77步夹爪异号，第79次动作后成功；同样的事件可能伴随改进。seed43初态7在第62步开始同时出现分量/夹爪差异，最终超时，可作为后续语义审查候选，但尚不能断言第62步导致失败。所有这些比较均是同一Stage9实际历史上两检查点的输出，不是两条不同轨迹的动作相减。','','成功通常提前结束，SF/FF持续600步，故全程“出现过”与比例有暴露时长偏差。额外固定前32步作事后描述性检查（所有轨迹均有32步；不选更优阈值、不作显著性/保护准入），结果如下。平移动作L2是原控制量单位，非毫米。','','|方法|结果|n|前32步分量不同|前32步夹爪异号|前32步平均平移动作差的轨迹中位数|','|---|---|---|---|---|---|']
for m in ['RA','ALIGN']:
 for tr,g in prefix_groups[m].items():text.append('|%s|%s|%d|%d|%d|%.6f|'%(m,tr,g['n'],g['mode_difference'],g['gripper_flip'],g['median_translation_action_l2']))
(D/'MECHANISM_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf8')
print(json.dumps({'pooled':pooled,'firsts':firsts,'groups':{m:{tr:{k:v for k,v in g.items() if k!='first_flip_steps'} for tr,g in groups[m]['9'].items()} for m in groups}},ensure_ascii=False,indent=2))

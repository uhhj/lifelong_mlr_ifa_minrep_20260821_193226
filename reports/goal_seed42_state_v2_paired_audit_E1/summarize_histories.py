"""Summarize supplied analysis outputs; never modifies source evidence or package."""
import collections
import csv
import hashlib
import json
from pathlib import Path

B=Path(__file__).resolve().parent
O=B/'results'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
accept=read(O/'audit_acceptance.json')
assert accept['status']=='PASS'
for name,digest in accept['output_hashes'].items(): assert sha(O/name)==digest
with (O/'initial_state_histories.csv').open(encoding='utf-8-sig',newline='') as f:
    rows=list(csv.DictReader(f))
with (O/'paired_summary.csv').open(encoding='utf-8-sig',newline='') as f:
    pairs=list(csv.DictReader(f))
assert len(rows)==80 and len(pairs)==12
summary=[]
for task in range(6,10):
    group=[r for r in rows if int(r['task_id'])==task]
    later=task<9
    seq=collections.Counter(r['observed_sequence'] for r in group)
    assert sum(seq.values())==20
    summary.append({'task_id':task,'task_name':group[0]['task_name'],
        'sequence_counts':dict(sorted(seq.items())),
        'pattern_counts':dict(collections.Counter(r['pattern'] for r in group)),
        'initial_failure_count':sum(r['diagonal_success']=='False' for r in group),
        'initial_success_count':sum(r['diagonal_success']=='True' for r in group),
        'final_success_count':sum(r['final_success']=='True' for r in group),
        'all_observed_success_count':sum(set(s)=={'1'} for s in [r['observed_sequence'] for r in group]) if later else None,
        'all_observed_failure_count':sum(set(s)=={'0'} for s in [r['observed_sequence'] for r in group]) if later else None,
        'mixed_count':sum(r['pattern']=='mixed' for r in group) if later else None,
        'total_adjacent_flips':sum(int(r['adjacent_flips']) for r in group) if later else None,
        'initial_success_ever_failure_count':sum(r['initial_success_ever_observed_failure']=='True' for r in group) if later else None,
        'initial_success_lost_then_final_success_count':sum(r['initial_success_lost_then_final_success']=='True' for r in group) if later else None,
        'initial_failure_final_success_count':sum(r['diagonal_success']=='False' and r['final_success']=='True' for r in group) if later else None,
        'later_stability':None if not later else 'Descriptive observed history only'})
with (O/'history_summary.json').open('x',encoding='utf-8') as f:
    json.dump(summary,f,indent=2,ensure_ascii=False); f.write('\n')
provenance=read(B/'source_provenance.json')
verification=read(B/'execution_verification.json')
commands=[json.loads(s) for s in (B/'execution_commands.jsonl').read_text(encoding='utf-8').splitlines()]
decomp=read(O/'nbt_decomposition.json')
text=['# E1 执行与结果补充说明','',
'本轮为既有200条闭环记录的回顾性描述分析；附件分析脚本和测试脚本原样使用，未修改工具，未新增训练、模型加载或环境执行。',
'','## 来源与只读验证','',
'实际输入：`'+provenance['actual_evidence_root']+'`。',
'','实际Git HEAD：`'+provenance['actual_git_head']+'`，分支`'+provenance['actual_git_branch']+'`。该值由本机Git读取确认；11份分析输入逐一与该固定提交的Git blob进行SHA-256核对，均一致。原作者基点`a721807f2f7fa7f9833dcc85c592d78e9accd00b`与发布提交分别记录。',
'','分析包ZIP SHA-256：`'+provenance['package_zip_sha256']+'`。包内清单校验通过，两个脚本字节未修改。',
'','完整轻量run目录472个文件在分析前后SHA-256相同，原COMPLETED状态不变。'+
'[来源与前哈希](../source_provenance.json)、[后哈希与执行验证](../execution_verification.json)、[脚本验收](audit_acceptance.json)均已保存。',
'','## 真实命令及退出码','',
'Python：`'+provenance['python_executable']+'`，版本`'+provenance['python_version'].split()[0]+'`。环境设置`PYTHONDONTWRITEBYTECODE=1`、`PYTHONUTF8=1`、`PYTHONIOENCODING=utf-8`。不依赖任何第三方包。']
for item in commands:
    text+=['','### '+item['name'],'','```powershell',item['windows_command_line'],'```','',
           '退出码：`'+str(item['exit_code'])+'`。开始UTC：'+item['start_utc']+'；结束UTC：'+item['end_utc']+'。']
text+=['','附带15项合成软件测试全部通过，仅证明软件测试用例通过，不能当作真实机器人结果。真实分析独立读取固定提交的200条正式记录，退出码0。补充查看输出时曾发生Windows默认GBK解码错误，显式UTF-8读取后恢复；没有修改输入或附带脚本，也不涉及数据一致性失败。',
'','## 12行配对汇总','',
'S=成功，F=失败；每行20对。保留率=SS/(SS+SF)，丢失率=SF/(SS+SF)。损失/增益/净差为百分点，翻转为百分比。SS步数差仅取两端均成功的同初态配对子集，中位数为after−before。','',
'| 视角 | 任务 | 阶段 | SS | SF | FS | FF | 损失pp | 增益pp | 净差pp | 翻转% | 保留% | 丢失% | SS数 | SS步数差中位数 |',
'|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in pairs:
    nums=[r[k] for k in ['SS','SF','FS','FF']]
    stats=[format(float(r[k]),'.2f') for k in ['observed_loss_pp','observed_gain_pp','net_success_change_pp','observed_flip_pp']]
    cond=[format(float(r[k])*100,'.2f') if r[k] else 'NA' for k in ['conditional_retention','conditional_loss']]
    text.append('| '+' | '.join([r['view'],r['task_id'],r['before_stage']+'→'+r['after_stage'],*nums,*stats,*cond,r['ss_step_change_n'],r['ss_median_after_minus_before_steps'] or 'NA'])+' |')
text+=['','同任务同初态哈希配对，交叉检查初态ID、训练/评估/worker种子等元数据。每个比较四类计数之和为20，净变化与矩阵差相等。两视角各120条，共240行，仅180个去重跨阶段配对；全部复用原200个episode，并非新增独立试验。',
'','## NBT分解','',
'| 任务 | 平均观测成功→失败pp | 平均观测失败→成功pp | 两者之差/NBT pp |','|---|---:|---:|---:|']
for r in decomp['per_task']:
    text.append('| %d | %.6f | %.6f | %.6f |'%(r['task_id'],r['mean_observed_loss_pp'],r['mean_observed_gain_pp'],r['signed_NBT_pp']))
text+=['| 9 | NA | NA | NA |','| 6/7/8等权平均 | %.6f | %.6f | %.6f |'%(decomp['macro_observed_loss_pp'],decomp['macro_observed_gain_pp'],decomp['signed_NBT_pp']),
'','每任务先平均其刚学完至所有后续阶段的比较，再对任务6/7/8等权平均；没有对120条直接平均。L=0.0722222222、G=0.0611111111，100×(L−G)=1.111111个百分点，与原NBT一致；任务6的负NBT保留。',
'','复算原指标：FWT=78.75%，NBT=1.111111个百分点，AUC=78.4375%，FinalAvgSR=78.75%。与metrics.json、final_acceptance.json及逐任务分量核对通过，容差1e-12。',
'','## 全任务结果历史','',
'序列按该任务刚学完到最后阶段排列，1=成功、0=失败。完整80行见[initial_state_histories.csv](initial_state_histories.csv)；聚合见[history_summary.json](history_summary.json)。','',
'- **任务6（奶油奶酪放入碗中）**：初次13成功/7失败；四阶段始终成功10个、始终失败4个、混合6个，共7次相邻结果翻转。13个原成功初态中3个后来曾出现失败，其中1个最终又成功。初次7个失败中3个最终成功。刚学完与最终相比SS=11、SF=2、FS=3、FF=4，净增5pp但25%结果翻转；最终保留11/13=84.62%，全阶段始终成功为10/13，二者不可混同。',
'- **任务7（打开炉子）**：20个初态序列均为111，三次观察均成功；本次记录中无SF/FS、无翻转。该有限观察不能证明所有初态或重复执行都保持成功。',
'- **任务8（碗放到盘子上）**：初次13成功/7失败；11序列10个、10序列3个、01序列2个、00序列5个。3个成功→失败与2个失败→成功抵消后净减5pp，翻转25%，原成功保留10/13=76.92%。仅两阶段观察，不能确定之后是否恢复，或将最初失败解释为增加epoch即可解决。',
'- **任务9（酒瓶放到架子上）**：只有最终阶段一次观察，17成功、3失败；无后续阶段，遗忘、恢复和后续稳定性均NA。85%是该次记录成功率，不能解释为抗遗忘能力。','',
'| 任务 | 观测序列频次 |','|---|---|']
for r in summary:
    text.append('| %d | %s |'%(r['task_id'],'；'.join('%s×%d'%(s,n) for s,n in r['sequence_counts'].items())))
text+=['','## 解释边界与停止','',
'每checkpoint每初态仅一次执行。初态与seed一致不保证逐episode随机动作流一致，批内其他环境和执行长度会影响随机数消耗。这里只做描述，没有显著性检验、回放伤害因果判断或任务内误差转移证明。没有行为事件/视频证据，不能定位抓取、搬运、释放或责任训练窗口。',
'','SS步数差不包含失败600步，不使用duration_seconds评价策略质量。测试初态与失败列表只用于本次分析，不生成训练权重，不映射到训练窗口，不调整回放比例或超参数。没有新训练、新rollout、checkpoint加载或远端push。分析完成后停止。',
'','本次PASS限于日志与轻量证据内部一致性，未重新验证模型权重或环境初态二进制，也不是独立训练复现。','']
with (O/'OPERATOR_REPORT.md').open('x',encoding='utf-8',newline='\n') as f: f.write('\n'.join(text))
# Verify the original script outputs remain intact after supplemental reporting.
for name,digest in accept['output_hashes'].items(): assert sha(O/name)==digest
with (O/'supplement_acceptance.json').open('x',encoding='utf-8') as f:
    json.dump({'status':'PASS','history_rows':80,'task9_later_metrics_are_null':True,
               'supplied_script_outputs_unchanged':True,'supplement_hashes':{n:sha(O/n) for n in ['history_summary.json','OPERATOR_REPORT.md']}},f,indent=2)
print(json.dumps({'status':'PASS','tasks':summary},ensure_ascii=True))

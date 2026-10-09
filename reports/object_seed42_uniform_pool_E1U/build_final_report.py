from pathlib import Path
import json,tarfile,hashlib,datetime
F=Path(__file__).resolve().parent;L=F.parent;E=F/'evidence';E.mkdir(exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert sha(F/'final_light_evidence.tar.gz')=='2bb9ba5f0ed063cab4d2de7ef1efbacd03c3a64a9a8d1c0245844b58ee35eb4e'
with tarfile.open(F/'final_light_evidence.tar.gz') as tf:
    for m in tf.getmembers():
        assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
        p=E/m.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tf.extractfile(m).read())
def read(n):return json.loads((E/n).read_text())
manifest=read('FINAL_DOWNLOAD_MANIFEST.json')
assert all(sha(E/n)==v['sha256'] and (E/n).stat().st_size==v['size'] for n,v in manifest.items())
a=read('INDEPENDENT_ACCEPTANCE.json');u=read('UNIFORM_AUDIT.json');pools=read('pool_diagnostics.json');flow=read('full_replay_flow.json')
assert a['status']=='COMPLETED' and a['independent_audit']=='PASS' and a['extension_seeds']==[]
assert (E/'pipeline.exit').read_text().strip()=='0' and (E/'independent_acceptance_v2.exit').read_text().strip()=='0'
draws=sum(p['draws'] for p in pools);unique=sum(p['unique'] for p in pools);expected=sum(p['expected_uniform_unique'] for p in pools)
repeat=1-unique/draws;expected_repeat=1-expected/draws
lines=['E1-U LIBERO-OBJECT seed42 最终报告','状态：COMPLETED；独立验收 PASS。',
    '正式流程：2026-10-09 12:55:02—19:45:02（北京时间），6小时50分钟。',
    '400 epoch、105500 optimizer step、28规定checkpoint、主200＋次680＝880条正式记录。',
    '独立验收在19:55之后执行；最初只读浮点精确比较失败，修复版验收exit0，原错误日志保留。',
    '没有追加训练或闭环；没有重跑E0/E1；没有推送远端。','',
    '一、主口径（各阶段epoch99；单位为百分比/百分点；差值使用未舍入指标）',
    '版本              AUC             FWT             NBT']
for name in ('E0','E1','E1U'):
    m=u['versions'][name];lines.append('%-10s %15.9f %15.9f %15.9f'%(name,m['auc'],m['fwt'],m['nbt']))
for name in ('E1_minus_E1U','E1U_minus_E0'):
    m=u['differences'][name];lines.append('%s: AUC %.9f, FWT %.9f, NBT %.9f'%(name,m['auc'],m['fwt'],m['nbt']))
lines+=['','本次取消评分加权后，AUC相对E1下降3.645833个百分点；均匀池结构相对E0下降4.0625个百分点。',
    '本次seed42中评分加权有正向观测增量，但原E1仍未超过E0。均匀池方案未满足超过E0的扩展条件，停止，不运行43/44，不叠加E2。',
    '这不是稳定效应或统计显著性结论，也不能推断小候选池造成全部损失。','']
for name in ('E0','E1','E1U'):
    lines.append(name+' 主矩阵：阶段\\任务       6       7       8       9')
    matrix=a['tables']['primary']['matrix'] if name=='E1U' else u['versions'][name]['matrix']
    for stage,row in zip(range(6,10),matrix):lines.append(str(stage)+'   '+'   '.join('—' if x is None else str(x) for x in row))
lines+=['','二、E1-U次口径（独立680条；按当前任务成功率选择，平局取较晚epoch）']
sec=a['tables']['secondary'];lines.append('所选epoch: '+json.dumps(sec['selected_epochs']))
lines.append('指标: '+json.dumps(sec['metrics']))
for stage,row in zip(range(6,10),sec['matrix']):lines.append(str(stage)+'   '+'   '.join('—' if x is None else str(x) for x in row))
lines+=['','三、采样与只读保护验收',
    '400个epoch的有序H/N/C池、刷新步和尝试替换位置与E1比较均通过；181次刷新，389个任务候选池。',
    '数值回退、无效候选与IFA否决均为0。每个池64个候选、概率严格1/64、ESS=64。',
    '230710次尝试替换，230416次真正改变ID；294次抽中原ID。全部1052400个实际batch槽位有身份记录。',
    '最高分四分位实际抽样占比%.4f%%，期望25%%；分数继续计算，只作旁路日志。'%(u['top_quartile_observed_share']*100),
    '按每个池自身实际抽样次数汇总：池内实际重复率%.6f%%，同池均匀有放回解析期望%.6f%%。'%(repeat*100,expected_repeat*100),
    '该池内统计不等于全回放流重复率；全回放流按阶段/旧任务记录如下：']
for r in flow:lines.append('stage%s/task%s: slots=%s unique=%s repeat=%.6f%%'%(r['stage'],r['task'],r['slots'],r['unique'],100*r['repeat_fraction']))
lines+=['阶段末buffer与E1的成员及载荷哈希一致：781、1521、2301、2961。',
    '168份冻结源码/配置、5921特征、官方权重与28新checkpoint哈希通过；10336份原E1文件/源码未改变。',
    '真实环境尝试：主20个10路group、次68个10路group，共880 lane attempts，880提交记录，0技术重试、0未提交尝试。',
    '原28规定checkpoint均被正式评估入口使用；不另加载模型进行额外环境测试。',
    '原20＋新增8项软件测试PASS，真实GPU隔离预检恰好5次optimizer调用；软件测试不充当正式实验结果。','',
    '四、成本（组件时间已包含在训练总时间；不可重复相加）']
for k,v in u['cost'].items():lines.append(k+': '+str(v))
lines+=['训练约57.09分钟；主评估71.05分钟；次评估281.43分钟。',
    'peak_gpu_bytes是PyTorch训练进程记录的峰值allocated内存，不代表nvidia-smi整卡峰值。',
    'IO时间已包含在评分和替换处理时间内；身份元数据构造等未单列成本包含在训练总时间。','',
    '五、真实命令与退出码',
    'source /data/mlr_ifa/experiment_workspaces/E1U_uniform_pool_20261009/source/protective_e1/env.sh',
    'python -B -m unittest discover -s ../tests -v ：exit0，28项。',
    'python -m torch.distributed.run --nproc_per_node=1 --master_port=37986 -m protective_e1.preflight ：exit0，恰好5次实际GPU optimizer调用。',
    'bash protective_e1/pipeline.sh ：pipeline.exit=0；详见commands.log。',
    'python -m torch.distributed.run --nproc_per_node=1 --master_port=37985 -m protective_e1.train ：exit0。',
    'bash protective_e1/evaluate.sh primary ：exit0，200条。',
    'bash protective_e1/evaluate.sh secondary ：exit0，680条。',
    'python -B -m protective_e1.finalize ：exit0。',
    '/data/mlr_ifa/env/bin/python <输出目录>/independent_acceptance.py ：exit1，仅55.00000000000001与55严格相等比较失败。',
    '/data/mlr_ifa/env/bin/python <输出目录>/independent_acceptance_v2.py ：exit0；绝对容差1e-10，仅百分比浮点比较，episode计数/布尔结果仍精确。',
    '验收修复为外置只读脚本；原脚本/错误日志保留，冻结源码与原始记录未改变。','',
    '六、解释边界',
    '单seed不是稳定效应估计；E1U−E0同时包含候选池限制和替换结构，不能解释为池大小的纯因果效应。',
    'E1缺少保留分支完整ID，不能伪造E1全回放流统计；本轮不以Loss定义行为受损。',
    '本轮保留评分计算，不宣称为移除评分计算后的高效版本。',
    'E0曾使用RTX4090，E1/E1-U使用RTX3090；E0训练耗时和阶段buffer历史快照缺失仍披露。','',
    '七、产物',
    '远端输出：/data/mlr_ifa/experiments/E1U_uniform_pool_seed42_20261009',
    '远端checkpoint：上述目录/run_001/mlrifa_{6,7,8,9}_model_ep{70,75,80,85,90,95,99}.pth',
    '本地证据：'+str(E),
    '代码diff：'+str(L/'source.patch'),
    '875个轻量证据文件逐一校验下载SHA256；不下载checkpoint、buffer或特征。']
(F/'FINAL_REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
receipt=dict(status='COMPLETED',independent_audit='PASS',verified_download_files=len(manifest),archive_sha256=sha(F/'final_light_evidence.tar.gz'),
    primary=a['tables']['primary'],secondary=sec,extension_seeds=[],pipeline_exit=0,independent_acceptance_v2_exit=0,report=str(F/'FINAL_REPORT.txt'))
(F/'FINAL_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
(L/'RUN_STATUS.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
h=json.loads((L/'MONITOR_HANDOFF.json').read_text(encoding='utf8'));h['status']='COMPLETED';h['latest_snapshot']=receipt
h['monitor'].insert(0,'COMPLETED: all880 records and independent acceptance PASS. Primary AUC73.645833 < E0. No extension, no relaunch. Monitor stopped.')
(L/'MONITOR_HANDOFF.json').write_text(json.dumps(h,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(status=receipt['status'],files=len(manifest),actual_pool_repeat=repeat,expected_pool_repeat=expected_repeat,report=receipt['report']),ensure_ascii=False,indent=2))

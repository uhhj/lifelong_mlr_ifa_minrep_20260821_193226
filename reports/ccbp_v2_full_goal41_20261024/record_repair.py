from pathlib import Path
import json
R=Path(__file__).resolve().parent
d=json.loads((R/'deploy_engineering02_20261025_000802/EVIDENCE.json').read_text(encoding='utf8'))
(R/'ACTIVE_ROOT.json').write_text(json.dumps(dict(root=d['root'],controller=d['controller'],contract=d['contract_sha256'],frozen_files=d['frozen_files'],output_layout='Original independent full-sequence output roots, not prior Task9 branch',automation_id=None,previous_root=d['cpu_preflight']['previous_root'],inherited_completed_B_task6_epochs=100),indent=2),encoding='utf8')
note='\n\n## 工程02接续\n\n原完整v1在B Task6完成100轮/10000步并实际tensor精确复现ALIGN后，于Task7教师准备前因CPU/GPU张量直接torch.equal报错退出。check_readonly_20261025_000623已绑定退出1、无进程/GPU空闲及44hash不变。0 Task7更新、0正式环境。新独立engineering02仅将比较移到CPU并按完整恢复继续Task7，不重跑Task6。CPU preflight核100键CP与恢复model精确、全部状态字段及100轮history；GPU实际恢复和自己的Task7准入待最新快照确认。原恢复文件未额外复制，不因紧张容量伪称保全额外payload副本；原model-only Task6 CP永久保留，完成history/原失败/hash和源代码均保留。\n\n新root='+d['root']+'，controller='+str(d['controller'])+'，合同'+d['contract_sha256']+'，65冻结项。旧根不再是活跃目标。科学参数/输入/初始化/预算不改，未重试科学负结果。\n'
for n in ['IMPLEMENTATION_NOTES.md','REPORT.md']:
 p=R/n;p.write_text(note+p.read_text(encoding='utf8'),encoding='utf8')
print(d['contract_sha256'])

"""Record an audited observation without changing a frozen experiment."""
from pathlib import Path
import json,sys,hashlib
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);a=json.loads((D/'LOCAL_AUDIT.json').read_text());v=json.loads((D/'EVIDENCE.json').read_text())
deployment=json.loads((R/'deploy_remote_20261027_234719/EVIDENCE.json').read_text())
state=dict(a,status='PARTIAL_TRAINING' if not a['failures'] else 'PARTIAL_ENGINEERING_FAILURE',latest_evidence=D.name,root=deployment['root'],controller=deployment['controller'],contract=deployment['contract'],mechanism_budget=dict(epochs=1700,accepted_updates=403000,physical_training_optimizer_calls=494700,formal_episodes=1120),followup=dict(required=True,after='mechanism acceptance and analysis',method='latest complete S: support weighting plus gate ONLY',suite='OBJECT',seeds=[41,42,43],other_full_methods=False))
state.update(automation_id='s-object-seed',monitor_status='ACTIVE')
(R/'RUN_STATUS.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf8');(R/'ACTIVE_ROOT.json').write_text(json.dumps(dict(deployment,status=state['status'],latest_evidence=D.name),indent=2)+'\n',encoding='utf8')
phase=json.loads(v['light']['STATUS.json']['text']);case=phase.get('case');arm=phase.get('arm');eta=None
if phase['phase']=='train':
    h=json.loads(v['light'].get(case+'/EPOCHS_'+arm+'.json',{'text':'[]'})['text'])
    if h:
        mean=sum(x['seconds'] for x in h[-10:])/len(h[-10:]);eta=dict(case=case,arm=arm,observed_epochs=len(h),recent_epoch_seconds=mean,remaining_training_minutes=(100-len(h))*mean/60,preparation_saving_extra=True,not_total_queue_prediction=True)
(R/'ETA.json').write_text(json.dumps(eta,indent=2)+'\n',encoding='utf8')
report='# 支持度与门控共同起点机制实验\n\n状态：'+state['status']+'；观察 '+a['observation']+'。\n\n已实际接受完整epoch账目 '+str(a['epochs'])+'；主更新 '+str(a['accepted_updates'])+'；候选实际训练优化器调用 '+str(a['physical_training_optimizer_calls'])+'；隔离准入 '+str(a['own_isolated_optimizer_calls'])+'；正式环境 '+str(a['formal_episodes'])+'/1120。尚无科学收益或最终验收结论。\n\n'+json.dumps(a['counts'],ensure_ascii=False,indent=2)+'\n\n'+json.dumps(eta,ensure_ascii=False,indent=2)+'\n\n首组GPU真实准入已证明lambda=0与原C完整更新相等、q=1与原S完整更新相等、10步C重放、完整恢复及探针/梯度测量不改变训练状态；其余组必须各自通过。额外候选和探针计算单列，不称等算力。\n\n机制实验保留C/S/Q/G四组与OBJECT41的U开发对照。之后仅最新完整S运行OBJECT41/42/43完整序列，旧方法不新增运行；不得将机制阶段结束误作用户全部任务完成。参见PROTOCOL和IMPLEMENTATION_NOTES。\n\n本地一次辅助编辑命令因PowerShell引号产生Python解析错误，未执行目标修改；部署仍使用完整已审科学PROTOCOL，未改科学源码或预算。最终外置wrappers仅编译，须实际运行验收。\n'
(R/'REPORT.md').write_text(report,encoding='utf8');print(json.dumps(dict(status=state['status'],eta=eta,epochs=a['epochs'],evidence=D.name),ensure_ascii=False))

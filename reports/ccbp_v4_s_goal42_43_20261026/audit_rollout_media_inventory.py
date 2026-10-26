"""Answer media availability from accepted evidence and frozen writer code; no rollout."""
import json,hashlib,datetime,ast
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'20261024_semantic_intervention'
js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=js(P/'LOCAL_SCREEN_AUDIT.json');sources={}
for name in ['REPORT.md','SEMANTIC_NOTES.md','ASSET_PREFLIGHT.json','WINDOW_COVERAGE_AUDIT.json']:
 assert sha(P/name)==old['evidence'][name];sources[str(P/name)]=sha(P/name)
pre=js(P/'ASSET_PREFLIGHT.json');assert pre['source_unchanged'] and pre['supervisor_exit']==pre['worker_exit']=='0'
assert all(pre['sample_npz_keys_shapes'][k]['dtype']=='uint8' for k in ['agentview_image','robot0_eye_in_hand_image'])
for seed in (42,43):
 for name in ['collect.py','evaluate.py']:
  p=R/f'seed{seed}'/name;s=p.read_text(encoding='utf8');ast.parse(s);assert all(x not in s for x in ['VideoWriter','imageio','save_video','.mp4','.png','.jpg'])
  sources[str(p)]=sha(p)
out=dict(status='PASS_LOCAL_EXISTING_EVIDENCE_AND_WRITER_INSPECTION',observation=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),current_S=dict(seeds=[42,43],training_memory='Sparse encoded feature windows plus teacher Normal parameters, metadata and hashes; not raw RGB.',formal_evaluation='Episode outcomes, attempt/initial/noise/action-stream hashes; no image/video writer.'),earlier_RA_ALIGN_paired=dict(seeds=[41,42,43],task=8,driving_stages=[8,9],raw_cameras=['agentview_image','robot0_eye_in_hand_image'],resolution=[128,128,3],storage='Lossless NPZ key windows in remote private storage',saved_windows=['first_mean_delta: preceding <=8 frames','first_mode_switch: preceding <=8 frames','first_gripper_flip: preceding <=8 frames','terminal: preceding16 frames plus terminal observation'],full_video=False,semantic_reviewed_seed=41,semantic_reviewed_trajectories=80,semantic_reviewed_source_windows=287,frames_reviewed_per_noninitial_window='first/middle/last'),sources=sources,limitations=['Earlier remote assets were verified in accepted 2026-10-24 audits; current server file existence was not rechecked in this question.','Encoded features cannot reconstruct missing original RGB.','No new rollout, training, remote mutation, media export or private payload download.'])
(R/'ROLLOUT_MEDIA_INVENTORY_20261026.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
text='''# Rollout截图与视频存档核对

当前S GOAL seed42/43没有保存完整rollout视频或原始RGB截图。训练S缓存保留的是抽样视觉编码和教师执行分布，正式评估记录结局/步数/初态和随机数等hash；这些不能替代相机录像。源码核对没有视频或图片文件写入路径。

之前RA/ALIGN的GOAL Task8配对诊断（seed41/42/43，Stage8/9驱动）保存过128×128双相机关键窗口，以无损NPZ存于服务器私有目录。包括首次动作差异、首次分量差异、首次夹爪异号各最多前8帧，以及终止前16帧和终止观测。它们可以用于导出截图或短片，但不能拼成没有缺帧的全程视频。

2026-10-24语义筛查实际看过seed41的80轨迹、287源窗口，每个非初始窗口首/中/末三帧；原始RGB未下载或发布。现存证据已核SHA，本次未重新连接服务器确认旧文件当前仍在。没有新增环境回合、训练或修改活跃协议。源绑定详见ROLLOUT_MEDIA_INVENTORY_20261026.json。
'''
(R/'ROLLOUT_MEDIA_INVENTORY_20261026.md').write_text(text,encoding='utf8');print(json.dumps({k:v for k,v in out.items() if k not in ['sources']},indent=2))

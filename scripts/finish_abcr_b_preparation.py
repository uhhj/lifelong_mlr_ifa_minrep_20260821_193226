"""Report B prerequisites honestly; never manufacture calibration labels."""
import argparse
import json
import platform
import subprocess
from pathlib import Path
import torch
from abcr_pair_availability_a_v1.io import write,sha,read_table


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root
    out=root/'reports/abcr_pair_validation_b_v1';package=Path(__file__).resolve().parents[1]/'abcr_pair_validation_b_v1'
    prep=json.loads((out/'PREPARATION_SUMMARY.json').read_text())
    assert prep['trajectory_overlap']==0
    assert not (out/'frozen.json').exists()
    log=(out/'unit_tests.log').read_text();assert 'Ran 11 tests' in log and '\nOK' in log
    write(out/'UNIT_TEST_REPORT.json',dict(status='PASS',tests=11,log_sha256=sha(out/'unit_tests.log')))
    checkpoint=root/'checkpoints/object_multitask_model_ep99.pth'
    cp=torch.load(str(checkpoint),map_location='cpu');s=cp['state_dict'];cfg=cp['cfg']
    image_keys=[k for k in s if 'image_encoders' in k or 'image_proj' in k]
    context=dict(repository_root=str(root),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        research_reference='84094706672f45777c03093bfa25c3fcb21e0c64',
        source_files={p.name:dict(path=str(p),sha256=sha(p)) for p in sorted(package.glob('*')) if p.is_file()},
        python=platform.python_version(),pytorch=torch.__version__,cpu_only=True,cuda_initialized=torch.cuda.is_initialized(),
        checkpoint=dict(path=str(checkpoint),bytes=checkpoint.stat().st_size,sha256=sha(checkpoint),
            cpu_load='PASS',image_parameter_keys=len(image_keys),
            encoder=cfg.policy.image_encoder.network,LoRA_rank=int(cfg.policy.rank),LoRA_alpha=int(cfg.policy.alpha),
            projection='image_proj:768->512 before FiLM; weights present',
            unmodulated_output='native image_proj(image_encoders(raw_preprocessed_pixels)), before film_scale/film_shift'),
        physical_source_files={str(p):sha(p) for p in [root/'libero/libero/envs/bddl_base_domain.py',
            root/'libero/lifelong/models/complete_policy.py',root/'libero/lifelong/models/modules/rgb_modules.py']},
        actual_feature_reextraction='NOT_RUN; not implied by checkpoint loading',
        formal_training_resumed=False)
    assert not context['cuda_initialized']
    write(out/'EXPERIMENT_CONTEXT.json',context)
    audit=read_table(out/'PHYSICAL_FIELD_AUDIT.csv')
    overshoot=sum(int(x['excluded_gripper_out_of_travel']) for x in audit)
    source=json.loads((root/'reports/abcr_pair_availability_a_v1_review_v1/B_PREPARATION.json').read_text())
    summary=dict(**prep,user_script_sha256=sha(package/'original_experiment_b.py'),user_protocol_sha256=sha(package/'USER_PROTOCOL.md'),
        unit_tests='11/11 PASS',mapping_counts=source['counts'],
        incremental_alignment_verified=False,excluded_sampled_out_of_travel_anchors=overshoot,
        blockers=['BASE_INDEPENDENT_HUMAN_LABELS_ABSENT','PRE_FILM_FEATURE_EXPORT_PENDING',
            'INCREMENTAL_PIXEL_FEATURE_ALIGNMENT_PENDING','BASE_TRAJECTORY_DISJOINT_VALIDATION_PENDING'],
        tau_s=1,tau_v=None,calibration_positive_labels=0,calibration_negative_labels=0,
        annotation_entry='annotation/index.html',ABCR_training=False)
    write(out/'SUMMARY.json',summary)
    text='''# ABCR 实验 B：输入核验与人工标注准备

状态：WAITING_FOR_INDEPENDENT_HUMAN_LABELS。不是实验 B 通过，也没有开始 ABCR 训练。

用户提供的脚本和协议保持原始字节。当前 CPU 合同测试 11/11 通过。
已读取原始 HDF5、核验记录字段和基础 checkpoint；没有从语言调制缓存伪造 pre-FiLM 特征。

## 已完成

- robot_states 的 9 维为夹爪 qpos(2)、末端位置(3)、xyzw 四元数(4)，与源码和 HDF5 obs 字段相互核对；四元数另与 ee_ori 旋转向量逐帧交叉检查。
- 夹爪左右滑动关节范围来自每条演示 XML，归一化使用总行程 0.08 m。发现超行程帧；未裁剪，本包排除了 %d 个抽样锚点并保留逐轨迹统计。
- 六个基础任务的机器人基座变换一致。HDF5 无直接末端速度字段，本包只表示位姿可比，未冒充完整动态等价。
- 原 A 共 17,763 条缓存：17,751 个唯一完整动作窗口来源候选，12 个歧义。动作匹配不是像素匹配；未把这些记录 alignment_verified 标为 true。
- 基础 checkpoint 的 CPU 加载通过，包含 CLIP+LoRA 和 FiLM 前投影；下一步应从原图使用同一编码器导出特征，不能从旧潜变量逆推。

## 固定待标注包

基础任务0–5 的 15 个无向任务对；demo0–24 用于校准，demo25–49 用于独立验证，轨迹交集为0。
固定种子仅在预注册位姿门通过的候选中随机抽取，每个 demo 对最多一次；不按视觉得分或动作标签选样。
已生成校准 %d 对、验证 %d 对。人类标注尚为0，不能把准备好的候选数量当作正负样本数。

打开 annotation/index.html，逐对填 1=局部物理及相关物体状态可比、0=不可比、U=不确定，导出CSV。
标注视图不显示视觉相似度或动作类别。图片上侧是 agent view，下侧是腕部视角；为显示按 OpenGL 约定翻转，不改变模型预处理。
不要为满足至少20正/40负的下限强行标注；若不足，报告而不自动增补直到“通过”。

## 尚未执行

视觉阈值校准、独立轨迹匹配性能核验、增量7/8/9筛查、类别人工准确率和轨迹bootstrap均未执行。
tau_s=1 固定；tau_v 尚未定义。没有 frozen.json，没有实验 B 的 SHARE/BIF 结果。
原协议第3节规定缺少独立正负标注时停止，用户也已确认“尚未标注”。
待收到标注后，还需完成原图 pre-FiLM 特征导出和像素/窗口对齐检查，再按冻结流程校准和验证。
不因候选稀疏调整阈值，不把 Task8/Task6 的 A 类别作为真值。

`formal_training_resumed=false`；`CUDA initialized=false`；未修改正式算法、缓存或历史结果。
'''%(overshoot,prep['pairs_by_role']['calibration'],prep['pairs_by_role']['validation'])
    (out/'FINAL_REPORT.md').write_text(text,encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['status','pairs_by_role','unique_records','trajectory_overlap',
        'excluded_sampled_out_of_travel_anchors','tau_v','formal_training_resumed']},ensure_ascii=False))


if __name__=='__main__':main()

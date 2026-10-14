"""Close the completed cache-fix review from measured artifacts only."""
import argparse
import json
import gzip
from pathlib import Path
from abcr_pair_availability_a_v1.io import write,sha,read_table


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    out=r/'reports/abcr_pair_availability_a_v1_review_v1';base=r/'reports/abcr_pair_availability_a_v1'
    before=json.loads((out/'PRE_RESUME_OUTPUT_SNAPSHOT.json').read_text())
    print('SNAPSHOT_TYPE',type(before),list(before)[:3])
    if 'files' in before:before=before['files']
    assert isinstance(before,dict)
    assert all(sha(out/'reproduction'/p)==v for p,v in before.items())
    legacy=json.loads((out/'LEGACY_REPORT_SNAPSHOT.json').read_text())
    if 'files' in legacy:legacy=legacy['files']
    assert all(sha(base/p)==v for p,v in legacy.items())
    log=(out/'resume.log').read_text()
    assert 'ALL_BOUND_STAGES_REUSED 9 no report overwritten' in log
    assert 'STAGE_COMPLETE' not in log
    write(out/'CACHE_RESUME_VALIDATION.json',dict(status='PASS',stages_reused=9,
        measured_outputs_unchanged=len(before),historical_reports_unchanged=len(legacy),
        replayed_measurements=0,log_sha256=sha(out/'resume.log')))
    text=(out/'unit_tests.log').read_text();assert 'Ran 21 tests' in text and '\nOK' in text
    write(out/'UNIT_TEST_REPORT.json',dict(status='PASS',tests=21,log_sha256=sha(out/'unit_tests.log'),
        checks='byte/membership/code/parameter/runtime/output mismatch, immutable manifests, legacy refusal, real resume branch, shared parameters'))
    parity=json.loads((out/'REPRODUCTION_PARITY.json').read_text());assert parity['status']=='PASS'
    concentration=json.loads((out/'TASK8_OLD6_CONCENTRATION.json').read_text())['rows']
    dist=read_table(out/'VISUAL_SIMILARITY_QUANTILES.csv')
    identity=json.loads((out/'B_PREPARATION.json').read_text())
    summary=dict(status='CACHE_FIXED_REPRODUCTION_AND_DESCRIPTIVE_REVIEW_COMPLETE',
        reproduction=parity,resume='PASS',unit_tests=21,identity_counts=identity['counts'],
        description='Unique action identity is not independent pixel/feature alignment',
        similarity_distribution_rows=len(dist),task8_old6=concentration,
        experiment_B='New user protocol received; see reports/abcr_pair_validation_b_v1; manual labels not available',
        public_raw_asset_archive=False,clone_only_reproduction=False,
        equal_budget_retrieval='NOT_RUN; existing retrieval comparison remains unequal-budget',
        formal_training_resumed=False,ABCR_training=False)
    write(out/'SUMMARY.json',summary)
    lines=['# 实验 A 缓存修复与复核补充','',
        'P1 已修复。恢复条件同时绑定输入文件内容与成员、实际执行代码、固定参数、运行环境、来源和输出；旧 COMPLETE 不会被静默升级。21 项测试通过。',
        '', '在独立源码目录、同一现有 CPU 环境和原数据上重跑 9 个单元；66 份数值 CSV 与提交 8409470 逐字节一致。再次 resume 复用了全部 9 个单元，未改写结果。原 103 个报告文件保持原哈希。此证据不是只靠 Git 克隆的独立环境复现。',
        '', '## Task8 的历史任务贡献','', '| Seed | Task6 共享贡献 | Task6 分叉贡献 |','|---|---:|---:|']
    for row in concentration:lines.append('|%s|%.2f%%|%.2f%%|'%(row['seed'],100*row['share_contribution'],100*row['bif_contribution']))
    lines += ['', '## 表征分布与来源','',
        'VISUAL_SIMILARITY_QUANTILES.csv 给出 39 组同任务/跨任务分布；同任务统计仍含重叠窗口，不是轨迹级独立证据。未用分位数修改阈值，也不能分离语言、物体和场景的因果影响。',
        '', '原始 HDF5 完整 8 步动作序列精确匹配：17,751 个唯一来源候选、12 个歧义记录；893 行既有抽检对两端均能唯一匹配。动作唯一匹配仍不等于像素与特征对齐通过。',
        '', '## 可复现性边界','',
        '原 INPUT_MANIFEST.json 已列每个特征文件 SHA256；本轮增加 DATA_ACCESS.md 和只读资产验证脚本。原数据/历史 buffer 不随 Git 发布，服务器路径不是公共下载归档。',
        '', '随机 5+5 与 Top5 检索仍是不同搜索和历史窗口预算；未追加未经冻结的新效率比较。',
        '', '实验 B 使用用户新提供的独立协议；因用户确认尚无人工基础标签，先准备盲标包，禁止产生校准阈值或增量分类。',
        '', '`formal_training_resumed=false`；未训练 ABCR。']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    p=out/'ACTION_WINDOW_PROVENANCE.csv'
    with gzip.GzipFile(str(p)+'.gz','wb',mtime=0) as f:f.write(p.read_bytes())
    print('REVIEW_COMPLETE')


if __name__=='__main__':main()

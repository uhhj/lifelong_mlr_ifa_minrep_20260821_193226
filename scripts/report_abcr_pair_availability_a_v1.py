"""Mechanical study-A findings; no threshold changes or new measurements."""
import argparse
import datetime
import json
import re
import statistics
from pathlib import Path
from abcr_pair_availability_a_v1.io import read_table,write,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=ap.parse_args()
    root=args.root.resolve();out=root/'reports/abcr_pair_availability_a_v1'
    verification=json.loads((out/'INDEPENDENT_VERIFICATION.json').read_text())
    assert verification['status']=='PASS'
    rows=read_table(out/'summary.csv');batches=read_table(out/'batch_comparison.csv')
    n=read_table(out/'by_old_task.csv')
    assert len(rows)==36 and len(batches)==72
    findings={}
    for threshold in (.7,.8,.9,.95):
        vals=[r for r in rows if float(r['visual_threshold'])==threshold]
        findings[str(threshold)]=dict(
            stage_seed_units=9,units_with_any_candidate=sum(int(r['visual_candidates'])>0 for r in vals),
            descriptive_sum_visual_pairs=sum(int(r['visual_candidates']) for r in vals),
            descriptive_sum_share=sum(int(r['share_pairs']) for r in vals),
            descriptive_sum_bif=sum(int(r['bif_pairs']) for r in vals),
            warning='Sums count overlapping window pairs across seed caches, not independent samples',
            share_anchor_coverage_range=[min(float(r['anchor_share_coverage_retrieved']) for r in vals),max(float(r['anchor_share_coverage_retrieved']) for r in vals)],
            bif_anchor_coverage_range=[min(float(r['anchor_bif_coverage_retrieved']) for r in vals),max(float(r['anchor_bif_coverage_retrieved']) for r in vals)])
    bystage={}
    for task in (7,8,9):
        vals=[r for r in rows if int(r['current_task'])==task and float(r['visual_threshold'])==.7]
        item={}
        for name in ['anchor_share_coverage_retrieved','anchor_bif_coverage_retrieved']:
            v=[float(r[name]) for r in vals];item[name]=dict(min=min(v),max=max(v),median=statistics.median(v))
        for strategy in ('RANDOM_5_PLUS_5','RETRIEVE_TOP5_PER_ANCHOR'):
            v=[float(r['fraction_batches_with_both']) for r in batches if int(r['current_task'])==task and float(r['visual_threshold'])==.7 and r['strategy']==strategy]
            item[strategy]=dict(min=min(v),max=max(v),median=statistics.median(v))
        bystage[str(task)]=item
    idle=[int(r['share_idle_pairs'])/int(r['share_pairs']) for r in rows if float(r['visual_threshold'])==.7]
    findings['share_idle_fraction_at_0.7_range']=[min(idle),max(idle)]
    findings['stage_ranges_at_0.7']=bystage
    findings['candidate_availability']='SPARSE_AND_STAGE_DEPENDENT_AT_FROZEN_THRESHOLDS'
    findings['abcr_training_go']='NOT_ESTABLISHED_REQUIRES_EXPERIMENT_B'
    findings['formal_training_resumed']=False
    write(out/'FINDINGS.json',findings)
    log=(out/'unit_tests.log').read_text();match=re.search(r'Ran (\d+) tests in ([\d.]+)s',log)
    assert match and '\nOK\n' in log and 'FAILED' not in log
    write(out/'UNIT_TEST_REPORT.json',dict(status='PASS',tests=int(match[1]),seconds=float(match[2]),log_sha256=sha(out/'unit_tests.log')))
    def rng(item):
        return '%.1f%%–%.1f%%'%(100*item['min'],100*item['max'])
    report=out/'FINAL_REPORT.md'
    text=report.read_text()
    marker='\n## 实测结论与0.70筛查细节\n'
    text=text.split(marker)[0]
    lines=[marker,'',
        '以下范围是三个seed单元的描述性范围，不是置信区间；seed43/44使用确定性原生缓存重建。',
        '', '|当前任务|Top5 share锚点覆盖|Top5 bif锚点覆盖|随机5+5双类频率|Top5检索双类频率|',
        '|---|---|---|---|---|']
    for task,item in bystage.items():
        lines.append('|%s|%s|%s|%s|%s|'%(task,rng(item['anchor_share_coverage_retrieved']),rng(item['anchor_bif_coverage_retrieved']),rng(item['RANDOM_5_PLUS_5']),rng(item['RETRIEVE_TOP5_PER_ANCHOR'])))
    lines += ['', '### 三个问题的回答','',
        '1. **候选对存在吗？** 0.70下9个阶段/seed单元均检出两类候选。0.80下仅task8有两类候选，task9/seed44另有2个不确定候选；0.90和0.95下9个单元全部为0。不能将此解释为真实跨任务相似状态不存在。',
        '2. **覆盖充足吗？** 当前规则下覆盖稀疏、阶段差异明显。task7尤其缺少bif候选；即使task8较好，Top5 bif覆盖仍只有约10%。当前A数据不足以支持充足而可信的ABCR监督。',
        '3. **能稳定进入batch吗？** 0.70下随机5+5双类同时出现仅0–0.2%；Top5检索提高到上表范围，但没有阶段达到“多数批次都有两类”。检索与随机用到的不同旧窗口数和搜索预算不同，不是等预算收益证明。',
        '', '0.70下 share 中双侧静止占比为 %.2f%%–%.2f%%，因此按脚本静止定义，这批share候选不是主要由双方静止造成。'%(100*min(idle),100*max(idle)),
        '', '### 执行与独立核验','',
        '- 输入全部finite，未静默丢弃文件；8项测试通过。',
        '- 独立检查9个单元的输出hash、全量/旧任务计数守恒、Top5覆盖率、每阈值/策略500次批次记录。',
        '- 对分层审计样本重新CPU加载原始张量，用NumPy float64重算双视角余弦、动作距离、gripper与静止标签；共%d行，通过。人工/物理状态复核尚未进行。'%verification['numeric_rows'],
        '- 所有阶段都未初始化CUDA。没有加载策略、改变buffer内容、训练模型或启动旧队列。',
        '', '### 结论边界','',
        '结论为：**在用户给定的探索性阈值与现有语言条件特征下，跨任务候选信号稀疏，检索有所改善但尚不足以直接支持ABCR训练。** 若继续研究，只能另行进行实验B的状态匹配与人工复核，不能通过下调阈值来把本轮结果改写成通过。',
        '', '`formal_training_resumed = false`','']
    report.write_text(text+'\n'.join(lines),encoding='utf-8')
    s=json.loads((out/'SUMMARY.json').read_text());s['findings']=findings;s['independent_verification']='PASS';s['unit_tests']='8/8 PASS'
    write(out/'SUMMARY.json',s)
    # Export simple scientific plots when the existing environment supports it.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    units=[(seed,task) for seed in (42,43,44) for task in (7,8,9)]
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),sharey=True)
    for ax,key,title in zip(axes,['anchor_share_coverage_retrieved','anchor_bif_coverage_retrieved'],['Share candidate anchor coverage (%)','Bif candidate anchor coverage (%)']):
        data=np.array([[100*float(next(r[key] for r in rows if int(r['seed'])==s and int(r['current_task'])==t and float(r['visual_threshold'])==v)) for s,t in units] for v in [.7,.8,.9,.95]])
        im=ax.imshow(data,vmin=0,vmax=35,cmap='Blues',aspect='auto')
        ax.set_xticks(range(9));ax.set_xticklabels(['s%d/T%d'%u for u in units],rotation=45,ha='right')
        ax.set_yticks(range(4));ax.set_yticklabels(['0.70','0.80','0.90','0.95']);ax.set_title(title)
        for i in range(4):
            for j in range(9):ax.text(j,i,'%.1f'%data[i,j],ha='center',va='center',fontsize=8,color='white' if data[i,j]>20 else 'black')
    axes[0].set_ylabel('Visual cosine threshold');fig.tight_layout();fig.savefig(out/'candidate_coverage.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5));x=np.arange(9)
    for offset,method,label in [(-.18,'RANDOM_5_PLUS_5','Random 5+5'),(.18,'RETRIEVE_TOP5_PER_ANCHOR','Top5 per anchor')]:
        data=[100*float(next(r['fraction_batches_with_both'] for r in batches if int(r['seed'])==s and int(r['current_task'])==t and float(r['visual_threshold'])==.7 and r['strategy']==method)) for s,t in units]
        bars=ax.bar(x+offset,data,width=.36,label=label)
        for bar,value in zip(bars,data):ax.text(bar.get_x()+bar.get_width()/2,value+.4,'%.1f'%value,ha='center',fontsize=8)
    ax.set_xticks(x);ax.set_xticklabels(['s%d/T%d'%u for u in units]);ax.set_ylim(0,48);ax.set_ylabel('Batches with both candidate labels (%)')
    ax.set_title('Visual threshold 0.70; 500 paired simulated batches per unit')
    ax.legend();fig.text(.5,.01,'Retrieval uses a larger old-window/search budget; trajectories unavailable.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(out/'batch_availability.png',dpi=160);plt.close(fig)
    print(json.dumps(findings,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

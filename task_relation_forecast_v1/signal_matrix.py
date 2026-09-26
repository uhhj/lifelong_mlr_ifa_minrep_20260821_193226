"""Display sealed raw signals and existing task-held-out predictions; no refit."""
import csv
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw
from .relation_matrix import font

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'reports/task_relation_forecast_v1'
OUT = SRC/'signal_matrix_v1'
TASKS = (6, 7, 8, 9)
SEEDS = (45, 46, 47)
FAMILIES = ('M_IFA', 'M_A', 'M_LA', 'M_GA', 'M_probe')
LABELS = {'ACTUAL':'真实结果', 'M_IFA':'IFA 视觉＋语言', 'M_A':'动作相容性 A', 'M_LA':'latent＋A', 'M_GA':'GN＋A', 'M_probe':'一步探针（原始值全零）'}
RAW = {'S_visual':'IFA：视觉余弦', 'S_language':'IFA：语言余弦', 'A':'动作相容性：NLL改善', 'S_lat':'潜特征质心余弦', 'S_GN':'GN 子空间重叠', 'R_old':'一步探针：旧任务 NLL变化', 'R_new':'一步探针：新任务 NLL变化'}


def read(name):
    with (SRC/name).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def triple(values, decimals=1):
    return ' / '.join(format(v, '+.%df'%decimals) for v in values)


def table(title, values, decimals=1):
    lines = ['', '## '+title, '', '|源 → 目标|task6|task7|task8|task9|', '|---|---|---|---|---|']
    for s in TASKS:
        cells=[]
        for t in TASKS:
            if s==t: cells.append('—'); continue
            v=values[s,t]
            cells.append(format(statistics.mean(v), '+.%df'%decimals)+'<br>('+triple(v,decimals)+')')
        lines.append('|task%d|%s|'%(s,'|'.join(cells)))
    return lines


def matrix_csv(prefix, values):
    for seed in (None,)+SEEDS:
        rows=[]
        for s in TASKS:
            row={'source':s}
            for t in TASKS:
                row['target%d'%t]='' if s==t else (statistics.mean(values[s,t]) if seed is None else values[s,t][SEEDS.index(seed)])
            rows.append(row)
        write_csv(prefix+('_MEAN' if seed is None else '_seed%d'%seed)+'.csv',rows)


def figure(outcome, panels):
    im=Image.new('RGB',(1720,1330),'#f6f8fc'); d=ImageDraw.Draw(im)
    title='目标迁移收益' if outcome=='TRANSFER_300' else '源任务损伤'
    d.text((35,22),title+'｜真实结果与五类信号预测',font=font(32),fill='#14263b')
    d.text((35,74),'严格留任务 OOF 预测；不重新拟合。行 = 源任务，列 = 目标任务；单位：百分点（pp）。',font=font(20),fill='#42536a')
    d.text((35,108),'大字 = 3 seeds 均值；小字 = seed45 / 46 / 47。预测不是实际成功率，不裁剪超出范围的数值。',font=font(20),fill='#42536a')
    bound=max(1,max(abs(statistics.mean(v)) for _, vals in panels for v in vals.values()))
    for p,(family,vals) in enumerate(panels):
        x0=30+(p%3)*565; y0=205+(p//3)*520; cw=115; ch=100
        d.text((x0,y0-49),LABELS[family],font=font(25),fill='#14263b')
        for j,t in enumerate(TASKS):d.text((x0+96+j*cw,y0-9),'T%d'%t,font=font(18),fill='#14263b')
        for i,s in enumerate(TASKS):
            d.text((x0,y0+37+i*ch),'T%d'%s,font=font(20),fill='#14263b')
            for j,t in enumerate(TASKS):
                x=x0+64+j*cw; y=y0+25+i*ch
                v=0 if s==t else statistics.mean(vals[s,t])
                base=(46,111,174) if v>=0 else (206,110,42)
                color='#e7ecf2' if s==t else tuple(round(249+(c-249)*(.12+.72*min(abs(v)/bound,1))) for c in base)
                d.rounded_rectangle((x,y,x+cw-6,y+ch-6),radius=6,fill=color)
                d.text((x+10,y+17),'—' if s==t else '%+.1f'%v,font=font(25),fill='#16263b')
                if s!=t:
                    text='/'.join('%+.0f'%a for a in vals[s,t])
                    d.text((x+5,y+66),text,font=font(13),fill='#283647')
    d.text((35,1210),'蓝 = 正值；橙 = 负值；所有面板使用同一色标。目标迁移正值有利；源任务损伤正值不利。',font=font(20),fill='#42536a')
    d.text((35,1250),'task6 黄油｜task7 牛奶｜task8 巧克力布丁｜task9 橙汁。五类预测增量的原判定均为 INCONCLUSIVE。',font=font(19),fill='#42536a')
    im.save(OUT/(outcome+'_PREDICTION_MATRICES.png'))


def main():
    OUT.mkdir(exist_ok=True)
    signals=read('SIGNALS.csv'); targets=read('TARGETS.csv'); oof=read('OOF_PREDICTIONS.csv')
    key=lambda r:tuple(int(r[k]) for k in ('seed','source','target'))
    expected={(seed,s,t) for seed in SEEDS for s in TASKS for t in TASKS if s!=t}
    assert len(signals)==len(targets)==36
    assert {key(r) for r in signals}=={key(r) for r in targets}==expected
    target_map={key(r):r for r in targets}
    for r in signals:
        sealed=json.loads((SRC/'signals'/('s%d_%d_%d'%key(r))/'SIGNALS.json').read_text())['scores']
        for k in RAW: assert float(r[k])==float(sealed[k]) and math.isfinite(float(r[k]))
    raw_report=['# 五类关系信号：原始矩阵', '', '每格为三seed等权均值；括号顺序45/46/47。对角线未测量，不填0。', '',
        '五类预测器：M_IFA = B+视觉+语言；M_A = B+A；M_LA = B+S_lat+A+交互；M_GA = B+S_GN+A+交互；M_probe = B+R_old+R_new。B包含目标基座NLL和源模型自身NLL。', '',
        '视觉/语言/latent相似度与GN重叠越大，只表示对应结构更接近，不表示有益。A = 目标基座NLL−源模型在目标上的NLL，正值表示目标动作解释改善。R_old/R_new正值表示一步后NLL变差。', '',
        '本轮原生第一步LR=0，R_old/R_new全零：表示冻结探针没有更新信息，不是安全或无冲突。余弦/重叠为无量纲，A/R为NLL差；禁止跨量纲比较大小。']
    raw_summary=[]
    for field,title in RAW.items():
        vals={(s,t):[float(next(r[field] for r in signals if key(r)==(seed,s,t))) for seed in SEEDS] for s in TASKS for t in TASKS if s!=t}
        matrix_csv(field,vals);raw_report+=table(title,vals,4)
        for (s,t),v in vals.items():raw_summary.append(dict(signal=field,source=s,target=t,mean=statistics.mean(v),seed_sd=statistics.stdev(v),seed45=v[0],seed46=v[1],seed47=v[2]))
    write_csv('RAW_SIGNAL_SUMMARY.csv',raw_summary)
    (OUT/'RAW_SIGNAL_MATRICES.md').write_text('\n'.join(raw_report)+'\n',encoding='utf-8')
    report=['# 五类关系信号：预测结果矩阵', '', '仅重排已封存的严格留任务OOF结果，不使用全数据拟合，不更改原判定。每格为均值及seed45/46/47。单位pp。', '',
        '模型预测与真实观测分开；不能将原始相似度直接转换为共享/冲突标签。所有预测保持原值，不裁剪。相邻格子共享任务与seed，不是独立统计样本。', '',
        'M_probe原始信号全零，其预测来自B基线，逐行与M0相同；图中非零预测不代表影子更新提供了信息。', '',
        '严格留任务：测试目标t时，训练排除source=t或target=t的全部关系。三seed是优化种子，不是独立预训练基座。', '',
        '|预测器|TRANSFER MSE|TRANSFER Spearman|DAMAGE MSE|DAMAGE Spearman|', '|---|---:|---:|---:|---:|']
    rows=[];metric_rows=[]
    metrics=json.loads((SRC/'PREDICTION_METRICS.json').read_text())
    for outcome in ('TRANSFER_300','DAMAGE_300'):
        panels=[]
        for family in ('ACTUAL',)+FAMILIES:
            pred={key(r):r for r in oof if r['mode']=='task' and r['outcome']==outcome and r['family']==family}
            if family!='ACTUAL':assert set(pred)==expected
            vals={}
            for s in TASKS:
                for t in TASKS:
                    if s==t:continue
                    v=[]
                    for seed in SEEDS:
                        k=(seed,s,t);actual=float(target_map[k][outcome])*100
                        if family=='ACTUAL':value=actual
                        else:
                            assert abs(float(pred[k]['observed'])*100-actual)<1e-10
                            value=float(pred[k]['predicted'])*100
                            assert math.isfinite(value)
                            rows.append(dict(seed=seed,source=s,target=t,outcome=outcome,family=family,observed_pp=actual,predicted_pp=value,error_pp=value-actual))
                        v.append(value)
                    vals[s,t]=v
            panels.append((family,vals));matrix_csv(outcome+'_'+family,vals)
            report+=table(outcome+' / '+LABELS[family],vals)
        figure(outcome,panels)
        for family in ('M0',)+FAMILIES:
            rr=[r for r in oof if r['mode']=='task' and r['outcome']==outcome and r['family']==family]
            assert len(rr)==36
            mse=statistics.mean((float(r['predicted'])-float(r['observed']))**2 for r in rr)
            stored=metrics['task/'+outcome+'/'+family]['all']
            assert abs(mse-stored['mse'])<1e-12
            metric_rows.append(dict(outcome=outcome,family=family,mse=mse,mae=stored['mae'],spearman=stored['spearman']))
    insert=[]
    for family in ('M0',)+FAMILIES:
        a,b=[next(r for r in metric_rows if r['family']==family and r['outcome']==o) for o in ('TRANSFER_300','DAMAGE_300')]
        insert.append('|%s|%.6f|%.3f|%.6f|%.3f|'%(family,a['mse'],a['spearman'],b['mse'],b['spearman']))
    report[12:12]=insert+['', '上表MSE沿用原报告的成功率比例平方单位，未乘10000；下方矩阵才转换为百分点。图中小字取整，精确数值见CSV，不以图中舍入后的0判定中性。']
    report+=['', '## 解释边界', '', 'A在源任务损伤预测上有描述性改善，但五类模型的原冻结增量判定均为INCONCLUSIVE。矩阵用于定位错判关系，不能代替原bootstrap、任务折和seed稳定性评估。', '', 'formal_training_resumed=false；未新增训练、rollout、参数搜索或预测器拟合。']
    write_csv('PREDICTIONS_BY_PAIR_SEED.csv',rows);write_csv('PREDICTION_METRICS_RECHECKED.csv',metric_rows)
    assert len(rows)==360
    for r in oof:
        if r['mode']=='task' and r['family']=='M_probe':
            m0=next(x for x in oof if x['mode']=='task' and x['family']=='M0' and x['outcome']==r['outcome'] and key(x)==key(r))
            assert abs(float(r['predicted'])-float(m0['predicted']))<1e-12
    (OUT/'PREDICTION_MATRICES.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    provenance=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),inputs={name:hashlib.sha256((SRC/name).read_bytes()).hexdigest() for name in ('SIGNALS.csv','OOF_PREDICTIONS.csv','TARGETS.csv','PREDICTION_METRICS.json')},raw_signal_cell_checks=36*len(RAW),prediction_rows=360,metric_recomputation='PASS',probe_equals_baseline='PASS',prediction_mode='task-held-out',new_fitting=False,new_experiments=False,formal_training_resumed=False)
    (OUT/'PROVENANCE.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()

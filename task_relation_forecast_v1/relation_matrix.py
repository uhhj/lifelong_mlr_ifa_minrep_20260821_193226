"""Descriptive directed outcome matrices; no new fitting, trials or thresholds."""
import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'reports/task_relation_forecast_v1'
OUT = SOURCE/'relation_matrix_v1'
TASKS = (6, 7, 8, 9)
SEEDS = (45, 46, 47)
METRICS = ('TRANSFER_300','DAMAGE_300','TRANSFER_0','ACQUISITION_DIFF_300')
ZH = {'TRANSFER_300':'目标条件迁移', 'DAMAGE_300':'源任务损伤', 'TRANSFER_0':'零样本迁移', 'ACQUISITION_DIFF_300':'后续学习改善差'}


def csv_write(name, rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def pp(v):return '%+.1f'%v if v else '0.0'


def font(size):
    candidates=['C:/Windows/Fonts/msyh.ttc','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    return ImageFont.truetype(next(p for p in candidates if Path(p).exists()),size)


def draw_matrix(rows, means):
    image=Image.new('RGB',(1620,940),'#f6f8fc');d=ImageDraw.Draw(image)
    d.text((45,24),'任务关系结果矩阵｜真实闭环结果',font=font(34),fill='#16263b')
    d.text((45,77),'行 = 源任务；列 = 目标任务。大字：3 seeds 均值；小字：45 / 46 / 47。单位：百分点（pp）',font=font(21),fill='#42536a')
    for panel,m in enumerate(METRICS[:2]):
        x0=45+panel*795;y0=202;cw=155;ch=135
        title='目标迁移 T：正值 = 源经验帮助目标' if panel==0 else '源任务损伤 D：正值 = 学新后源任务退化'
        d.text((x0,135),title,font=font(25),fill='#16263b')
        for j,t in enumerate(TASKS):d.text((x0+110+j*cw+45,y0-40),'task%d'%t,font=font(21),fill='#16263b')
        for i,s in enumerate(TASKS):
            d.text((x0,y0+i*ch+45),'task%d'%s,font=font(22),fill='#16263b')
            for j,t in enumerate(TASKS):
                x=x0+102+j*cw;y=y0+i*ch
                if s==t:
                    d.rounded_rectangle((x,y,x+cw-8,y+ch-8),radius=8,fill='#e8edf3');d.text((x+57,y+42),'—',font=font(28),fill='#758299');continue
                vals=[r[m+'_pp'] for r in rows if r['source']==s and r['target']==t];v=means[s,t,m]
                benefit=v if panel==0 else -v
                strength=min(abs(benefit)/80,1)
                base=(38,137,115) if benefit>=0 else (197,72,67)
                color=tuple(round(248+(c-248)*(.1+.68*strength)) for c in base)
                d.rounded_rectangle((x,y,x+cw-8,y+ch-8),radius=8,fill=color)
                d.text((x+16,y+28),pp(v),font=font(29),fill='#16263b')
                d.text((x+10,y+84),'/'.join('%+g'%q if q else '0' for q in vals),font=font(15),fill='#283647')
    d.text((45,770),'task6 黄油　　task7 牛奶　　task8 巧克力布丁　　task9 橙汁（均为拾取后放入篮子）',font=font(22),fill='#16263b')
    d.text((45,814),'绿 = 对该指标有利；红 = 不利。单个 seed 每端点20次评估，1次差异 = 5 pp。',font=font(21),fill='#42536a')
    d.text((45,854),'这些是1000步源训练 + 300步目标适应的观测关系，不是显著性结论，也不是永久“共享/冲突”标签。',font=font(21),fill='#42536a')
    image.save(OUT/'TASK_RELATION_MATRIX.png')


def main():
    OUT.mkdir(exist_ok=True)
    raw=list(csv.DictReader((SOURCE/'TARGETS.csv').open(encoding='utf-8-sig')))
    names=json.loads((SOURCE/'DATA_MANIFEST.json').read_text())['tasks']
    assert len(raw)==36
    rows=[];means={};summary=[]
    for r in sorted(raw,key=lambda r:(int(r['source']),int(r['target']),int(r['seed']))):
        seed,s,t=map(lambda k:int(r[k]),('seed','source','target'))
        assert seed in SEEDS and s!=t
        values={k:float(v) for k,v in r.items() if k not in ('seed','source','target')}
        def successes(model,task):
            p=SOURCE/'behavior'/('s%d'%seed)/model/('task%d'%task)/'TRIALS.json'
            trials=json.loads(p.read_text());assert len(trials)==20
            return sum(v['success'] for v in trials)
        expected={
            'TRANSFER_300':(successes('transfer%d_%d'%(s,t),t)-successes('reference%d'%t,t))/20,
            'DAMAGE_300':(successes('source%d'%s,s)-successes('transfer%d_%d'%(s,t),s))/20,
            'TRANSFER_0':(successes('source%d'%s,t)-successes('base',t))/20}
        expected['ACQUISITION_DIFF_300']=expected['TRANSFER_300']-expected['TRANSFER_0']
        assert all(abs(values[k]-v)<1e-12 for k,v in expected.items())
        rows.append(dict(seed=seed,source=s,target=t,source_description=names[str(s)]['language'],target_description=names[str(t)]['language'],**{k+'_pp':round(v*100,8) for k,v in expected.items()},**{k+'_pct':round(v*100,8) for k,v in values.items() if k.startswith('P_')}))
    for s in TASKS:
        for t in TASKS:
            if s==t:continue
            one=dict(source=s,target=t)
            rr=[r for r in rows if r['source']==s and r['target']==t];assert [r['seed'] for r in rr]==list(SEEDS)
            for m in METRICS:
                vals=[r[m+'_pp'] for r in rr];avg=statistics.mean(vals);means[s,t,m]=avg
                for seed,v in zip(SEEDS,vals):one[m+'_s%d_pp'%seed]=v
                one.update({m+'_mean_pp':avg,m+'_seed_sd_pp':statistics.stdev(vals),m+'_min_pp':min(vals),m+'_max_pp':max(vals),m+'_positive_seeds':sum(v>0 for v in vals),m+'_zero_seeds':sum(v==0 for v in vals),m+'_negative_seeds':sum(v<0 for v in vals)})
            summary.append(one)
    csv_write('PAIR_SEED_RESULTS.csv',rows);csv_write('PAIR_SUMMARY.csv',summary)
    for m in METRICS:
        for seed in [None]+list(SEEDS):
            table=[]
            for s in TASKS:
                one={'source':s}
                for t in TASKS:
                    one['target%d'%t]='' if s==t else (means[s,t,m] if seed is None else next(r[m+'_pp'] for r in rows if (r['source'],r['target'],r['seed'])==(s,t,seed)))
                table.append(one)
            csv_write('%s_%s.csv'%(m,'MEAN' if seed is None else 'seed%d'%seed),table)
    lines=['# 任务关系结果矩阵 v1','', '本矩阵整理实际闭环结局，不是关系信号预测矩阵。行是先训练的源任务，列是后适应的目标任务；对角线未测，留空。', '', '单位为成功率百分点（pp），不是相对百分比。每个单元使用3个seed等权平均，并保留seed45/46/47；每个seed的每个模型×任务端点20次配对trial。', '', '|任务|操作对象|原始描述|','|---|---|---|']
    for t,cn in zip(TASKS,['黄油','牛奶','巧克力布丁','橙汁']):lines.append('|task%d|%s|%s|'%(t,cn,names[str(t)]['language']))
    definitions={
        'TRANSFER_300':'T = 目标在源经验→目标300步后的成功率 − 基座→目标300步后的成功率。正值：源经验帮助目标；负值：负迁移。',
        'DAMAGE_300':'D = 源任务在1000步源训练后的成功率 − 学目标300步后源任务成功率。正值：源任务受损；负值：源任务改善。',
        'TRANSFER_0':'T0 = 目标在源训练后、目标适应前的成功率 − 共同基座目标成功率。',
        'ACQUISITION_DIFF_300':'Q = T − T0，即两支后续改善量之差；受起点与地板/天花板影响，不等于内在学习速度。'}
    for m in METRICS:
        lines+=['','## '+ZH[m],'',definitions[m],'','大字为均值，括号内依次为seed45 / seed46 / seed47。','','|源 → 目标|task6|task7|task8|task9|','|---|---:|---:|---:|---:|']
        for s in TASKS:
            cells=[]
            for t in TASKS:
                if s==t:cells.append('—');continue
                vals=[r[m+'_pp'] for r in rows if (r['source'],r['target'])==(s,t)]
                cells.append('**%s** (%s)'%(pp(means[s,t,m]),' / '.join(pp(v) for v in vals)))
            lines.append('|task%d|%s|'%(s,'|'.join(cells)))
    lines+=['','## 12个方向的联合阅读','','T正代表帮目标，D正代表伤源。以下仅按已观察符号描述，零差不是等效性证明。','','|方向|T均值 pp|D均值 pp|T seed符号 +/0/−|D seed符号 +/0/−|','|---|---:|---:|---|---|']
    for q in summary:
        lines.append('|%d→%d|%s|%s|%d/%d/%d|%d/%d/%d|'%(q['source'],q['target'],pp(q['TRANSFER_300_mean_pp']),pp(q['DAMAGE_300_mean_pp']),*[q[m+'_'+k+'_seeds'] for m in METRICS[:2] for k in ['positive','zero','negative']]))
    lines+=['','## 先看绝对成功率，防止地板误读','','|任务|源训练1000步自身成功率：45/46/47|基座→该任务300步参考成功率：45/46/47|','|---|---|---|']
    for t in TASKS:
        a=[];b=[]
        for seed in SEEDS:
            rs=next(r for r in rows if r['seed']==seed and r['source']==t);rt=next(r for r in rows if r['seed']==seed and r['target']==t)
            a.append('%.0f%%'%rs['P_source_source_pct']);b.append('%.0f%%'%rt['P_reference_target_pct'])
        lines.append('|task%d|%s|%s|'%(t,' / '.join(a),' / '.join(b)))
    lines+=['','## 可直接读出的观测模式','',
        '- **9→6：三个seed都帮助目标，同时三个seed都损害源任务。** T=+85/+80/+60 pp，D=+95/+45/+75 pp；是明显的目标获益与源损伤并存，而不是双方共享已证明。',
        '- **6→8：三个seed都负迁移，同时源任务都改善。** T=−20/−15/−80 pp，D=−85/−55/−5 pp；与9→6的折中方向相反。',
        '- **6→7、7→6、8→6：目标迁移均为3/3正向，但源损伤并不一致。** 不能从“帮新任务”推出“不伤旧任务”。',
        '- **9→7、8→9等关系有seed方向反转。** 必须看逐seed数值，不能只给平均值永久标签。',
        '- **所有以task9为源的分支，适应其他目标后task9均为0/20成功。** 它们的D恰好等于源起点成功率（95%/45%/75%）；这是强源退化，不是目标负迁移的同义词。',
        '- task8源先验自身成功率只有0%/15%/20%，task6有一个seed为0%；源起点地板下的D≈0不证明保留良好。',
        '', '## 证据边界与复现','',
        '数值直接从已封存TARGETS.csv读取，并逐项用原始TRIALS.json成功数重新校验。没有新增训练、rollout、关系模型、阈值或标签门禁；没有重跑bootstrap或对12个方向做事后显著性筛选。',
        '矩阵中的正/负和3/3一致都是描述性观察；每seed单次trial对应5 pp。3个优化seed共享同一预训练基座，12个有向关系仅来自4个任务；不能外推为所有任务的稳定关系。',
        '固定条件为无回放诊断、源训练1000步、目标适应300步；不是完整lifelong矩阵，也不是先前sample-QP结果。',
        '逐seed绝对起点/终点见PAIR_SEED_RESULTS.csv，均值/样本标准差/范围/符号计数见PAIR_SUMMARY.csv。标准差描述seed波动，不充当置信区间。',
        '', 'formal_training_resumed=false']
    (OUT/'TASK_RELATION_MATRIX.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    draw_matrix(rows,means)
    info=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_targets_sha256=hashlib.sha256((SOURCE/'TARGETS.csv').read_bytes()).hexdigest(),directed_pairs=12,seeds=list(SEEDS),rows=36,raw_trial_recomputation='PASS',unit='percentage points',mean='equal seed weighting',diagonal='NOT_MEASURED',new_experiments=False,formal_training_resumed=False)
    (OUT/'MATRIX_PROVENANCE.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()

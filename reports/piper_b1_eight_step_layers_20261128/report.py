"""Emit a Chinese report and per-episode results from the audited offline outputs."""
from assemble import *
from viewer import html,LABEL
import sys
NAMES={'field':'现场版','candidate':'离线候选版'}
GROUP={'both':'合计','regression':'11 月 23 日，32 条','archive':'11 月 20 日，20 条'}
def f(x):return '—' if x is None else f'{x:.2f}'
def sg(x):return '—' if x is None else f'{0 if abs(x)<.005 else x:+.2f}'
def typ(r):return LABEL[r['phase']]
def cell(v,e,p):
    if v is None:return '—'
    return '/'.join(sg(a) for a in v)+f'（{f(size(v))} mm，{f(heading(v))}°'+(f'，投影 {sg(projection(e,v))}' if p=='correct' else '')+'）'
def light_stats(s):
    return dict(counts=s['counts'],regions=s['regions'],sensitivity=s['sensitivity'],previous_zero=s['previous_zero'],exploratory_micro=s['micro'],file_sha256=s['file_sha256'],
        groups={g:{'phases':{p:{n:s['primary'][g][p][n] for n in VERSIONS} for p in PHASES[:2]},
                   'completed_correction':{n:compact_from(s['paired'][g][n]['correct']) for n in VERSIONS},
                   'small_correction':{n:compact_from(s['small'][g][n]) for n in VERSIONS},
                   'image_and_grasp_agree':{n:compact_from(s['agreement'][g]['correct'][n]) for n in VERSIONS},
                   'episode_representatives':{n:compact_from(s['representatives'][g]['correct'][n]) for n in VERSIONS}} for g in GROUP})
def compact_from(a):
    return dict(windows=a['windows'],episodes=a['episodes'],stages={k:{j:z[j] for j in ('n','xy','z','projection','negative','negligible','passes')} for k,z in a['stages'].items()},
                transitions=a['transitions'])
def local_representatives(byep,meta,reps):
    parts=['# B1 的 52 条示范逐条对照','',
        '每格为基座 ΔX/ΔY/ΔZ（mm），括号内为水平幅度、从 +X 朝 +Y 的方向角及纠偏窗沿专家方向的投影。角度方向只在横移 ≥0.5 mm 时计算。每条每类从非重叠窗优先，选专家位移最接近该条该类中位数的一窗。专家未出现某类时保留空缺。','']
    for ep in sorted(byep):
        e=meta[ep]
        parts.extend([f"## {ep}",'',f"摆位{e['region']}。滑动窗含横向纠偏 {e['all'].get('correct',0)} 个，已对准下降 {e['all'].get('descend',0)} 个，其他 {e['all'].get('other',0)} 个。",'',
         '| 阶段／起始帧 | 专家实际 | 专家指令 | 原始 ODE | 现场版视觉计划 | 现场版滚动最终目标 | 候选版视觉计划 | 候选版滚动最终目标 |','|---|---|---|---|---|---|---|---|'])
        for r in reps[ep].values():
            vs=values(r,'field');cs=values(r,'candidate');e=r['expert'];p=r['phase']
            data=[vs['expert'],vs['command'],vs['raw'],vs['plan'],vs['final'],cs['plan'],cs['final']]
            parts.append('| '+ ' | '.join([f"{typ(r)}／{r['t']}"]+[cell(v,e,p) for v in data])+' |')
        for r in reps[ep].values():
            if r['phase']=='correct':parts.extend(['','同窗判定。现场版 '+attribution(r,'field')+'。候选版 '+attribution(r,'candidate')+'。'])
        parts.append('')
    return '\n'.join(parts)
def public_report(s):
    count=s['counts']['both'];a=s['primary']['both'];small=s['small']['both'];n=count['grid_phase']['correct'];nd=count['grid_phase']['descend']
    text=['# B1 抓取前八步的专家位移、ODE 预测与执行台目标','',
          f"52 条已有成功示范，涵盖原 16 对摆位与跨批次 20 条。共 {count['windows']} 个合格滑窗。以首次专家目标开度降至 55 mm 或以下为锚点向前每八帧取一窗，得到 {count['grid']} 个互不重叠窗。横向纠偏 {n} 个，已对准下降 {nd} 个，其余 {count['grid_phase']['other']} 个。滑窗与代表窗用于复查，主统计用互不重叠窗。",'',
          '## 分层结论','',
          '表中横向纠偏使用沿专家实际方向的有符号投影（mm）。下降窗分别报告 XY 幅度和有符号 ΔZ。每格为窗口中位数，中断的窗口没有第八步目标。','',
          '| 阶段 | 专家实际 | 专家第八条指令 | 原始 ODE | 现场版视觉计划 | 现场版滚动最终目标 | 候选版视觉计划 | 候选版滚动最终目标 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for p,title,metric in [('correct','纠偏，沿专家方向','projection'),('descend','对准后，水平幅度','xy'),('descend','对准后，垂直位移','z')]:
        series=[a[p][k]['stages'][j] for k,j in [('field','expert'),('field','command'),('field','raw'),('field','plan'),('field','final'),('candidate','plan'),('candidate','final')]]
        text.append('| '+' | '.join([title]+[f(x[metric])+(f"（{x['n']} 窗）" if i in [4,6] else '') for i,x in enumerate(series)])+' |')
    raw=a['correct']['field']['stages']['raw'];exp=a['correct']['field']['stages']['expert']
    text+=['',f"纠偏时专家水平位移中位数 {f(exp['xy'])} mm，ODE 水平幅度 {f(raw['xy'])} mm。ODE 反向 {raw['negative']}/{n}，水平幅度小于 0.5 mm 的有 {raw['negligible']} 个，沿正确方向超过专家两倍的有 {raw['along_above_twice']} 个。从上一条专家指令为起点另比 ODE 与监督指令，夹角不超过 60° 的有 {a['correct']['field']['raw_command_cos60']}/{a['correct']['field']['raw_command_direction_defined']} 个可判方向的窗口。ODE 与专家指令的横向端点误差中位数 {f(a['correct']['field']['raw_command_xy_error'])} mm，专家指令与实测端点误差 {f(a['correct']['field']['expert_command_xy_error'])} mm。",'',
          '横向保留判据为水平幅度至少 0.5 mm、与专家夹角至多 60°、沿专家方向走到专家实际幅度的一半。该判据只衡量纠偏方向和最低幅度，过冲另计。以下转移均在同一窗口内匹配。','',
          '| 执行台 | 原始达标 | 原始达标但视觉计划丢失 | 原始未达标但视觉计划补回 | 视觉计划横移 <0.5 mm | 滚动最终达标／有终点 | 未走满八步 |','|---|---:|---:|---:|---:|---:|---:|']
    for name in VERSIONS:
        z=a['correct'][name];stats=z['stages'];tr=z['transitions']['counts'];fin=stats['final']
        text.append(f"| {NAMES[name]} | {raw['passes']}/{n} | {tr.get('raw_good_plan_weak',0)} | {tr.get('raw_weak_plan_good',0)} | {stats['plan']['negligible']} | {fin['passes']}/{fin['n']} | {n-fin['n']} |")
    mix=a['correct']['candidate']['transitions']['source_of_plan_loss']
    text+=['',f"在候选版把原始达标动作改为未达标的窗口中，{mix.get('no_target_hold',0)} 个发生于无可靠目标时保持 XY，{mix.get('wrist_deadband',0)} 个发生于腕部估计落入误差带，{mix.get('wrist_guidance',0)} 个发生于腕部引导，{mix.get('planar_correction',0)} 个发生于其余平面纠偏。方向误差在首个 ODE 输出中已经可见，后续视觉改写又进一步削弱部分正确动作。观察输入到原始 ODE 输出这一段同时包含视觉预处理与已训练策略，本次对照不能进一步归因到其中某一项。"]
    desc=a['descend']['field']['stages']
    text+=['',f"在已对准下降窗，专家指令有 {desc['command']['z_up_or_level']}/{nd} 个持平或向上，原始 ODE 则有 {desc['raw']['z_up_or_level']}/{nd} 个持平或向上。两版平面纠偏逐窗保留原始 ODE 的第八步高度。",'','## 真正的小幅横向修正','',
       '| 执行台 | 专家 3–8 mm 的窗 | 专家幅度 | 原始投影 | 视觉计划投影 | 滚动最终投影／有终点 | 计划中近零 |','|---|---:|---:|---:|---:|---:|---:|']
    for name in VERSIONS:
        q=small[name];z=q['stages'];text.append(f"| {NAMES[name]} | {q['windows']} | {f(z['expert']['xy'])} | {sg(z['raw']['projection'])} | {sg(z['plan']['projection'])} | {sg(z['final']['projection'])}／{z['final']['n']} | {z['plan']['negligible']} |")
    ss=small['field']['stages'];cc=small['candidate']['transitions'];ff=small['field']['transitions']
    text+=['',f"小幅纠偏窗中，原始 ODE 的水平幅度中位数为 {f(ss['raw']['xy'])} mm，{ss['raw']['negative']}/{small['field']['windows']} 个逆着专家方向。原始已达标的 {ss['raw']['passes']} 个窗中，现场版视觉改写后失去 {ff['counts'].get('raw_good_plan_weak',0)} 个，候选版失去 {cc['counts'].get('raw_good_plan_weak',0)} 个。",'','## 完成八步的同一批窗口','',
       '此表同时剔除中断窗，从而可以逐列比较。固定计划限速后是假定八步都沿首帧计划走的对照。滚动末帧请求与最终目标之差反映最后一次限幅，前几步的限幅也可能触发重新预测。','',
       '| 纠偏窗 | 有终点 | 专家投影 | 原始 | 视觉计划 | 固定计划限速后（可算数） | 滚动末帧请求 | 滚动最终目标 | 滚动重算计划次数中位数 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in VERSIONS:
        q=s['paired']['both'][name]['correct'];z=q['stages'];text.append(f"| {NAMES[name]} | {q['windows']} | {f(z['expert']['projection'])} | {sg(z['raw']['projection'])} | {sg(z['plan']['projection'])} | {sg(z['bound']['projection'])}（{z['bound']['n']}） | {sg(z['requested']['projection'])} | {sg(z['final']['projection'])} | {f(q['rolling_plans'])} |")
    text+=['','若改用第八条指令所看到的最新专家反馈为起点，最终目标沿专家纠偏方向的投影中位数为 '+ '、'.join(NAMES[n]+' '+sg(s['paired']['both'][n]['correct']['final_vs_last_observation_along'])+' mm' for n in VERSIONS)+'。这是目标相对最新反馈的超前或落后量。','',
       '接近时另查了小于 3 mm 的水平微调。要求非重叠窗水平位移在 1 至 3 mm 之间，按抓取点与图像标注分别算，均从 12 至 60 mm 外起步、接近至少 1 mm 且方向余弦至少 0.8。两种参照均支持的有 '+str(len(s['micro']))+' 窗，逐窗结果保存在 aggregate.json 的 exploratory_micro。','',
       '## 按批次、区域与参照复核','',
        '| 批次 | 区域 | 纠偏窗 | 专家 mm | ODE 投影 mm | 现场最终投影 mm／窗数 | 候选最终投影 mm／窗数 |','|---|---|---:|---:|---:|---:|---:|']
    for g in ('regression','archive'):
        for region,rr in s['regions'][g].items():
            z=rr['correct']['field']['stages'];c=rr['correct']['candidate']['stages']['final']
            text.append(f"| {GROUP[g]} | {region} | {z['expert'][0]} | {f(z['expert'][1])} | {sg(z['raw'][2])} | {sg(z['final'][2])}／{z['final'][0]} | {sg(c[2])}／{c[0]} |")
    agree=s['agreement']['both']['correct']['field']
    text+=['',f"按成功抓取反馈得到的参考点与分批校准的俯视标注之间，中位距离为 {f(count['ref_median'])} mm，最大 {f(count['ref_max'])} mm。两种参照都判断为纠偏的有 {agree['windows']}/{n} 个。在这部分，专家、ODE、现场视觉计划、候选视觉计划的投影分别为 "+'、'.join(f(x) for x in [agree['stages']['expert']['projection'],agree['stages']['raw']['projection'],agree['stages']['plan']['projection'],s['agreement']['both']['correct']['candidate']['stages']['plan']['projection']])+' mm。两种参照都判断为对准下降的有 '+str(s['agreement']['both']['descend']['field']['windows'])+f'/{nd} 个。','',
        '上轮跨批次 12 个首帧八步计划水平位移为零的锚点，这次用专家实际接近动作回看，分组为 '+ '、'.join(LABEL[p]+str(sum(r['phase']==p for r in s['previous_zero']))+' 个' for p in PHASES)+'。','',
        '## 对照口径','',
        '- 机器人基座坐标系，工具中心使用同一套六轴正运动学与指尖偏移。所有端点减去起始时刻的专家实测夹口位置。八条预测指令覆盖 t 到 t+7，实测位移取反馈 t 到 t+8。额外保存 t+7 反馈、同一终点的专家指令及初始指令与反馈的偏差。',
        '- 抓取参考点取该条专家夹爪首次连续三帧反馈开度至多 49 mm 时的起始夹口中心，是成功抓取位置的代理。横向纠偏要求初始距参考点至多 60 mm、横移至少 3 mm、距离缩短至少 2 mm、向参考点的方向余弦至少 0.7。下降允许范围取 12 mm 与指间有效间隙中较小者，间隙按（开度 − 35 mm）/2 − 2 mm 计算并以零为下限。起、终点均在范围内，横移至多 2 mm，下降至少 3 mm 且为横移的两倍。保留同时横移下降的纠偏窗。其余窗单列。',
        '- 检查八帧历史及前后反馈连续有效，逐帧间隔不得超过 70 ms，八条专家目标开度均大于 55 mm。各窗口先以专家历史与上条专家指令为起点重算。逐帧回放使用当时的图像和专家反馈，首帧生成新计划，随后按执行台原有时钟、上次发送确认、限速、跟踪保护、视觉与夹爪状态更新。相机采样间隔代替现场未采集的发送间隔。专家仍在接近作为每个窗口的起始阶段。只处理代码中的命令和反馈路径，不模拟机械臂响应、USB 调度或真实接触。',
        '- 原始 ODE 的权重指纹为 2e14ef23，现场接口指纹为 27fca3b2，上一轮离线候选为 89cd61aa。两种执行台复算出的原始 ODE 端点一致。52 条的旧锚点逐条与上轮存档核对，最大误差 '+f(count['max_anchor_error'])+' mm。轨迹、逐帧预测和模型文件保留在实验工作区。',
        '- 所有明确分组窗口和每条的收爪前 18 帧锚点均回放八次。目标保护或逆解重建中断均按没有终点计，不计零位移或达标。分区、独立图像参照、标签阈值变化、七间隔敏感性与每条示范代表窗的统计见轻量结果文件。']
    return '\n'.join(text)+'\n'
def main():
    byep,groups,meta,rep,hashes=gather();s=summary(groups,meta,rep,hashes)
    write(here/'full_summary.json',s)
    write(here/'aggregate.json',light_stats(s))
    write(here/'episode_outcomes.json',{ep:dict(**m,representatives={p:{'start_frame':r['t'],'image_label':r['image_phase'],'field_completed':values(r,'field')['final'] is not None,'candidate_completed':values(r,'candidate')['final'] is not None}
                    for p,r in rep[ep].items()}) for ep,m in meta.items()})
    (here/'B1_52条逐条对照.md').write_text(local_representatives(byep,meta,rep),encoding='utf-8')
    html(here/'B1_八步逐窗对照.html',byep,meta,rep)
    (here/'README.md').write_text(public_report(s),encoding='utf-8')
    print('verified',len(byep),sum(map(len,byep.values())),'windows;',sum(len(z) for z in rep.values()),'representatives')
if __name__=='__main__':main()

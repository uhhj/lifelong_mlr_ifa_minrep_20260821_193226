"""Render the reviewed offline report from compact frame masks and episode outcomes."""
from pathlib import Path
from statistics import median
from collections import Counter
import json,sys
from summarize import summarize,read
def render(p):
    s=summarize(p);diag=read(p/'gate_frames.json');identity=read(p/'identity.json')
    controls=read(p/'controls.json');long=read(p/'continuous_archive.json');fixed=read(p/'archive.json')['episodes']
    labels=read(p/'labels.json');after=read(p/'after_gates.json')
    def mm(pair):return ' → '.join('—' if v is None else ('0.00' if abs(v)<.005 else f'{v:+.2f}') for v in pair)
    def counts(c):return '/'.join(str(c.get(k,0)) for k in ('aligned','outside','uncertain','miss'))
    def event_state(evt):
        if not evt:return '漏收'
        k='outside' if evt['delta_z_to_expert_mm']>35 else labels[evt['token']]
        return {'aligned':'已发，画面支持','uncertain':'已发，深度不确定','outside':'已发，误收'}[k]+f"（{evt['relative_to_expert']:+d} 帧）"
    fit=read(p/'fit_meta.json');early=read(p/'early_swaps.json')
    qualified=[r for r in early if r['initial_error_px'] is not None and r['initial_error_px']<=12 and r['distance_px']>24]
    a=s['archive']['regions']['全部'];positive=a['positive_continuous'];miss=[c.get('miss',0) for c in a['closure']]
    out=['# B1 遮挡与漏收逐帧回归','',
         f"跨批次 20 条的窗口漏收从 {miss[0]} 条降到 {miss[1]} 条。按整条录像连续回放，八步自然接近量中位数为 {mm(a['continuous'])} mm，真正靠近木块的条数为 {positive[0]} → {positive[1]}。本轮结果支持腕部边缘是漏收的重要来源，尚不足以支持将整版用于实机。",'',
         '本轮对照为相对位置与腕部边缘版 793bb192589d，本轮离线候选为 '+identity['profile'][:12]+'。专家首次将目标开度从 55 mm 以上降到 55 mm 或以下记为第 0 帧。所有运行只读同步归档或使用虚拟设备。','',
         '## 专家收爪时卡在哪里','',
         '15 条在第 0 帧都有收爪意图，15 条都缺少当前可用边缘。旧算法要求前面至少两个完整木面框，12 条当时缓存里没有框，另 3 条只有一个。13 条连当帧的初始木面框都没找到。没有边缘框时，尺寸与夹口位置无读数。按独立条件，第 0 帧有 10 条只确认边缘失败，3 条同时超出俯视范围，1 条同时超出高度，1 条同时超出俯视范围与高度。','',
         '每条回看专家起收前后各八帧，共 255 帧。226 帧没有边缘。其余 29 帧里，13 帧木面宽度不足，23 帧横向位置超出夹口容差，2 帧没有到达夹持深度的图像范围。同帧可同时卡多项。表中的通过帧数分母为 17；尺寸与夹口只在边缘可测的帧上计通过。逐帧布尔值见 gate_frames.json，从左到右对应 −8 到 +8，问号表示缺测。','',
         '| 录像时间 | 第 0 帧另有阻断 | 有边缘 | 尺寸通过 | 夹口横向通过 | 夹口深度通过 | 新版回放 |','|---|---|---:|---:|---:|---:|---|']
    for e,r in diag.items():
        k=r['masks'];fail=[name for key,name in [('coarse','俯视'),('height','高度')] if k[key][8]=='0']
        out.append(f"| {e[9:15]} | {'、'.join(fail) or '无'} | {k['edge'].count('1')} | {k['size'].count('1')} | {k['jaw_x'].count('1')} | {k['jaw_depth'].count('1')} | {event_state(fixed[e]['close_events'][1])} |")
    assert set(after)=={e for e in diag if not fixed[e]['close_events'][1]}
    out+=['',f"其中 {15-len(after)} 条恢复触发，余下 {len(after)} 条的第 0 帧重新检查。仍缺少三边木面读数 {sum(d['lines'][8]=='0' for d in after.values())} 条，已测得三边但木面宽度不足 {sum(d['size'][8]=='0' for d in after.values())} 条，已测得三边但夹口横向未通过 {sum(d['jaw_x'][8]=='0' for d in after.values())} 条，高度超限 {sum(d['height'][8]=='0' for d in after.values())} 条，俯视粗对准或双相机一致性未通过 {sum(d['coarse'][8]=='0' for d in after.values())} 条。条件可能重合，逐帧记录见 after_gates.json。",'','## 本轮改动','',
          '腕部在原门控外直接查找木面的上边和左右两条侧边，并测量边界内外的颜色差。新路径连续三帧定位相容后，还要满足实际开度对应的横向余量、桌面夹持深度、木面尺度、专家高度上界与策略收爪意图。边界延伸到画面外时只使用可见部分。原有腕部完整木面与短期边缘路径仍可触发。','',
          '俯视要连续三帧看见相容的方形木面才能建立或改换定位。出现无边缘支撑的热图跳点时，最多保留已确认位置 2 秒；会话、视场或采样连续性改变就清空证据。腕部定位由各批另外的训练示范拟合，结合当前机械臂姿态估计桌面偏移。腕部三次估计一致时可独立指导小步接近，并扣除留一示范误差作为修正余量。仅有腕部时还要得到俯视与腕部在 12 mm 内的一致意见且俯视位置距夹口不超过 20 mm，才允许进入原收爪判断。缺乏可靠目标时保持八步水平位置。所有方案沿用 ODE 高度、姿态、夹爪开度预测与 STOP。','',
          f"腕部估计使用较早批次 {fit[0]['episodes']} 条训练示范的 {fit[0]['observations']} 个观测、较新批次 {fit[1]['episodes']} 条的 {fit[1]['observations']} 个观测。按整条示范留一，工具相机平面偏移误差中位数分别为 {fit[0]['loo_median_mm']:.2f} 与 {fit[1]['loo_median_mm']:.2f} mm。这两个数作为腕部单独指导时扣除的修正余量。",'',
          '## 八步接近量与收爪结果','',
          '短历史沿用原协议，从评分帧之前预热至多 16 帧。连续回放从各录像起点输入，直到同一专家收爪前第 18 帧，再比较第八步。自然接近量为到参考木块的起点距离减去八步终点距离，正值表示靠近。原 32 条使用既定人工俯视标注和桌面换算，跨批次 20 条用专家抓住时的反馈夹口中心做代理。换位响应只交换末八帧俯视图，衡量第八步终点沿木块换位方向的变化。数字先列旧版，再列新版。','',
          '每条只判读首条虚拟 35 mm 收爪指令。四个收爪数依次为画面支持可夹、可见偏出或高度高于本条专家动作超过 35 mm、画面不确定、截至专家起收后八帧未发指令。最后一项记作窗口漏收。','']
    for g,title in [('regression','原 16 对，共 32 条'),('archive','跨批次 20 条，其中 9 对有匹配换位')]:
        out+=['### '+title,'','| 区域 | 条数 | 短历史自然接近 mm | 连续自然接近 mm | 连续靠近条数 | 单向换位响应 mm | 旧版收爪四项 | 新版收爪四项 |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for region,r in s[g]['regions'].items():
            out.append(f"| {region} | {r['n']} | {mm(r['fixed'])} | {mm(r['continuous'])} | {' → '.join(map(str,r['positive_continuous']))} | {mm(r['swapped'])} | {counts(r['closure'][0])} | {counts(r['closure'][1])} |")
        agg=s[g]['regions']['全部']
        out+=['',f"对内两方向平均后，成对换位响应中位数为 {mm(s[g]['paired'])} mm。短历史下真正靠近的条数为 {agg['positive_fixed'][0]} → {agg['positive_fixed'][1]}，绝对变化不超过 0.001 mm 的为 {agg['unchanged_fixed'][0]} → {agg['unchanged_fixed'][1]}。连续回放中变化不超过该精度的条数为 {agg['unchanged_continuous'][0]} → {agg['unchanged_continuous'][1]}。",'']
    out+=['## 三处已知遮挡跳点','',
        '| 归档时间 | 原始俯视误差 px | 新俯视锚点误差 px | 当帧腕部指导 | 连续回放八步接近 mm |','|---|---:|---:|---|---:|']
    for e,r in sorted(long.items()):
        if r['raw_error_px']<45:continue
        out.append(f"| {e[9:15]} | {r['raw_error_px']:.2f} | {r['used_error_px'] if r['used_error_px'] is not None else '无锚点'} | {'有' if r['tracking']['wrist'] else '无'} | {mm(r['natural'])} |")
    zeros=Counter(r['reason'] for r in read(p/'zero_motion.json').values())
    out+=['',f"跨批次连续回放中，{zeros['wrist_inside_error_allowance']} 条的腕部估计偏差落在训练示范的误差余量内，规划的八步水平位移为零。另有 {zeros['no_reliable_target']} 条因当时没有可靠定位而保持水平位置。",'']
    jumps=sum(r['tracking']['jump_ticks'] for r in long.values());wrist=sum(r['tracking']['wrist_ticks'] for r in long.values());held=sum(r['tracking']['correct_ticks'] for r in long.values());empty=sum(r['tracking']['withheld_ticks'] for r in long.values())
    out+=[f'复核从录像起点至评分帧的全部输入，原始热图与静止木块人工标注相差大于 45 px 的共有 {jumps} 帧。其中 {wrist} 帧采用独立腕部估计，{held} 帧使用距人工中心不超过 20 px 的俯视锚点，{empty} 帧缺少可靠目标并停止水平纠偏。','',
          f"另用跨批次实际录像各自最前面 12 帧做可见性检查，在原 9 对的两个方向直接切换画面。在 18 个方向中，{sum(r['reacquired_frame'] is not None for r in early)} 个切换后出现过距新木块 12 px 以内的检测。部分新、旧位置的 12 px 容差区重叠。原位置先成功定位、两位置相隔大于 24 px 的仅有 {len(qualified)} 个方向，其中 {sum(r['reacquired_frame'] is not None for r in qualified)} 个在切换后 12 帧内重新定位，尚不能确认大幅换位时能可靠恢复跟踪。",'',
          f"三轮实际失败录像的虚拟收爪指令数依次为 {', '.join(str(len(r['versions'][1]['simulated_width_goals'])) for r in controls['field'])}。虚拟设备检查 {len(controls['cases'])} 项，心跳为 {controls['heartbeat_ns']/1e6:.0f} ms，预测有效期 {controls['freshness_ns']/1e6:.0f} ms。水平八步回放校验了高度、末端方向和开度。",'',
          '两组均为以前使用过的回归档案，跨批次样本集中在右侧。固定反馈回放只能验证预测与放行条件，无法给出实机抓取成功率。复算区域表运行 `python summarize.py`。诊断用 diagnose.py，训练侧标定用 measure.py 与 fit_wrist.py。build.py 仅在独立离线目录生成候选，score.py 与 audit.py 分别计算固定窗口、连续历史和保护复核。']
    return '\n'.join(out)+'\n'
if __name__=='__main__':print(render(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent),end='')

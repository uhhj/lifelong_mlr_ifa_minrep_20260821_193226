"""Render the offline B1 comparison from reviewed episode outcome records."""
from collections import Counter
from pathlib import Path
from statistics import median
import json,sys
from summarize import summary

def report(root):
    read=lambda name:json.loads((root/name).read_bytes())
    new=summary(root)
    geom=new['registration']
    au=read('audit.json');ep=read('endpoints.json')
    m=read('misses.json');first=read('intermediate.json')
    ids=read('identity.json')
    def mm(values):return ' / '.join('—' if a is None else f'{a:+.2f}' for a in values)
    def counts(c):return '/'.join(str(c.get(k,0)) for k in ('aligned','outside','uncertain','miss'))
    out=['# B1 跨批次俯视定位与腕部收爪回归','',
        '跨批次换位响应有所提高，八步自然接近量中位数仍为负。两版使用相同 ODE 权重、双相机原图、八步时序与逆运动学。对照为上一轮相对位置候选 6e92f46e，本轮候选为 '+ids['profile'][:12]+'。候选仅保存在离线目录。','',
        '## 俯视误差来自哪里','',
        f"11 月 20 日固定 20 条回放，使用各条成功抓住木块时由关节反馈算出的夹口中心作为桌面位置参考。该参考与同条下降最低点的平面差中位数为 {geom['contact_ref_difference_mm']:.2f} mm。人工标注初段可见木块的上表面中心，和模型前八帧估计相差 {geom['early_px']:.2f} px。接近末段与静止木块标注的差异中位数为 {geom['late_static_px']:.2f} px，其中三条在遮挡时偏差超过 80 px。",'',
        '| 提供给换算的坐标 | 11 月 23 日原换算 | 本批训练示范换算 |','|---|---:|---:|',
        f"| 人工木块中心 | {geom['manual_runtime_mm']:.2f} mm | {geom['old_map_mm']:.2f} mm |",
        f"| 接近末段八帧检测中心 | {geom['runtime_old_mm']:.2f} mm | {geom['old_px_new_map_mm']:.2f} mm |",'',
        '40 条可标注的另外一组成功示范用于拟合换算，与被评分的 20 条没有交集。仅依赖上方背景的特征来辨认两种已知视场，回放时没有提供批次或日期。以上四项均为同一批 20 条的中位数。套用 11 月 23 日换算时，即使手工标注，机器人基座坐标的误差中位数在 X 轴为 +15.4 mm、Y 轴为 +62.5 mm。档案可以确认换算失配，无法进一步区分相机移动、底座与桌面的相对位姿变化或历史标定差异。初步用紫色纸片对齐时有两条误匹配超过 140 px，最终表统一使用抓取前静止木块的人工中心。','',
        '## 收爪如何判定','',
        '在二至五次近期可见木面的位置上估计当前边缘应出现在哪里。当前帧要同时支持木面的上边和至少一条侧边，边框内颜色符合木面，且木面落在实际开度对应的夹持区。离上次可信观测超过 100 ms 就清空，最长使用过去 250 ms 的证据。保留原有完整木面几何判定路径、ODE 收爪意图、俯视 20 mm 粗对准和连续三帧确认。','',
        f"仅增加时间和边缘证据的中间版在跨批次发出 {first['archive']['fired']} 次虚拟收爪，最早比专家提前 {-first['archive']['earliest_frame']} 帧。最终版增加按各自训练示范专家开始收爪时拟合的高度上界（第 95 百分位），边缘路径同时检查木面像素宽度下界（第 10 百分位）。两种约束只控制夹爪放行，保持模型的六轴下压轨迹。当前深度范围只覆盖两种归档桌面配置。",'',
        '较早批次的高度上界为机器人基座系 9.44 mm（50 条），能清楚读出夹口内木面的 12 条给出宽度下界 144.2 px。较新批次对应为 5.46 mm（48 条）和 138.0 px（38 条）。这些值仅用于匹配归档里的夹持高度与图像尺度。','',
        '## 八步位移与收爪','',
        '自然接近量为起点到本条目标的距离减去第八步终点到目标的距离。换位响应用同一起姿、同一腕部和反馈，只交换最后八张俯视图，测第八步终点变化在木块换位方向上的投影。正值分别表示靠近目标、随换位向正确方向偏转。区域沿用上一轮按训练摆位中位数的划分。表中数值为中位数，顺序均为改前 / 改后。','',
        '收爪按每条的首次虚拟 35 mm 指令判读。四个计数依次为腕部图像支持可夹、明显偏出或高出本条专家起收位置超过一整个木块（35 mm）、投影或深度不可判、截至专家开始收爪后八帧仍未发令。最后一项记作协议窗口漏收。没有指令就没有可判读的误收事件。物理接触和抓起结果无法从固定反馈重放得到。','']
    for group,caption in [('regression','原 16 对，32 条'),('archive','跨批次 20 条，原 9 对')]:
        g=new[group]
        out+=['### '+caption,'','| 区域 | 自然条 / 换位方向数 | 自然接近量 mm | 换位响应 mm | 改前收爪四项 | 改后收爪四项 |','|---|---:|---:|---:|---:|---:|']
        for zone,r in g['regions'].items():
            out.append(f"| {zone} | {r['episodes']} / {r['directions']} | {mm(r['natural_mm'])} | {mm(r['swap_mm'])} | {counts(r['closes'][0])} | {counts(r['closes'][1])} |")
        out+=['',f"先在每一对内平均两个方向，再取所有对的中位数，改前 / 改后为 {mm(g['paired_mm'])} mm，正确方向的配对数为 {g['positive_pairs'][0]} / {g['positive_pairs'][1]}（分母 {g['pairs']}）。",'']
    err=sorted(au['registration'],key=lambda x:x['late_static_px'],reverse=True)[:3]
    out+=['## 残余误差与保护复核','',
        '俯视末段三处最大的像素偏差约为 '+ '、'.join(f"{r['late_static_px']:.0f}" for r in err)+' px，夹爪和前臂覆盖了原木块位置。即使套用正确换算，这三条的桌面定位仍分别偏离专家参考 '+ '、'.join(f"{r['old_px_new_map_mm']:.0f}" for r in err)+' mm。','',
        f"跨批次八步起点距专家目标中位数为 {median(r['initial'] for r in ep):.2f} mm，俯视末段定位误差中位数为 {median(r['coarse_error'] for r in ep):.2f} mm。向估计目标的八步接近量中位数为 {median(r['coarse_advance'] for r in ep):+.2f} mm。原本已接近真实位置的样本，仍可能被带向有偏的估计位置。",'']
    reasons={'no_intent':'没有策略收爪意图','no_coarse_agreement':'出现意图时，俯视没有进入允许范围','no_wrist_confirmation':'进入俯视允许范围后，腕部或高度等末端条件未同时满足','insufficient_consecutive_frames':'有单帧放行，但意图与连续确认没有同时满足'}
    for group,caption in [('regression','原 32 条'),('archive','跨批次 20 条')]:
        c=Counter(v['reason'] for v in m.values() if v['group']==group)
        assert sum(c.values())==new[group]['regions']['全部']['closes'][1].get('miss',0)
        out+=[caption+'的窗口漏收原因分别为'+'，'.join(f'{reasons[key]} {v} 条' for key,v in c.items())+'。','']
    out+=['三轮既有现场失败录像共 347 个已发送帧，按归档输入进行虚拟回放。三轮均未触发虚拟收爪。15 个虚拟设备保护用例通过，STOP 心跳 500 ms、预测有效期 100 ms。所有配对八步窗口均核对了 FK 高度、姿态和 ODE 开度。','',
          '本次使用已被上一轮检查过的 20 条做回归。没有新增独立盲测；跨批次左半区无样本。基线使用上一轮冻结全量逐条结果，并在本次每组重算首末两条的自然位移和收爪事件。','',
          '复算区域表可在此目录执行 `python summarize.py`。完整回放依赖私有同步档案和先前冻结的 replay.py、场景参数与模型。依次运行 study.py、build.py、score.py 的 regression 与 archive 两组，以及 audit.py 的默认、endpoints 和 misses 三种模式。未发布原图、示范、模型权重或私有几何参数。']
    return '\n'.join(out)+'\n'
if __name__=='__main__':
    folder=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
    print(report(folder),end='')

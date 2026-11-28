"""Render a public result and a private per-window comparison from the completed paired runs."""
from analysis import *
from html import escape
def num(f):return f'{f:.4f}' if abs(f)<.1 else f'{f:.2f}'
def report(s):
    modes=s['modes']['all'];w=s['windows'];common=s['input_max'];tim=s['timing']['all']
    lines=['# B1 原始 ODE 训练端与执行台同窗复算','',
        f"比对原 16 对摆位及跨批次 20 条示范。{s['heldout_episodes']} 条共 {w} 个滑动八步窗口，互不重叠的窗口有 {s['grid']} 个。按示范 ID 与真正入训清单核对，重合 {len(s['overlap'])} 条。训练侧另核对真实的 {s['training_episodes']} 条、{s['training_rows']} 行缓存，控制窗使用同样的专家分组和非重叠取窗规则。",'',
        f"采用相同零偏移时，八步中任一步两端的最大夹口差只有 {num(modes['train_zero']['tip_max']['max'])} mm。原始 ODE 在未入训纠偏和下降窗的错向数为 {s['labels']['correct']['wrong']}/{s['labels']['correct']['n']} 和 {s['labels']['descend']['wrong']}/{s['labels']['descend']['n']}，两条路径的方向判定分别翻转 {s['labels']['correct']['flips_zero_vs_station']} 和 {s['labels']['descend']['flips_zero_vs_station']} 个。与实际训练缓存对照，现有证据把主要问题定位到已训练策略在未见示范上的泛化，尤其是较早批次。历史验收与现场的混合分量选择另有可测差别。",'',
        '## 两条路径预测相差多少','',
        '| 训练侧同窗预测与现场原始 ODE 对比 | 范围 | 第八步夹口距离差中位数／P95／最大（mm） | 八步中任一步最大夹口差（mm） | 任一步最大单轴差（°） | 最大开度差（mm） |','|---|---|---:|---:|---:|---:|']
    for k,name,ns in [('train_zero','双方同用零偏移',w),('component_a','验收脚本抽取分量中心，种子 2026',w),('component_b','分量中心，种子 2027',s['grid']),('aug_a','仅恢复训练的逐帧随机裁剪，种子 2026',s['grid']),('aug_b','仅恢复随机裁剪，种子 2027',s['grid'])]:
        q=modes[k];dd=q['tip8'];lines.append(f"| {name} | {ns} 窗 | "+'／'.join(num(dd[a]) for a in ('p50','p95','max'))+f" | {num(q['tip_max']['max'])} | {num(q['joint_deg']['max'])} | {num(q['jaw_mm']['max'])} |")
    lines+=['', '混合分量的差异集中在跨批次数据。固定种子 2026 时，在 11 月 23 日及 11 月 20 日两批，第八步差的 P95 分别为 '+ ' 和 '.join(num(s['modes'][g]['component_a']['tip8']['p95']) for g in ('regression','archive'))+' mm。超过 100 mm 的窗口分别有 '+' 和 '.join(str(s['component_tail'][g]['above100']) for g in ('regression','archive'))+' 个。它们是未经限幅的预测目标差，未下发。','',
        '## 专家在纠偏和下降时预测的方向','',
        '横向用沿专家实测方向的投影，正数为同向。高度用基座 ΔZ，负数为下降。每组取中位数。表中的反向和上抬按专家反馈判定；与实际监督指令另作核对。','',
        '| 数据 | 横向窗 | 专家／原始 ODE 横向投影（mm） | ODE 反向 | 下降窗 | 专家／原始 ODE 高度（mm） | ODE 持平或向上 |','|---|---:|---:|---:|---:|---:|---:|']
    for g,label in [('regression','11 月 23 日未入训，32 条'),('archive','11 月 20 日跨批次，20 条'),('all','未入训合计'),('training','真实训练缓存，50 条')]:
        a,b=[s['phase'][g][k] for k in ('correct','descend')];av,bv=a['versions'],b['versions']
        lines.append(f"| {label} | {a['windows']} | {num(av['expert']['median'])}／{num(av['train_zero']['median'])} | {av['train_zero']['wrong']} | {b['windows']} | {num(bv['expert']['median'])}／{num(bv['train_zero']['median'])} | {bv['train_zero']['wrong']} |")
    lines+=['','## 输入、标签及采样时刻','',
       '- 两路相机分别按角色读取。同一源图在训练转换中经 RGB、上下各补 80 个黑色像素、面积插值至 128×128、转 CHW、除以 255。评估时中心裁剪 112×112 并做 ImageNet 标准化。执行台单帧编码并缓存八帧特征，训练评估一次编码八帧。两路最终视觉张量的逐元素最大差分别为 '+str(common['overhead_transformed'])+'、'+str(common['wrist_transformed'])+'。',
       f"- 第 t 帧观测包含 t−7…t 的反馈；起点是 t−1 已下发指令，监督是 t…t+7 八条绝对指令。前六维是弧度，第七维是毫米。用实际训练采样器逐元素比对，状态、锚点、八步标签最大差依次为 {common['state']:g}、{common['anchor']:g}、{common['labels']:g}。按训练源转换重建的图像与真实缓存逐张核对，俯视和腕部 uint8 最大像素差为 {s['training_checks']['head_camera_stored_u8']:g}、{s['training_checks']['wrist_camera_stored_u8']:g}。所有训练行的时间戳、状态、指令和锚点都复核过。",
       f"- 状态与动作各用其训练集逐维最小、最大值做线性映射。独立重算与两端实际归一化的最大差为 {max(common['action_manual'],common['agent_pos_manual'],common['agent_pos_norm'],common['action_norm']):g}。反归一化浮点往返最大误差为 {max(common['action_roundtrip'],common['agent_pos_roundtrip']):.8f}（各维原单位）。",
       f"- 逐一比对帧名与源单调时钟，最大舍入差 {common['filename_rounding_ns']:g} ns。去重后，双相机时间差中位数／P95／最大值为 "+'／'.join(num(tim['camera_skew'][k]) for k in ('p50','p95','max'))+' ms；相机采样间隔中位数 '+num(tim['sample_interval_ms']['p50'])+' ms，六轴目标中最旧分量的年龄 P95 为 '+num(tim['six_joint_target_oldest']['p95'])+' ms。',
       '- 反馈位移取 t 到 t+8；监督位移另按同一实测起点比较 t+7 指令，并把标签终点前后各移动一帧。等效时序筛选要求窗内双相机最大差 <16.67 ms，最旧六轴目标和反馈均 <33.34 ms。','',
       '| 错误种类 | 零偏移两端判定翻转 | 原始错误中监督方向正确 | 错误中标签提前／原位／延后一帧都支持专家 | 时序筛选后仍错／总数 |','|---|---:|---:|---:|---:|']
    for phase,label in [('correct','横向反向'),('descend','该下降时持平或上抬')]:
        d=s['labels'][phase];lines.append(f"| {label} | {d['flips_zero_vs_station']} | {d['official_label_right_when_wrong']}/{d['wrong']} | {d['all_shifts_right_when_wrong']}/{d['wrong']} | {d['fresh_wrong']}/{d['fresh']} |")
    p=s['packets']['all'];n=p['counts']
    lines+=['',f"额外按 CAN 原始负载独立解码 {n.get('decoded',0)} 条去重目标记录，六轴顺序、符号及弧度换算不一致 {n.get('joint_mismatch',0)} 条，夹爪毫米换算不一致 {n.get('jaw_mismatch',0)} 条。缺少原始四帧的目标记录有 {n.get('not_stored',0)} 条。",'',
       '## 分量选择、图像增强与关节误差','',
       '训练的轨迹损失监督零偏移主路径。现场也使用零偏移。历史验收接口的 eps_center=True 会按权重抽取 12 个分量之一，再取该分量中心。下面按各窗口自身的 12 个概率精确加权。随机裁剪试验仅开启图像裁剪器，固定权重、上一指令、归一化和零偏移。','',
       '| 指标 | 执行台零偏移 | 验收分量中心（种子 2026／2027） | 全部 12 分量按概率加权的预期数 | 随机裁剪（种子 2026／2027） |','|---|---:|---:|---:|---:|']
    for ph,label in [('correct','反向'),('descend','持平／上抬')]:
        z=s['phase']['all'][ph]['versions'];a=s['mix'][ph];lines.append(f"| {label}（{a['n']} 窗） | {z['station']['wrong']} | {z['component_a']['wrong']}／{z['component_b']['wrong']} | {a['expected_wrong']:.2f} | {z['aug_a']['wrong']}／{z['aug_b']['wrong']} |")
    for ph,label in [('correct','纠偏'),('descend','下降')]:
        d=s['mix'][ph];z=s['span'][ph];bf=s['phase']['all'][ph]['versions']['bf16']
        lines+=['',f"{label}窗中，恢复训练所用 BF16 自动混合精度、固定中心裁剪及零偏移后，夹口端点与 FP32 的差中位数 {num(d['bf16_tip_mm']['p50'])} mm，错误 {bf['wrong']}/{d['n']}。直接调用原 loss_terms，在评估模式关闭训练随机性后，同显式八步监督重算的轨迹损失最大差 {d['max_loss_error']:.7f}。实测六轴历史超出训练归一化范围的有 {z['state']['all_joints']}/{d['n']} 窗，错误窗中有 {z['state']['wrong_joints']} 个。"]
    lines+=['','对专家监督方向正确而 ODE 输出错误的窗，用同一套正运动学计算六轴的全部 64 种替换组合，按六轴替换顺序的平均边际贡献分摊“预测目标减专家目标”的夹口误差。正数表示把夹口推向错误方向。数值是输出几何贡献，不代表对应驱动器或电机故障。','',
        '| 错误 | 窗口数 | J1 | J2 | J3 | J4 | J5 | J6 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for ph,label in [('correct','朝专家反方向的误差（平均 mm）'),('descend','向上的误差（平均 mm）')]:
        z=s['axes'][ph];lines.append('| '+' | '.join([label,str(z['windows'])]+[num(v) for v in z['mean_mm']])+' |')
    a=s['axes']['correct'];d=s['axes']['descend'];st=s['span']['correct']['state'];la=s['labels']['correct']
    lines+=['',f"逐窗看，横向误差的最大不利贡献来自 J4 的有 {a['most_adverse'][3]} 窗，来自 J5 的有 {a['most_adverse'][4]} 窗，共 {a['most_adverse'][3]+a['most_adverse'][4]}/{a['windows']}。上抬误差以 J2 为最大贡献的有 {d['most_adverse'][1]}/{d['windows']} 窗，J2 的贡献中位数为 +{num(d['median_mm'][1])} mm。",'',
        f"按实际关节历史是否超出训练集逐轴最小至最大值分组，范围外纠偏窗有 {st['wrong_joints']}/{st['all_joints']} 个反向，范围内有 {la['wrong']-st['wrong_joints']}/{la['n']-st['all_joints']} 个反向。这一关联提示训练姿态覆盖值得进一步区分，摆位、视觉条件与关节姿态在现有批次中同时变化。"]
    lines+=['','## 解释范围','',
        '52 条均以真实示范反馈和实际上一条专家指令为条件。未入训示范的训练侧输入从原始文件按归档导出转换重建，并调用原 TrainingFrames.batch；训练对照直接读取原来持久化的缓存。核对了两种算法路径的同窗一致性、训练缓存的源数据和部署的运行版本。回放没有调用设备接口，也没有重训。离线结果可以定位到输入、推理实现或已学出的预测，但不能凭一个检查点区分训练时长、增强、样本覆盖与损失权重各自的因果贡献。']
    return '\n'.join(lines)+'\n'
def private_detail(rows):
    data=[[r['ep'],r['t'],r['phase'],int(r['grid']),*[r['v'][k] for k in ('expert','command','station','train_zero','component_a')],r['delta']['train_zero']['tip_max'],r['delta']['train_zero']['tip8'],r['delta']['component_a']['tip8']] for r in rows]
    page="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>B1 同窗双路径复算</title><style>body{font:15px/1.5 system-ui;margin:22px;color:#20303b}table{border-collapse:collapse;font-size:13px}td,th{padding:6px 12px;border-bottom:1px solid #ccc}th{position:sticky;top:0;background:#eef2f4}select,button{font:inherit}#panel{max-height:65vh;overflow:auto}</style><h1>B1 同窗双路径复算</h1><p>端点均为相对起始时刻专家实测夹口的基座位移 ΔX / ΔY / ΔZ，单位毫米。分量中心列固定种子 2026。</p><label>示范 <select id="ep"><option value="">全部</option>OPTIONS</select></label> <label>阶段 <select id="ph"><option value="">全部</option><option value="correct">纠偏</option><option value="descend">已对准下降</option><option value="other">其余</option></select></label> <label><input id="grid" type="checkbox" checked>互不重叠窗</label> <label><input id="err" type="checkbox">只看反向或上抬</label><p id="count"></p><div id="panel"><table><thead><tr><th>示范</th><th>起始帧</th><th>阶段</th><th>专家实测</th><th>专家监督</th><th>现场零偏移</th><th>训练评估零偏移</th><th>验收分量中心</th><th>零偏移八步最大差</th><th>零偏移第八步差</th><th>分量中心第八步差</th></tr></thead><tbody></tbody></table></div><button id="back">上一页</button> <button id="next">下一页</button><script>const rows=DATA;const $=x=>document.getElementById(x);let page=0;function render(){const a=rows.filter(x=>(!$('ep').value||x[0]===$('ep').value)&&(!$('ph').value||x[2]===$('ph').value)&&(!$('grid').checked||x[3])&&(!$('err').checked||(x[2]==='correct'&&x[4][0]*x[6][0]+x[4][1]*x[6][1]<0)||(x[2]==='descend'&&x[6][2]>=0)));let n=Math.ceil(a.length/70);page=Math.max(0,Math.min(page,n-1));$('count').textContent=a.length+' 个窗口，第 '+(n?page+1:0)+' / '+n+' 页';$('back').disabled=!page;$('next').disabled=page+1>=n;let body=document.querySelector('tbody');body.replaceChildren();for(let x of a.slice(page*70,page*70+70)){let tr=body.insertRow();for(let v of [x[0],x[1],{correct:'纠偏',descend:'下降',other:'其余'}[x[2]],...x.slice(4,9).map(v=>v.map(q=>q.toFixed(3)).join(' / ')),...x.slice(9).map(q=>q.toFixed(5))])tr.insertCell().textContent=v} $('panel').scrollTop=0}for(let k of ['ep','ph','grid','err'])$(k).onchange=()=>{page=0;render()};$('back').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};render();</script></html>"""
    return page.replace('OPTIONS',''.join('<option>'+escape(e)+'</option>' for e in sorted({r['ep'] for r in rows}))).replace('DATA',json.dumps(data,separators=(',',':')))
if __name__=='__main__':
    s,eps,rows=collect()
    write(base/'summary.json',s);write(base/'episode_outcomes.json',eps)
    (base/'README.md').write_bytes(report(s).encode())
    (base/'B1_双路径逐窗核对.html').write_bytes(private_detail(rows).encode())
    print(len(eps),len(rows),'verified and rendered')

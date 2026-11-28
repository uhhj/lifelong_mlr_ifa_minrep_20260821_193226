"""Create a local self-contained, filterable review of every assessed frame window."""
from pathlib import Path
from html import escape
import json
from measures import *
LABEL={'correct':'横向纠偏','descend':'已对准下降','other':'混合／停留／其余'}
def html(path,byep,meta,reps):
    records=[]
    for ep,rows in sorted(byep.items()):
        for r in rows:
            item=[ep,r['t'],r['phase'],r['grid'],any(q['t']==r['t'] for q in reps[ep].values()),r['image_phase'],
                  r['d0'],r['d8'],r['expert'],r['command'],r['raw']]
            for n in VERSIONS:
                roll=r.get('rolling',{}).get(n,{})
                item.append([r[n]['plan'],r[n]['bound'],roll.get('requested'),roll.get('final'),r[n]['source'],roll.get('guard'),
                             roll.get('clips'),roll.get('plans'),r[n]['blocked']])
            item.extend([r['micro'],[attribution(r,n) for n in VERSIONS]])
            records.append(item)
    first=''.join('<option>'+escape(ep)+'</option>' for ep in sorted(byep))
    page="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B1 八步位移逐窗对照</title>
<style>*{box-sizing:border-box}body{font:15px/1.5 system-ui,"Microsoft YaHei",sans-serif;margin:22px;color:#1a2932;background:#fbfcfc}h1{font-size:24px;margin-bottom:7px}p{max-width:94ch;margin:5px 0 12px}label,button{margin-right:12px}select,button{font:inherit;padding:4px}button{cursor:pointer}table{border-collapse:collapse;width:100%;background:white;font-size:13px}td,th{border-bottom:1px solid #dce3e5;text-align:left;vertical-align:top;padding:8px}th{position:sticky;top:0;background:#e9eff1}td span{white-space:nowrap}td small{color:#42515a}button:disabled{cursor:default}#wrap{max-height:70vh;overflow:auto}#status{margin:10px 0} .src{color:#734321}</style>
<h1>B1 八步位移逐窗对照</h1><p>每格第一行是基座坐标系的 ΔX、ΔY、ΔZ（毫米）。第二行依次是横向幅度、从 +X 朝 +Y 的方向角。纠偏窗第三行是沿专家方向的投影，负数表示走反。横移不足 0.5 mm 时不显示方向角。默认每条示范显示一个代表性纠偏窗。可切换查看全部滑窗、非重叠窗和 3–8 mm 小幅纠偏窗。</p>
<p>第八个专家指令是 t+7，实际反馈终点是 t+8。执行台滚动回放每步使用示范反馈及上一条模拟发送目标。若目标保护或逆解重建中断，保留原因并留空终点。其余类另外复算旧锚点和两种参照都支持的 1–3 mm 微调。</p>
<label>示范 <select id="ep"><option value="">全部 52 条</option>OPTIONS</select></label><label>阶段 <select id="phase"><option value="correct">横向纠偏</option><option value="descend">已对准下降</option><option value="other">混合／停留／其余</option><option value="">全部</option></select></label><label>取窗 <select id="sample"><option value="all">所有滑窗</option><option value="grid">互不重叠</option><option value="rep" selected>每条代表窗</option><option value="small">3–8 mm 非重叠纠偏窗</option><option value="micro">双参照支持的 1–3 mm 微调</option></select></label><label>执行台 <select id="ver"><option value="11">现场版</option><option value="12">离线候选版</option></select></label>
<div id="status"></div><div id="wrap"><table><thead><tr><th>示范／起始帧</th><th>专家阶段／距离</th><th>专家实际</th><th>专家指令</th><th>原始 ODE</th><th>视觉纠偏后计划</th><th>固定计划限速后</th><th>滚动末帧请求</th><th>执行台滚动最终目标</th><th>窗口状态</th></tr></thead><tbody></tbody></table></div><p><button id="prev">上一页</button><button id="next">下一页</button></p>
<script>const data=PAYLOAD;const labels={correct:'横向纠偏',descend:'已对准下降',other:'其余'};const src={raw_pass:'ODE 直通',planar_correction:'平面纠偏',wrist_guidance:'腕部引导',wrist_deadband:'腕部误差带内',no_target_hold:'无可靠目标，保持 XY'};const $=x=>document.getElementById(x);let page=0;
const fmt=x=>(Math.abs(x)<.005?0:x).toFixed(2);function vector(v,e,p){if(!v)return '—';let h=Math.hypot(v[0],v[1]),a=h<.5?'—':(Math.atan2(v[1],v[0])*180/Math.PI).toFixed(0)+'°';let pr=p==='correct'?'<br><small>投影 '+fmt((e[0]*v[0]+e[1]*v[1])/Math.hypot(e[0],e[1]))+'</small>':'';return '<span>'+v.map(fmt).join(' / ')+'</span><br><small>'+fmt(h)+' mm · '+a+'</small>'+pr}
function render(){let a=data.filter(r=>(!$('ep').value||r[0]===$('ep').value)&&(!$('phase').value||r[2]===$('phase').value)&&($('sample').value==='all'||($('sample').value==='grid'?r[3]:$('sample').value==='small'?r[3]&&r[2]==='correct'&&Math.hypot(r[8][0],r[8][1])<=8:$('sample').value==='micro'?r[13]:r[4])));let limit=60,n=Math.ceil(a.length/limit);page=Math.max(0,Math.min(page,n-1));$('status').textContent=a.length+' 个窗口 · 第 '+(n?page+1:0)+' / '+n+' 页';$('prev').disabled=!page;$('next').disabled=page+1>=n;let tbody=document.querySelector('tbody');tbody.replaceChildren();for(let r of a.slice(page*limit,(page+1)*limit)){let v=r[+$('ver').value],tr=tbody.insertRow();let cells=[r[0]+'<br><small>帧 '+r[1]+(r[3]?' · 网格':'')+(r[4]?' · 代表':'')+'</small>',(r[13]?'双参照微调':labels[r[2]])+'<br><small>抓取参考距 '+fmt(r[6])+' → '+fmt(r[7])+' mm<br>图像参考分组 '+labels[r[5]]+'</small>',...[r[8],r[9],r[10],v[0],v[1],v[2],v[3]].map(x=>vector(x,r[8],r[2])),src[v[4]]+(v[8]?'<br>固定计划保护中断':'')+(v[5]?'<br>滚动中断':v[3]?'<br>生成 '+v[7]+' 次计划，限幅 '+v[6]+' 次':'<br>此窗未滚动复算')];for(let text of cells)tr.insertCell().innerHTML=text;if(r[2]==='correct'){tr.lastChild.appendChild(document.createElement('br'));tr.lastChild.appendChild(document.createTextNode(r[14][+$('ver').value-11]))}if(v[5])tr.lastChild.appendChild(document.createTextNode(' · '+v[5]))} $('wrap').scrollTop=0}
for(let k of ['ep','phase','sample','ver'])$(k).onchange=()=>{if(k==='sample'&&$('sample').value==='micro')$('phase').value='';if(k==='sample'&&$('sample').value==='small')$('phase').value='correct';page=0;render()};$('prev').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};render();</script></html>"""
    path.write_text(page.replace('OPTIONS',first).replace('PAYLOAD',json.dumps(records,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')),encoding='utf-8')

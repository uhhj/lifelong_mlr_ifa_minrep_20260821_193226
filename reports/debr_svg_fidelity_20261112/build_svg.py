"""Reference-coordinate SVG reconstruction; preserve refined A's appearance."""
from pathlib import Path
from html import escape
import json, hashlib, xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
REFERENCE=HERE.parent/'20261111_debr_framework_A_refined/DEBR_A_refined.png'
W,H=1755,896
p=[]
BLUE='#0062CA';TEAL='#009C90';ORANGE='#FF6909';BLACK='#080808'
def put(s):p.append(s)
def text(x,y,s,size=21,bold=False,italic=False,anchor='middle',fill=BLACK,raw=False):
    put(f'<text x="{x}" y="{y}" font-family="Times New Roman, Times, serif" font-size="{size}" font-weight="{"bold" if bold else "normal"}" font-style="{"italic" if italic else "normal"}" text-anchor="{anchor}" fill="{fill}">{s if raw else escape(s)}</text>')
def math(base,sub='',sup=''):
    # Matching script positions, both referenced to the same baseline.
    a=f'<tspan font-style="italic">{escape(base)}</tspan>'
    if sup:
        a+=f'<tspan baseline-shift="super" font-size="70%">{escape(sup)}</tspan>'
    if sub:
        a+=f'<tspan'+(' dx="-9"' if sup else '')+f' baseline-shift="sub" font-size="70%">{escape(sub)}</tspan>'
    return a
def lines(x,y,strings,size=21,step=24,bold=False):
    for i,s in enumerate(strings):text(x,y+i*step,s,size,bold)
def box(x,y,w,h,theme='blue',r=13):
    stroke={'blue':BLUE,'teal':TEAL,'orange':ORANGE,'gray':'#226AB9'}[theme]
    put(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="url(#{theme}Fill)" stroke="{stroke}" stroke-width="1.15"/>')
def band(x,y,w,h,fill,r=14):
    put(f'<path d="M{x+r},{y} H{x+w-r} Q{x+w},{y} {x+w},{y+r} V{y+h} H{x} V{y+r} Q{x},{y} {x+r},{y} Z" fill="{fill}"/>')
def panel(x,y,w,h,theme,title,header_h=45,fontsize=29):
    color={'blue':BLUE,'teal':TEAL,'orange':ORANGE,'bottom':BLUE}[theme]
    fill='url(#orangePanel)' if theme=='orange' else 'url(#whitePanel)'
    put(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="17" fill="{fill}" stroke="{color}" stroke-width="2.2"/>')
    band(x,y,w,header_h,f'url(#{theme}Header)',17)
    text(x+30,y+30,title,fontsize,True,anchor='start',fill='white')
def headbox(x,y,w,h,title,theme='blue',header_h=29,title_size=23):
    box(x,y,w,h,theme)
    band(x+1,y+1,w-2,header_h,f'url(#{theme}Tint)',12)
    text(x+w/2,y+22,title,title_size,True)
def path(d,color=BLACK,end=True,width=2.4):
    marker={'#080808':'black',BLUE:'blue',TEAL:'teal'}[color]
    put(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"'+(f' marker-end="url(#{marker}Arrow)"' if end else '')+'/>')
def arrow(points,color=BLACK,end=True):
    path('M'+' L'.join(f'{x},{y}' for x,y in points),color,end)

put(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">')
put('<title id="title">DEBR Dual-Experience Behavioral Retention</title><desc id="desc">Editable vector rendition preserving the selected refined A composition, gradients, stage-memory rows, and right-side deployment layout.</desc>')
put('<defs>')
for name,start,end,direction in [
 ('blueHeader','#0073D9','#0062C5','horizontal'),('tealHeader','#009D92','#008E88','horizontal'),
 ('orangeHeader','#FF7510','#FF6809','horizontal'),('bottomHeader','#387AC9','#2E6AB8','horizontal'),
 ('blueFill','#F1F8FC','#DFEFF9','vertical'),('tealFill','#F2FCF8','#E1F4EF','vertical'),
 ('orangeFill','#FFF8ED','#FFEFDA','vertical'),('grayFill','#F8FCFF','#EDF7FC','vertical'),
 ('blueTint','#D7EAF7','#D3E7F6','horizontal'),('tealTint','#E7F6F0','#D9F0EB','horizontal'),
 ('orangeTint','#FFE7C7','#FFE1BD','horizontal'),('orangePanel','#FFF4E6','#FFF0DF','vertical'),
 ('whitePanel','#FFFFFF','#FAFCFE','vertical')]:
    put(f'<linearGradient id="{name}" x1="0" y1="0" x2="{"1" if direction=="horizontal" else "0"}" y2="{"0" if direction=="horizontal" else "1"}"><stop stop-color="{start}"/><stop offset="1" stop-color="{end}"/></linearGradient>')
for name,col in [('black',BLACK),('blue',BLUE),('teal',TEAL)]:
    put(f'<marker id="{name}Arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L12,6 L0,12 Z" fill="{col}"/></marker>')
put('</defs><rect width="1755" height="896" fill="white"/>')

# Major frames use coordinates measured from the refined PNG.
panel(52,46,866,263,'blue','CADL | Expert demonstrations')
panel(52,326,866,284,'teal','SHBR | Successful experience')
panel(938,20,525,594,'orange','Candidate-update gate',41,29)
p.pop()
text(1200.5,49,'Candidate-update gate',29,True,fill='white')
panel(90,633,1614,206,'bottom','At task end | Memory update for task k + 1',35,27)

# CADL. Preserve the reference widths, line breaks and type hierarchy.
box(86,106,222,83)
lines(197,130,['Current task','demonstrations'],22,23)
text(197,177,math('D','k'),26,raw=True)
box(86,211,222,78)
text(197,239,'Expert memory',23,True)
text(197,270,math('M','(k−1)','D'),27,raw=True)
headbox(428,143,265,109,'Execution supervision',title_size=24,header_h=37)
lines(560.5,209,['Behavior cloning + IFA','Execution error + Routing'],22,26)
box(766,147,129,102)
lines(830.5,175,['Base','objective'],24,25)
text(830.5,229,math('L','C'),28,raw=True)
path('M308,147 H339 Q344,147 344,152 V245 Q344,250 339,250 H308',end=False)
arrow([(344,198),(428,198)])
arrow([(693,198),(766,198)])

# SHBR: memory and support both feed retention.
box(90,387,208,184,'teal')
lines(194,412,['Successful-history','memory'],22,23,True)
text(194,463,math('M','(k−1)','S'),27,raw=True)
put('<path d="M107,477 H281" stroke="#49A29C" stroke-width="1"/>')
text(194,505,'Latent histories '+math('H','i')+';',21,raw=True)
text(194,528,'fixed Gaussian',21)
text(194,551,'references ('+math('μ','i','T')+', '+math('σ','i','T')+')',21,raw=True)
box(329,458,194,111,'teal')
text(426,488,math('q','j')+' = '+math('n','j','+')+' / '+math('N','roll'),25,raw=True)
lines(426,521,['Independent successful','trajectories','contributing to cache'],18.8,19)
box(552,387,230,180,'teal')
lines(667,412,['Support-weighted','retention'],23,24,True)
put('<path d="M575,451 H762 M584,493 H762" stroke="#78BEB5" stroke-width=".8"/>')
text(667,481,'Task-balanced slots',22)
text(667,520,'Fixed denominator '+math('B','R')+';',21,raw=True)
text(667,545,'empty slots = 0',22)
box(813,442,91,66,'teal')
text(858.5,483,math('L','SHBR'),27,raw=True)
arrow([(298,423),(552,423)])
arrow([(523,500),(552,500)])
arrow([(782,475),(813,475)])

# Candidate gate.
box(1025,74,354,78,'orange')
text(1202,97,'Same full state Ω',23,True)
text(1202,120,'(model, optimizer, scheduler, RNG)',20.5)
text(1202,143,'Same main-batch sequence',21)
headbox(989,194,174,87,'Base C','orange',27,23)
text(1076,242,math('L','C'),24,raw=True)
text(1076,271,'B = 10 updates',21)
headbox(1224,194,210,87,'Retention Q','orange',27,23)
text(1329,242,math('L','Q')+' = '+math('L','C')+' + '+math('λ','R')+math('L','SHBR'),23,raw=True)
text(1329,271,'B = 10 updates',21)
path('M1198,152 V168 Q1198,173 1193,173 H1079 Q1074,173 1074,178 V194')
path('M1198,168 Q1198,173 1203,173 H1323 Q1328,173 1328,178 V194')
path('M895,198 H940 Q945,198 945,203 V226 Q945,230 950,230 H989',BLUE)
path('M904,472 H953 Q958,472 958,467 V304 Q958,299 963,299 H1188 Q1193,299 1193,294 V262 Q1193,257 1198,257 H1224',TEAL)
headbox(1042,322,330,108,'Fixed training probes','orange',28,24)
for y,b,op in [(369,'BC','≤'),(395,'Exec','≤'),(420,'Ret','<')]:
    text(1207,y,math(b,'Q')+' '+escape(op)+' '+math(b,'C'),22,raw=True)
path('M1076,281 V302 Q1076,307 1081,307 H1105 Q1109,307 1109,311 V322')
path('M1339,281 V302 Q1339,306 1335,306 H1303 Q1299,306 1299,310 V322')
put(f'<path d="M1207,443 L1295,485 L1207,527 L1119,485 Z" fill="url(#orangeTint)" stroke="{ORANGE}" stroke-width="1.2"/>')
arrow([(1207,430),(1207,443)])
text(1207,480,'All three',22)
text(1207,505,'hold?',22)
box(989,468,77,35,'orange',9)
text(1027.5,493,math('Ω','Q'),25,raw=True)
box(1350,468,77,35,'orange',9)
text(1388.5,493,math('Ω','C'),25,raw=True)
arrow([(1119,485),(1066,485)])
arrow([(1295,485),(1350,485)])
text(1095,470,'Yes',22)
text(1320,470,'No',22)
box(1035,542,345,31,'orange',8)
text(1207.5,564,'Commit selected full state Ω*',22,True)
path('M1027,503 V520 Q1027,526 1033,526 H1121 Q1126,526 1126,531 V542')
path('M1389,503 V520 Q1389,526 1383,526 H1293 Q1287,526 1287,531 V542')
put('<rect x="1035" y="581" width="344" height="22" rx="8" fill="#FFE2C0"/>')
text(1207,599,'First task: base updates only',20)

# Preserve horizontal Learned policy -> Deployment, not stacked.
box(1480,264,109,96,'gray',12)
lines(1534.5,290,['Learned','stage policy'],21,24)
text(1534.5,342,math('π','k'),24,raw=True)
panel(1610,209,136,166,'blue','Deployment |',42,23)
# Narrow panel header overrides the generic left-padding title.
# Remove the just-added header text and center it exactly as in the reference.
p.pop()
text(1678,236,'Deployment |',23,True,fill='white')
box(1622,263,112,99,'gray',12)
lines(1678,292,['Original','policy','inference'],23,27)
arrow([(1589,314),(1610,314)])
path('M1380,558 H1529 Q1535,558 1535,552 V360')

# Bottom: two narrow rows centered under the left/middle of the diagram.
box(300,679,235,69)
text(417.5,706,'Current task',21)
text(417.5,731,'demonstrations '+math('D','k'),22,raw=True)
box(609,679,309,69,'blue')
text(763.5,699,'RA-Coreset',23,True)
lines(763.5,721,['Protected random anchors +','Coverage-aware exchange'],20,21)
box(1004,677,303,77)
text(1155.5,695,'Expert memory '+math('M','k','D'),21,True,raw=True)
lines(1155.5,715,['Fixed quota; previous','selections unchanged','Used in task k + 1'],19.7,17)
arrow([(535,714),(609,714)])
arrow([(918,714),(1004,714)])
box(300,770,235,55,'teal')
text(417.5,791,'Learned stage',21)
text(417.5,815,'policy '+math('π','k'),22,raw=True)
box(613,770,306,55,'teal')
text(766,792,math('N','roll')+' rollout attempts',21,True,raw=True)
text(766,816,'Retain successes only',21)
box(1001,764,333,69,'teal')
text(1167.5,785,'Successful-history memory '+math('M','k','S'),21,True,raw=True)
text(1167.5,807,'Store '+math('H','i')+', '+math('μ','i','T')+', '+math('σ','i','T')+' and task support',20,raw=True)
text(1167.5,828,'Used in task k + 1',20.5)
arrow([(535,798),(613,798)])
arrow([(919,798),(1001,798)])
text(877.5,878,'DEBR: Dual-Experience Behavioral Retention',29,True)
put('</svg>')
output=HERE/'DEBR_A_refined_faithful.svg'
output.write_text('\n'.join(p)+'\n',encoding='utf8')
root=ET.fromstring(output.read_bytes())
assert not root.findall('.//{http://www.w3.org/2000/svg}image')
audit=dict(status='STRUCTURAL_PASS_PENDING_VISUAL',reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
           svg_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),canvas=[W,H],embedded_rasters=0,
           text_elements=len(root.findall('.//{http://www.w3.org/2000/svg}text')),
           font='Times New Roman',original_png_unchanged=True,layout='reference-coordinate reconstruction',
           fidelity='Close visual vector reconstruction; not a pixel-identical conversion')
(HERE/'REVIEW.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
print(json.dumps(audit))

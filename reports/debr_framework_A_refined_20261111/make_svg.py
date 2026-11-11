"""Author an editable vector reconstruction of selected DEBR architecture A."""
from pathlib import Path
from html import escape
import xml.etree.ElementTree as ET
import hashlib, json

HERE=Path(__file__).resolve().parent
parts=[]
BLUE='#1763A0'; TEAL='#008B84'; ORANGE='#D9781C'; INK='#202A32'
def add(s): parts.append(s)
def text(x,y,s,size=22,bold=False,color=INK,anchor='middle',raw=False):
    add(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{"bold" if bold else "normal"}" fill="{color}" text-anchor="{anchor}">{s if raw else escape(s)}</text>')
def sym(base,sub='',sup=''):
    return escape(base)+(f'<tspan baseline-shift="super" font-size="72%">{escape(sup)}</tspan>' if sup else '')+(f'<tspan baseline-shift="sub" font-size="72%">{escape(sub)}</tspan>' if sub else '')
def lines(x,y,items,size=21,step=28):
    for i,s in enumerate(items):text(x,y+i*step,s,size)
def rect(x,y,w,h,fill='white',stroke=BLUE,r=12):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
def panel(x,y,w,h,title,color):
    rect(x,y,w,h,stroke=color,r=16)
    add(f'<path d="M {x+16} {y} H {x+w-16} Q {x+w} {y} {x+w} {y+16} V {y+46} H {x} V {y+16} Q {x} {y} {x+16} {y}" fill="{color}"/>')
    text(x+22,y+32,title,27,True,'white','start')
def arrow(points,color=INK,end=True):
    d='M '+' L '.join(f'{x} {y}' for x,y in points)
    marker='black' if color==INK else 'blue' if color==BLUE else 'teal'
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"'+(f' marker-end="url(#{marker})"' if end else '')+'/>')

add('<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="990" viewBox="0 0 1800 990" role="img" aria-labelledby="title desc">')
add('<title id="title">DEBR Dual-Experience Behavioral Retention</title><desc id="desc">Editable vector diagram with CADL expert demonstrations, SHBR successful histories, same-state candidate gate, and stage-end memory updates.</desc>')
add('<defs>')
for name,color in [('black',INK),('blue',BLUE),('teal',TEAL)]:
    add(f'<marker id="{name}" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L9 4.5 L0 9 Z" fill="{color}"/></marker>')
add('</defs><rect width="1800" height="990" fill="white"/><g font-family="Times New Roman, Times, serif">')

# Upper expert-demonstration pathway.
panel(30,30,850,275,'CADL | Expert demonstrations',BLUE)
rect(55,95,235,85,'#EDF4FB',BLUE)
lines(172.5,122,['Current task','demonstrations'],21,25)
text(172.5,169,sym('D','k'),23,raw=True)
rect(55,198,235,82,'#EDF4FB',BLUE)
text(172.5,226,'Expert memory',23,True)
text(172.5,261,sym('M','k−1','D'),25,raw=True)
rect(405,130,290,110,'#EDF4FB',BLUE)
text(550,160,'Execution supervision',24,True)
lines(550,191,['Behavior cloning + IFA','Execution error + Routing'],21,28)
rect(755,133,100,105,'#EDF4FB',BLUE)
lines(805,160,['Base','objective'],21,24)
text(805,218,sym('L','C'),26,raw=True)
arrow([(290,137),(330,137),(330,185)],end=False)
arrow([(290,239),(330,239),(330,185),(405,185)])
arrow([(695,185),(755,185)])

# Lower successful-experience pathway. Records and support are distinct inputs.
panel(30,330,850,305,'SHBR | Successful experience',TEAL)
rect(55,400,240,204,'#EBF7F5',TEAL)
lines(175,430,['Successful-history','memory'],23,27)
text(175,492,sym('M','k−1','S'),25,raw=True)
add(f'<path d="M75 509 H275" stroke="{TEAL}" stroke-width=".8"/>')
text(175,536,'Latent histories Hᵢ',21)
text(175,562,'Fixed Gaussian references',19)
text(175,588,'('+sym('μ','i','T')+', '+sym('σ','i','T')+')',23,raw=True)
rect(325,482,215,122,'#EBF7F5',TEAL)
text(432.5,513,sym('q','j')+' = '+sym('n','j','+')+' / '+sym('N','roll'),24,raw=True)
lines(432.5,542,['Independent successful','trajectories contributing','to the cache'],18,23)
rect(565,400,210,204,'#EBF7F5',TEAL)
lines(670,429,['Support-weighted','retention'],23,27)
lines(670,496,['Task-balanced slots'],20)
text(670,541,'Fixed denominator '+sym('B','R'),20,raw=True)
text(670,575,'Empty slots = 0',21)
rect(800,461,65,70,'#EBF7F5',TEAL)
text(832.5,502,sym('L','SHBR'),23,raw=True)
arrow([(295,432),(565,432)])
arrow([(540,547),(565,547)])
arrow([(775,496),(800,496)])

# Same-state candidate update gate.
panel(920,30,605,605,'Candidate-update gate',ORANGE)
rect(1000,92,445,80,'#FFF6EC',ORANGE)
text(1222.5,118,'Same full state Ω',25,True)
text(1222.5,143,'Model · optimizer · scheduler · RNG',20)
text(1222.5,165,'Same main-batch sequence',20)
rect(960,208,210,95,'#FFF6EC',ORANGE)
text(1065,235,'Base C',24,True)
text(1065,263,sym('L','C'),25,raw=True)
text(1065,290,'B = 10 updates',21)
rect(1270,208,220,95,'#FFF6EC',ORANGE)
text(1380,235,'Retention Q',24,True)
text(1380,263,sym('L','Q')+' = '+sym('L','C')+' + '+sym('λ','R')+sym('L','SHBR'),23,raw=True)
text(1380,290,'B = 10 updates',21)
arrow([(1222.5,172),(1222.5,190),(1065,190),(1065,208)])
arrow([(1222.5,190),(1380,190),(1380,208)])
arrow([(855,185),(900,185),(900,253),(960,253)],BLUE)
arrow([(865,496),(905,496),(905,318),(1245,318),(1245,255),(1270,255)],TEAL)
rect(1025,345,395,113,'#FFF6EC',ORANGE)
text(1222.5,370,'Fixed training probes',24,True)
for y,base,op in [(396,'BC','≤'),(422,'Exec','≤'),(448,'Ret','<')]:
    text(1222.5,y,sym(base,'Q')+' '+escape(op)+' '+sym(base,'C'),23,raw=True)
arrow([(1065,303),(1065,330),(1100,330),(1100,345)])
arrow([(1380,303),(1380,330),(1345,330),(1345,345)])
add(f'<path d="M1222.5 476 L1302.5 510 L1222.5 544 L1142.5 510 Z" fill="#FFF6EC" stroke="{ORANGE}" stroke-width="1.6"/>')
text(1222.5,505,'All three',21)
text(1222.5,528,'hold?',21)
arrow([(1222.5,458),(1222.5,476)])
rect(967,492,90,36,'#FFF6EC',ORANGE,8)
rect(1390,492,90,36,'#FFF6EC',ORANGE,8)
text(1012,518,sym('Ω','Q'),24,raw=True)
text(1435,518,sym('Ω','C'),24,raw=True)
arrow([(1142.5,510),(1057,510)])
arrow([(1302.5,510),(1390,510)])
text(1100,494,'Yes',20)
text(1345,494,'No',20)
rect(1020,565,405,35,'#FFF6EC',ORANGE,8)
text(1222.5,589,'Commit selected full state Ω*',23,True)
arrow([(1012,528),(1012,548),(1095,548),(1095,565)])
arrow([(1435,528),(1435,548),(1350,548),(1350,565)])
text(1222.5,625,'First task: base updates only',20)

# Output is attached to the commit node rather than the frame.
rect(1550,220,215,105,'#F2F5F8','#667787')
lines(1657.5,249,['Learned stage policy'],22)
text(1657.5,293,sym('π','k'),27,raw=True)
rect(1550,380,215,115,'#EDF4FB',BLUE)
text(1657.5,410,'Deployment',24,True,BLUE)
lines(1657.5,445,['Original policy','inference'],22,28)
arrow([(1425,582.5),(1538,582.5),(1538,272.5),(1550,272.5)])
arrow([(1657.5,325),(1657.5,380)])

# Stage-end updates, separate from the repeated update blocks.
panel(30,665,1735,258,'At task end | Memory update for task k + 1',BLUE)
rect(140,733,300,68,'#EDF4FB',BLUE)
text(290,761,'Current task demonstrations',22)
text(290,790,sym('D','k'),24,raw=True)
rect(580,733,410,68,'#EDF4FB',BLUE)
text(785,756,'RA-Coreset',23,True)
lines(785,778,['Protected random anchors +','Coverage-aware exchange'],19,20)
rect(1130,726,505,83,'#EDF4FB',BLUE)
text(1382.5,750,'Expert memory '+sym('M','k','D'),23,True,raw=True)
text(1382.5,775,'Fixed quota; previous selections unchanged',20)
text(1382.5,800,'Used in task k + 1',20)
arrow([(440,767),(580,767)])
arrow([(990,767),(1130,767)])
rect(140,835,300,64,'#EBF7F5',TEAL)
text(290,861,'Learned stage policy',22)
text(290,887,sym('π','k'),24,raw=True)
rect(580,835,410,64,'#EBF7F5',TEAL)
text(785,859,sym('N','roll')+' rollout attempts',23,True,raw=True)
text(785,886,'Retain successes only',21)
rect(1130,823,505,84,'#EBF7F5',TEAL)
text(1382.5,848,'Successful-history memory '+sym('M','k','S'),23,True,raw=True)
text(1382.5,875,'Store Hᵢ, '+sym('μ','i','T')+', '+sym('σ','i','T')+' and task support',21,raw=True)
text(1382.5,899,'Used in task k + 1',20)
arrow([(440,867),(580,867)])
arrow([(990,867),(1130,867)])
text(900,970,'DEBR: Dual-Experience Behavioral Retention',29,True)
add('</g></svg>')
output=HERE/'DEBR_A_editable.svg'
output.write_text('\n'.join(parts)+'\n',encoding='utf8')
root=ET.fromstring(output.read_bytes())
assert not root.findall('.//{http://www.w3.org/2000/svg}image')
audit=dict(status='VECTOR_RECONSTRUCTION_STRUCTURAL_PASS',embedded_bitmap_count=0,
           text_elements=len(root.findall('.//{http://www.w3.org/2000/svg}text')),
           font_family='Times New Roman',editable_text=True,
           sha256=hashlib.sha256(output.read_bytes()).hexdigest(),new_experiments=0)
(HERE/'VECTOR_REVIEW.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
print(json.dumps(audit))

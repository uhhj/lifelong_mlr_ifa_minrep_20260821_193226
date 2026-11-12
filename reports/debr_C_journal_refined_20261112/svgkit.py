"""Small native-SVG authoring helpers; text stays editable."""
from pathlib import Path
from html import escape

INK='#193247'; MUTED='#536A7B'; BLUE='#2368AC'; TEAL='#058C81'; GOLD='#C47A32'
BG_BLUE='#EFF6FD'; BG_TEAL='#ECF8F4'; BG_GOLD='#FFF5E8'; BORDER='#D7E1E8'
def sub(s): return f'<tspan baseline-shift="sub" font-size="70%">{escape(s)}</tspan>'
def sup(s): return f'<tspan baseline-shift="super" font-size="70%">{escape(s)}</tspan>'
def m(s,lo=''): return f'<tspan font-style="italic">{escape(s)}</tspan>'+ (sub(lo) if lo else '')

class SVG:
    def __init__(self,w=1800,h=940):
        self.w,self.h=w,h
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img"><title>DEBR: Dual-Experience Behavioral Retention</title><desc>Expert demonstrations and autonomous successful experience cooperate during learning. Native editable vectors, Times New Roman typography.</desc>',
        '<defs><linearGradient id="blueWash" x2="1" y2="1"><stop stop-color="#F6FAFE"/><stop offset="1" stop-color="#E7F1FC"/></linearGradient><linearGradient id="tealWash" x2="1" y2="1"><stop stop-color="#F4FCF9"/><stop offset="1" stop-color="#E0F3EE"/></linearGradient><linearGradient id="goldWash" x2="1" y2="1"><stop stop-color="#FFFAF1"/><stop offset="1" stop-color="#FCEEDB"/></linearGradient><filter id="shadow" x="-15%" y="-15%" width="130%" height="140%"><feDropShadow dx="0" dy="4" stdDeviation="7" flood-color="#17364D" flood-opacity=".07"/></filter>']
        for name,col in [('ink',INK),('blue',BLUE),('teal',TEAL),('gold',GOLD),('gray','#8396A3')]:
            self.parts.append(f'<marker id="arr-{name}" markerWidth="13" markerHeight="13" refX="11" refY="6.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M1,1 L12,6.5 L1,12 Z" fill="{col}"/></marker>')
        self.parts+=['</defs>',f'<rect width="{w}" height="{h}" fill="#FFFFFF"/>']
    def add(self,s): self.parts.append(s)
    def rect(self,x,y,w,h,fill='white',stroke=BORDER,r=18,sw=1.5,shadow=False):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'+(' filter="url(#shadow)"' if shadow else '')+'/>')
    def text(self,x,y,s,size=25,color=INK,bold=False,anchor='start',raw=False,italic=False):
        self.add(f'<text x="{x}" y="{y}" fill="{color}" font-family="Times New Roman" font-size="{size}" font-weight="{700 if bold else 400}" font-style="{"italic" if italic else "normal"}" text-anchor="{anchor}">{s if raw else escape(s)}</text>')
    def lines(self,x,y,ss,size=24,color=INK,step=30,**kw):
        for i,s in enumerate(ss): self.text(x,y+i*step,s,size,color,**kw)
    def line(self,x1,y1,x2,y2,color=BORDER,sw=1.5,dash=None):
        self.add(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="{sw}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
    def path(self,d,color=INK,sw=2.6,arrow=True,dash=None):
        name={INK:'ink',BLUE:'blue',TEAL:'teal',GOLD:'gold','#8396A3':'gray'}.get(color,'ink')
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"'+(f' marker-end="url(#arr-{name})"' if arrow else '')+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
    def circle(self,x,y,r,fill,stroke='none',sw=1.5):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def pill(self,x,y,w,label,color=BLUE,fill=BG_BLUE,size=20):
        self.rect(x,y,w,34,fill,'none',17);self.text(x+w/2,y+24,label,size,color,True,'middle')
    def trajectory(self,x,y,color=BLUE,scale=1):
        self.add(f'<g transform="translate({x} {y}) scale({scale})">')
        self.path('M0,45 C20,40 14,10 39,17 S66,52 91,21',color,2.5,False)
        for a,b in [(0,45),(39,17),(91,21)]:self.circle(a,b,5,'white',color,2)
        self.circle(91,21,2.4,color)
        self.add('</g>')
    def memory(self,x,y,color=BLUE):
        for dx,dy in [(10,-10),(5,-5),(0,0)]:self.rect(x+dx,y+dy,61,45,'white',color,6,1.6)
        for yy in [13,22,31]:self.line(x+12,y+yy,x+46,y+yy,color,1.4)
    def finish(self,path):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text('\n'.join(self.parts+['</svg>'])+'\n',encoding='utf8')

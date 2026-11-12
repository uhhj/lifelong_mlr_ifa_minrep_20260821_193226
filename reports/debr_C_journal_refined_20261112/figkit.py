"""Mixed-vector framework graphics with source-verified local frame embeds."""
from svgkit import *
from pathlib import Path
import base64
HERE=Path(__file__).resolve().parent
class Figure(SVG):
    def __init__(self,w=1800,h=1000):
        super().__init__(w,h);self.clip_id=0
    def photo(self,x,y,size,key,color=BLUE,label=None):
        self.clip_id+=1;cid=f'photo{self.clip_id}'
        data=base64.b64encode((HERE/'assets_local'/f'{key}.png').read_bytes()).decode()
        self.rect(x-4,y-4,size+8,size+8,'white',color,13,1.4)
        self.add(f'<defs><clipPath id="{cid}"><rect x="{x}" y="{y}" width="{size}" height="{size}" rx="9"/></clipPath></defs>')
        self.add(f'<image x="{x}" y="{y}" width="{size}" height="{size}" href="data:image/png;base64,{data}" clip-path="url(#{cid})"/>')
        if label:self.text(x+size/2,y+size+28,label,19,MUTED,False,'middle')
    def film(self,x,y,size,role,color=BLUE,gap=28,labels=None):
        for i in range(3):
            xx=x+i*(size+gap)
            self.photo(xx,y,size,f'{role}_{i}',color,labels[i] if labels else None)
            if i<2:self.path(f'M{xx+size+8},{y+size/2} H{xx+size+gap-8}',color,2.3)
    def badge(self,x,y,text,color=BLUE):
        self.circle(x,y,18,'white',color,1.6);self.text(x,y+7,text,21,color,True,'middle')
    def gate(self,x,y,w=1080,h=250):
        self.rect(x,y,w,h,'url(#goldWash)','#DFC6A7',20,1.5,True)
        self.text(x+28,y+40,'Dual-experience coordination',31,INK,True)
        self.text(x+w-28,y+38,'Same state · Same batches · B = 10',21,MUTED,False,'end')
        self.line(x+28,y+57,x+w-28,y+57,'#E0CDB7')
        self.rect(x+28,y+78,w*.34,70,'white','#BDD1E4',12)
        self.text(x+28+w*.17,y+107,'Base C',23,BLUE,True,'middle')
        self.text(x+28+w*.17,y+136,m('L','C'),25,INK,False,'middle',True)
        self.rect(x+w*.43,y+78,w*.54-28,70,'white','#B5D8CE',12)
        self.text(x+w*.70-14,y+107,'Retention Q',23,TEAL,True,'middle')
        self.text(x+w*.70-14,y+136,m('L','C')+' + '+m('λ','R')+m('L','SHBR'),25,INK,False,'middle',True)
        self.text(x+w*.20,y+184,m('BC','Q')+' ≤ '+m('BC','C'),24,INK,False,'middle',True)
        self.text(x+w*.50,y+184,m('Exec','Q')+' ≤ '+m('Exec','C'),24,INK,False,'middle',True)
        self.text(x+w*.80,y+184,m('Ret','Q')+' &lt; '+m('Ret','C'),24,INK,False,'middle',True)
        self.text(x+w/2,y+225,'All pass → Q; otherwise → C. Commit the selected full state.',23,INK,True,'middle')
    def disclosure(self,y=972):
        self.text(self.w/2,y,'Simulation imagery illustrates the two experience sources; different tasks, not a paired performance comparison.',17,MUTED,False,'middle')
    def footer(self,title,y=933):
        self.text(self.w/2,y,title,30,INK,True,'middle')
    def memory_pool(self,x,y,color=BLUE,kind='expert'):
        # Abstract experience representations, not empirical latent data.
        for i in range(3):self.rect(x+10*i,y-7*i,85,58,'white',color,7,1.4)
        if kind=='expert':
            for xx,yy in [(12,16),(36,12),(58,24),(18,40),(56,44)]:self.circle(x+xx,y+yy,4,color)
            self.path(f'M{x+12},{y+16} L{x+36},{y+12} L{x+58},{y+24} L{x+56},{y+44} L{x+18},{y+40}',color,1.3,False)
        else:
            self.path(f'M{x+10},{y+43} C{x+28},{y+44} {x+26},{y+10} {x+42},{y+10} S{x+58},{y+44} {x+75},{y+43}',color,2,False)

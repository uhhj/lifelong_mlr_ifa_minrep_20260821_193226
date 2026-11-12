"""B: balanced dual-experience wings with a compact coordinated-update layer."""
from pathlib import Path
from svgkit import *

HERE=Path(__file__).resolve().parent
s=SVG()

# Two complementary sources receive equal visual weight.
s.rect(58,40,812,350,'url(#blueWash)','#B6CFE6',24,1.5,True)
s.rect(930,40,812,350,'url(#tealWash)','#B5DAD1',24,1.5,True)
s.pill(88,64,158,'EXPERIENCE 1',BLUE,'#DDECF9',19)
s.pill(960,64,158,'EXPERIENCE 2',TEAL,'#D9F0E8',19)
s.text(88,139,'Expert demonstrations',36,BLUE,True)
s.text(960,139,'Autonomous successful experience',36,TEAL,True)
s.text(88,171,'Learn actions from organized expert supervision',23,MUTED)
s.text(960,171,'Retain behavior that the policy has already mastered',23,MUTED)

s.memory(96,223,BLUE)
s.lines(190,231,['Current demonstrations','+ expert memory'],24,INK,31)
s.path('M425,242 H473',BLUE,2.8)
s.rect(486,196,352,111,'white','#BED4E8',14)
s.text(662,226,'CADL',28,BLUE,True,'middle')
s.text(662,256,'Execution-aligned supervision',23,INK,False,'middle')
s.text(662,286,'BC + IFA + execution & routing',22,MUTED,False,'middle')
s.line(88,327,840,327,'#BFD2E1')
s.text(88,359,'At task end:',21,MUTED)
s.text(210,359,'demonstrations → RA-Coreset → expert memory',23,BLUE)

s.trajectory(969,209,TEAL,.72)
s.lines(1060,229,['Successful histories','+ fixed behavior references'],23,INK,31)
s.path('M1315,242 H1360',TEAL,2.8)
s.rect(1373,196,337,111,'white','#B8DCD2',14)
s.text(1541.5,226,'SHBR',28,TEAL,True,'middle')
s.text(1541.5,256,'Support-weighted retention',23,INK,False,'middle')
s.text(1541.5,286,m('q','j')+' = '+m('n','j')+sup('+')+' / '+m('N','roll')+'; fixed denominator',21,MUTED,False,'middle',raw=True)
s.line(960,327,1712,327,'#B7D9CF')
s.text(960,359,'At task end:',21,MUTED)
s.text(1082,359,'policy rollouts → successes → history memory',23,TEAL)

# The paired wings feed a shared training decision, rather than two policies.
s.path('M662,390 V426 Q662,437 674,437 H760 V476',BLUE,3)
s.path('M1542,390 V426 Q1542,437 1530,437 H1040 V476',TEAL,3)
s.rect(288,476,1110,300,'url(#goldWash)','#E5C9A8',22,1.6,True)
s.text(324,515,'Coordinated candidate update',30,INK,True)
s.text(1364,514,'Same full state • same main batches',23,MUTED,False,'end')
s.line(322,533,1364,533,'#E4CFB6')

s.rect(324,550,275,91,'white','#C4D5E5',14)
s.text(462,580,'Base candidate C',25,BLUE,True,'middle')
s.text(462,616,m('L','C')+'  ·  B = 10 updates',25,INK,False,'middle',raw=True)
s.rect(630,550,420,91,'white','#BCDCD1',14)
s.text(840,580,'Retention candidate Q',25,TEAL,True,'middle')
s.text(840,616,m('L','Q')+' = '+m('L','C')+' + '+m('λ','R')+m('L','SHBR'),26,INK,False,'middle',raw=True)
s.text(1222,578,'B = 10 updates',24,MUTED,False,'middle')
s.text(1222,612,'for both candidates',22,MUTED,False,'middle')

s.text(324,680,'TRAINING PROBES',18,GOLD,True)
s.text(550,681,m('BC','Q')+' ≤ '+m('BC','C'),26,INK,False,'start',raw=True)
s.text(790,681,m('Exec','Q')+' ≤ '+m('Exec','C'),26,INK,False,'start',raw=True)
s.text(1080,681,m('Ret','Q')+' &lt; '+m('Ret','C'),26,INK,False,'start',raw=True)
s.rect(324,707,1038,43,'#F9E7CE','none',12)
s.text(843,736,'All pass → Q   |   Otherwise → C   ·   Commit the selected full state',24,INK,True,'middle')

# One output policy; original inference remains unchanged.
s.path('M1398,625 H1452',INK,3)
s.rect(1466,558,276,146,'#F6F8FA','#B9C9D5',20,1.7,True)
s.text(1604,599,'Learned policy '+m('π','k'),29,INK,True,'middle',raw=True)
s.line(1494,617,1714,617,'#CDD8E0')
s.text(1604,647,'Original inference',24,MUTED,False,'middle')
s.text(1604,678,'at deployment',24,MUTED,False,'middle')

s.text(843,811,'First task: base updates only',22,MUTED,False,'middle')
s.text(900,868,'Two memories for task k + 1 • One shared policy • Coordinated learning and retention',24,MUTED,False,'middle')
s.text(900,918,'DEBR: Dual-Experience Behavioral Retention',32,INK,True,'middle')
s.finish(HERE/'DEBR_B.svg')
print(HERE/'DEBR_B.svg')

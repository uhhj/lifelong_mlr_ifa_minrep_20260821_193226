"""Selected B design: sharpen the complementary roles of both experiences."""
from pathlib import Path
from svgkit import *
HERE=Path(__file__).resolve().parent
s=SVG(1800,940)

# Equal visual weight and a shared grid make the complementary roles explicit.
for x,color,gradient,edge in [(58,BLUE,'blueWash','#B6CFE6'),(930,TEAL,'tealWash','#B5DAD1')]:
    s.rect(x,40,812,355,f'url(#{gradient})',edge,24,1.5,True)
    s.rect(x+30,63,4,99,color,'none',2)
s.text(106,99,'EXPERT DEMONSTRATION EXPERIENCE',19,BLUE,True)
s.text(978,99,'AUTONOMOUS SUCCESS EXPERIENCE',19,TEAL,True)
s.text(106,143,'Guide action learning',38,BLUE,True)
s.text(978,143,'Retain self-acquired behavior',38,TEAL,True)
s.text(106,176,'Learn from expert actions across tasks',24,MUTED)
s.text(978,176,'Preserve the policy’s own successful executions',24,MUTED)

s.memory(97,237,BLUE)
s.lines(191,243,['Current demonstrations','+ past expert memory'],24,INK,31)
s.path('M426,253 H473',BLUE,3)
s.rect(486,208,352,109,'white','#BED4E8',14)
s.text(662,239,'CADL',29,BLUE,True,'middle')
s.text(662,270,'Execution-aligned supervision',23,INK,False,'middle')
s.text(662,298,'BC + IFA + execution & routing',21.5,MUTED,False,'middle')

s.trajectory(970,221,TEAL,.72)
s.lines(1061,242,['Successful histories','+ frozen behavior references'],22.5,INK,31)
s.path('M1317,253 H1363',TEAL,3)
s.rect(1373,208,337,109,'white','#B8DCD2',14)
s.text(1541.5,239,'SHBR',29,TEAL,True,'middle')
s.text(1541.5,270,'Support-weighted retention',23,INK,False,'middle')
s.text(1541.5,298,m('q','j')+' = '+m('n','j')+sup('+')+' / '+m('N','roll')+'; fixed denominator',21,MUTED,False,'middle',raw=True)

# Two independent experience memories, created only at the task boundary.
for x,edge in [(88,'#BFD2E1'),(960,'#B7D9CF')]:s.line(x,338,x+752,338,edge)
s.text(88,370,'Task end',20,MUTED)
s.text(197,370,'Demonstrations → RA-Coreset → expert memory',22,BLUE)
s.text(960,370,'Task end',20,MUTED)
s.text(1069,370,'Fixed rollouts → successes → history memory',22,TEAL)

# Interlocking rings mean coordination, not arithmetic averaging of updates.
s.path('M662,395 V430 Q662,442 674,442 H874',BLUE,3.2)
s.path('M1542,395 V430 Q1542,442 1530,442 H926',TEAL,3.2)
s.circle(892,442,17,'white',BLUE,3)
s.circle(908,442,17,'white',TEAL,3)
s.path('M900,463 V492',GOLD,2.6)

# The gate is the shared decision layer beneath the two experience sources.
s.rect(288,494,1110,309,'url(#goldWash)','#E5C9A8',22,1.6,True)
s.text(324,535,'Dual-experience coordination',32,INK,True)
s.text(1364,534,'Same full state · Same main batches',22,MUTED,False,'end')
s.line(322,551,1364,551,'#E4CFB6')
s.rect(324,568,302,86,'white','#BDD0E3',14)
s.text(475,599,'Base candidate C',25,BLUE,True,'middle')
s.text(475,633,m('L','C'),27,INK,False,'middle',raw=True)
s.rect(658,568,397,86,'white','#B7D9CF',14)
s.text(856.5,599,'Retention candidate Q',25,TEAL,True,'middle')
s.text(856.5,633,m('L','Q')+' = '+m('L','C')+' + '+m('λ','R')+m('L','SHBR'),27,INK,False,'middle',raw=True)
s.text(1216,599,'B = 10 updates',23,MUTED,False,'middle')
s.text(1216,630,'for each candidate',22,MUTED,False,'middle')

s.text(475,682,'Current-task imitation',19,MUTED,False,'middle')
s.text(826,682,'Current-task execution',19,MUTED,False,'middle')
s.text(1178,682,'Successful-behavior retention',19,MUTED,False,'middle')
s.text(475,715,m('BC','Q')+' ≤ '+m('BC','C'),26,INK,False,'middle',raw=True)
s.text(826,715,m('Exec','Q')+' ≤ '+m('Exec','C'),26,INK,False,'middle',raw=True)
s.text(1178,715,m('Ret','Q')+' &lt; '+m('Ret','C'),26,INK,False,'middle',raw=True)
s.rect(324,741,1038,40,'#F7E4C9','none',12)
s.text(843,768,'All probes pass → Q; otherwise → C. Commit the selected full state.',23.5,INK,True,'middle')

s.path('M1398,649 H1452',INK,3)
s.rect(1466,577,276,145,'#F6F8FA','#B9C9D5',20,1.7,True)
s.text(1604,615,'One evolving policy',27,INK,True,'middle')
s.text(1604,651,m('π','k'),31,INK,True,'middle',raw=True)
s.line(1494,667,1714,667,'#CDD8E0')
s.text(1604,697,'Original inference at deployment',19,MUTED,False,'middle')

s.text(843,837,'Fixed training probes · First task: base updates only',21,MUTED,False,'middle')
s.text(900,878,'Complementary experiences. Coordinated learning and retention.',24,INK,False,'middle')
s.text(900,921,'DEBR: Dual-Experience Behavioral Retention',31,INK,True,'middle')
s.finish(HERE/'DEBR_B_refined.svg')

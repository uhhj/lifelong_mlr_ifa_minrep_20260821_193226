"""B: two grounded experience sources around one coordinated policy."""
from pathlib import Path
from figkit import *
HERE=Path(__file__).resolve().parent
s=Figure()

# Tall photo-led wings leave the coordination path at the visual center.
for x,fill,stroke in [(55,'url(#blueWash)','#B9D1E8'),(1235,'url(#tealWash)','#B5DAD0')]:
    s.rect(x,48,510,532,fill,stroke,23,1.5,True)
s.text(85,92,'Expert demonstration experience',29,BLUE,True)
s.text(85,126,'Guide action learning',24,MUTED)
s.text(1265,92,'Autonomous success experience',29,TEAL,True)
s.text(1265,126,'Retain self-acquired behavior',24,MUTED)

s.photo(94,164,220,'expert_0',BLUE)
s.photo(354,164,100,'expert_1',BLUE)
s.photo(354,294,100,'expert_2',BLUE)
s.text(481,224,'130',19,MUTED,False,'middle')
s.text(481,354,'260',19,MUTED,False,'middle')
s.text(204,411,'Frame 0',19,MUTED,False,'middle')
s.text(310,443,'Official demo · bowl into drawer',21,BLUE,False,'middle')

s.photo(1274,164,220,'success_0',TEAL)
s.photo(1534,164,100,'success_1',TEAL)
s.photo(1534,294,100,'success_2',TEAL)
s.text(1661,224,'65',19,MUTED,False,'middle')
s.text(1661,354,'77',19,MUTED,False,'middle')
s.text(1384,411,'Step 0',19,MUTED,False,'middle')
s.text(1490,443,'Archived success · bowl onto plate',21,TEAL,False,'middle')

s.rect(85,463,450,85,'white','#C0D5E9',15)
s.text(310,495,'CADL',28,BLUE,True,'middle')
s.text(310,526,'BC + IFA + execution supervision',23,INK,False,'middle')
s.rect(1265,463,450,85,'white','#B9DCD2',15)
s.text(1490,495,'SHBR',28,TEAL,True,'middle')
s.text(1490,526,'Support-weighted behavior retention',23,INK,False,'middle')

# Color-separated contributions enter their respective candidate objectives.
s.rect(620,165,560,400,'url(#goldWash)','#DFC7A8',22,1.7,True)
s.text(900,207,'Dual-experience coordination',29,INK,True,'middle')
s.text(900,240,'Same full state · Same main batches',21,MUTED,False,'middle')
s.line(646,253,1154,253,'#E3CFB6')
s.rect(646,270,216,78,'white','#B9D1E7',13)
s.text(754,299,'Base C',24,BLUE,True,'middle')
s.text(754,331,m('L','C'),27,INK,False,'middle',True)
s.rect(886,270,268,78,'white','#B5D9CD',13)
s.text(1020,299,'Retention Q',24,TEAL,True,'middle')
s.text(1020,331,m('L','Q')+' = '+m('L','C')+' + '+m('λ','R')+m('L','SHBR'),24,INK,False,'middle',True)
s.text(900,374,'B = 10 updates per candidate',22,MUTED,False,'middle')
s.text(900,400,'Fixed training probes',18,GOLD,False,'middle')
s.text(900,427,m('BC','Q')+' ≤ '+m('BC','C'),25,INK,False,'middle',True)
s.text(900,454,m('Exec','Q')+' ≤ '+m('Exec','C'),25,INK,False,'middle',True)
s.text(900,481,m('Ret','Q')+' &lt; '+m('Ret','C'),25,INK,False,'middle',True)
s.rect(646,497,508,43,'#F7E5CB','none',12)
s.text(900,526,'All pass → Q; otherwise → C',24,INK,True,'middle')
s.path('M535,506 H591 Q599,506 599,498 V308 H644',BLUE,2.8)
s.path('M1265,506 H1209 Q1201,506 1201,498 V308 H1156',TEAL,2.8)
s.text(900,599,'Commit the selected full state',23,INK,True,'middle')
s.path('M900,565 V576',INK,2.7)
s.path('M900,608 V632',INK,2.7)

# Shared policy, unchanged deployment path.
s.rect(720,636,360,99,'#F5F8FA','#B7CAD8',20,1.6,True)
s.text(900,674,'One evolving policy',29,INK,True,'middle')
s.text(900,713,m('π','k'),32,INK,True,'middle',True)
s.path('M900,735 V769',INK,2.8)
s.rect(720,780,360,64,'white','#C2D0D9',16)
s.text(900,819,'Original inference at deployment',22,INK,False,'middle')

# The dotted upward arrows are temporal memory reuse, not additional policies.
s.rect(55,694,510,150,'#F4F8FD','#BFD3E6',20)
s.memory_pool(85,759,BLUE,'expert')
s.text(224,732,'Task-end expert memory',26,BLUE,True)
s.text(224,768,'Demonstrations → RA-Coreset',22,INK)
s.text(224,800,'Coverage + random anchors',21,MUTED)
s.path('M310,694 V587',BLUE,2.3,True,'5 7')
s.text(335,655,'Reuse in task k + 1',21,BLUE,False,'start')

s.rect(1235,694,510,150,'#F0F9F5','#B7DACE',20)
s.memory_pool(1265,759,TEAL,'history')
s.text(1404,732,'Task-end success memory',26,TEAL,True)
s.text(1404,768,'Fixed rollouts → retain successes',21.5,INK)
s.text(1404,800,'Histories + frozen references',21,MUTED)
s.path('M1490,694 V587',TEAL,2.3,True,'5 7')
s.text(1515,655,'Reuse in task k + 1',21,TEAL,False,'start')
s.path('M1080,686 H1176 Q1190,686 1190,700 V758 H1230',TEAL,2.6)

s.text(900,883,'Support-weighted retention uses a fixed denominator · First task: base updates only',21,MUTED,False,'middle')
s.footer('DEBR · Two experience memories, one coordinated policy',936)
s.disclosure(978)
s.finish(HERE/'DEBR_B.svg')
print('DEBR_B.svg ready')

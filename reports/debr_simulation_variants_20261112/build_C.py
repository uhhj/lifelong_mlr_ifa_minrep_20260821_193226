"""C: two scene sequences on a horizontal task-to-task learning ribbon."""
from pathlib import Path
from figkit import *

s=Figure()
s.pill(70,35,258,'DUAL EXPERIENCE',INK,'#F0F4F7',21)
s.text(1730,60,'Learn from demonstrations. Remember successful behavior.',28,INK,True,'end')

# Two representative, source-verified sequences, without a claim of paired tasks.
s.text(70,121,'Expert demonstrations',32,BLUE,True)
s.film(70,160,160,'expert',BLUE,24)
s.text(70,411,'Autonomous successful experience',32,TEAL,True)
s.film(70,450,160,'success',TEAL,24)

# The source sequences feed their complementary learning paths.
s.path('M610,240 H670',BLUE,3)
s.path('M610,530 H670',TEAL,3)
s.rect(675,160,295,180,'url(#blueWash)','#BED3E8',20,1.4,True)
s.text(698,194,'CADL',32,BLUE,True)
s.text(698,225,'Execution supervision',23,INK)
s.memory_pool(701,271,BLUE,'expert')
s.text(823,277,'Expert memory',21,BLUE,True)
s.text(823,307,'+ current demos',21,INK)

s.rect(675,450,295,180,'url(#tealWash)','#B7DCD3',20,1.4,True)
s.text(698,484,'SHBR',32,TEAL,True)
s.text(698,515,'Support-weighted retention',22,INK)
s.memory_pool(701,561,TEAL,'success')
s.text(823,567,'Success history',21,TEAL,True)
s.text(823,596,m('q','j')+' = '+m('n','j')+sup('+')+' / '+m('N','roll'),23,INK,False,raw=True)

# Gate: enough detail to retain the scientific identity without another maze.
s.rect(1050,160,680,470,'url(#goldWash)','#DEC4A4',24,1.5,True)
s.text(1080,201,'Candidate-update gate',33,INK,True)
s.text(1080,235,'Same full state · Same batches · B = 10',23,MUTED)
s.line(1080,251,1700,251,'#E2CDB4',1.2)
s.rect(1080,275,250,95,'white','#BDCEE1',13,1.4)
s.text(1205,309,'Base C',26,BLUE,True,'middle')
s.text(1205,348,m('L','C'),29,INK,False,'middle',True)
s.rect(1370,275,330,95,'white','#B3D7CD',13,1.4)
s.text(1535,309,'Retention Q',26,TEAL,True,'middle')
s.text(1535,348,m('L','C')+' + '+m('λ','R')+m('L','SHBR'),27,INK,False,'middle',True)
s.path('M970,250 H1010 Q1028,250 1028,268 V323 H1080',BLUE,2.8)
s.path('M970,540 H1022 Q1035,540 1035,526 V389 H1535 V370',TEAL,2.8)
s.text(1080,415,'Fixed training probes',24,INK,True)
s.text(1160,455,m('BC','Q')+' ≤ '+m('BC','C'),25,INK,False,'middle',True)
s.text(1385,455,m('Exec','Q')+' ≤ '+m('Exec','C'),25,INK,False,'middle',True)
s.text(1610,455,m('Ret','Q')+' &lt; '+m('Ret','C'),25,INK,False,'middle',True)
s.line(1080,478,1700,478,'#E2CDB4',1.2)
s.text(1390,516,'All pass → Q; otherwise → C',27,INK,True,'middle')
s.text(1390,550,'Commit the selected full training state',23,INK,False,'middle')
s.text(1390,599,'First task: base updates only',21,MUTED,False,'middle')
s.path('M1500,630 V659',GOLD,2.6)
s.rect(1320,665,410,86,'white','#9EB1C2',18,1.5,True)
s.text(1525,700,'Single stage policy '+m('π','k'),29,INK,True,'middle',True)
s.text(1525,732,'Deployment: original inference',23,MUTED,False,'middle')

# Stage-end memory creation is kept out of the current-task optimization chain.
s.text(70,678,'AT TASK END',20,MUTED,True)
s.rect(70,703,540,102,'#F7FAFE','#BED3E8',17,1.4)
s.text(340,737,'Demonstrations → RA-Coreset → Expert memory',22,BLUE,True,'middle')
s.text(340,774,'Protected random anchors + coverage-aware exchange',21,MUTED,False,'middle')
s.path('M70,240 H40 V754 H70',BLUE,2)
s.rect(675,703,590,102,'#F4FBF8','#B7DCD3',17,1.4)
s.text(970,737,'Policy rollouts → Successes → History memory',23,TEAL,True,'middle')
s.text(970,774,'Fixed attempts; store histories and Gaussian references',21,MUTED,False,'middle')
s.path('M1320,708 H1295 V754 H1265',TEAL,2.3)

# Both memories join the next stage, rather than re-entering the finished stage.
s.path('M340,805 V878 H1708',BLUE,2.3,True)
s.path('M1245,805 V891 H1708',TEAL,2.3,True)
s.text(70,848,'Task k',26,INK,True)
s.text(1040,848,'Two memories, carried forward',25,INK,True,'middle')
s.text(1730,848,'Task k + 1',26,INK,True,'end')
s.footer('DEBR: Dual-Experience Behavioral Retention',937)
s.disclosure(975)
s.finish(Path(__file__).with_name('DEBR_C.svg'))
print('DEBR_C.svg written')

import fs from 'node:fs/promises';
async function png(path){const b=await fs.readFile(path);return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);}
const C={blue:'#1668B2',cyan:'#06B9D5',orange:'#EB8D23',green:'#67A53F',dark:'#172536',muted:'#566678'};
const FONT='Microsoft YaHei';
function text(s,x,y,w,h,t,size=24,bold=false,color=C.dark,align='left'){
 const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 a.text=t;a.text.style={fontSize:size,typeface:FONT,bold,color,alignment:align,verticalAlignment:'middle',insets:0};return a;
}
function rect(s,x,y,w,h,fill,line='none',dash='solid',radius=0){return s.shapes.add({geometry:radius?'roundRect':'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:line,width:line==='none'?0:1.6,style:dash},borderRadius:radius});}
function arrow(s,x,y,w=32){return s.shapes.add({geometry:'rightArrow',position:{left:x,top:y,width:w,height:16},fill:'#202B37',line:{fill:'none',width:0}});}
function base(s,sub,num){s.background.fill='#FFFFFF';text(s,0,21,1280,66,'4. 前期工作',49,false,'#000000','center');
 rect(s,0,101,1280,8,{type:'gradient',gradientKind:'linear',angleDeg:0,stops:[{offset:0,color:'#3B9EEB'},{offset:26000,color:'#06C5CB'},{offset:68000,color:'#363AD7'},{offset:100000,color:'#0875AD'}]});
 text(s,45,125,1190,51,'•  '+sub,33,true,'#111111');
 rect(s,45,687,1190,3,{type:'gradient',gradientKind:'linear',angleDeg:0,stops:[{offset:0,color:'#3CA8DA'},{offset:50000,color:'#A7CC4A'},{offset:100000,color:'#B91E50'}]});
 text(s,46,694,1090,18,'机器人终身学习 · MLR＋IFA复现与方法验证',14,false,C.muted);text(s,1190,691,44,23,String(num),17,false,C.muted,'right');
}
function summary(s,t){rect(s,85,589,1110,72,'#FFFFFF','#2C619A','solid',12);text(s,110,599,1060,52,t,25,true,'#152D49','center');}
function panel(s,x,w,col,head,body,tag){rect(s,x,203,w,335,'#FFFFFF',col,'dashed',13);text(s,x+10,219,w-20,45,head,29,true,col,'center');rect(s,x+15,274,w-30,2,col);text(s,x+12,310,w-24,136,body,26,false,C.dark,'center');rect(s,x+14,482,w-28,40,col);text(s,x+17,486,w-34,31,tag,22,true,'#FFFFFF','center');}
function table(s,x,y,w,h,vals,widths,size=24){const tb=s.tables.add({rows:vals.length,columns:vals[0].length,left:x,top:y,width:w,height:h,values:vals,columnWidths:widths});
tb.borders.assign({fill:'#D2DFEA',width:1,style:'solid'});
for(let r=0;r<vals.length;r++)for(let c=0;c<vals[0].length;c++){let cell=tb.getCell(r,c);cell.fill=r===0?'#E9F3FA':r%2?'#FFFFFF':'#F6FAFD';cell.text.style={fontSize:size,typeface:FONT,bold:r===0,color:r===0?'#1A527E':C.dark,alignment:c===0?'left':'center',verticalAlignment:'middle'};}
return tb;}
const NOTES={
base:'证据来源：execution/20261017_coreset_goal_three_seed/PROTOCOL.md；execution/20261018_ra_coreset_goal_seed42/PROTOCOL.md；execution/20261022_align_object_extension_goal_conditional/PROTOCOL.md；execution/20261025_ccbp_v4_full/PROTOCOL.md。基线是本项目原MLR+IFA（Random+IFA）复现成绩，不是论文表格原值。使用官方六任务预训练权重，后续Task6–9序贯训练；state=True、调度及ep99规则属于披露的实现设置，官方预训练state元数据未明确，不能声称逐位复现作者全部设置。RA只改回放成员，但IFA也会随真实buffer间接变化。ALIGN修正到a[u+1]且保留人口，通过mask屏蔽无效BC。S还包含C专家cost与路由辅助，不是仅RA+ALIGN增加一次成功筛选。',
ra:'证据来源：execution/20261018_ra_coreset_goal_seed42/PROTOCOL.md；execution/20261022_align_object_extension_goal_conditional/PROTOCOL.md；execution/20261025_ccbp_v4_full/GOAL_THREE_SEED_BASELINES.json与OBJECT_THREE_SEED_BASELINES.json。RA：q=ceil(N/2)，同seed Random有序名单前floor(q/2)为硬锚，原K-Center视觉/动作距离各0.5，缺失锚点最少交换，非锚删除按全体候选最大最近距离、均值及ID排序。ALIGN有效条件为唯一时间身份、8个真实未填充帧、下一动作存在，人口/IFA/步数保留，只有无效BC置零。结果为41/42/43开发种子描述性比较，非显著性结论。GOAL的ALIGN并非全面提升，主页面仅陈述有证据支持的OBJECT改善。',
s:'证据来源：execution/20261025_ccbp_v4_full/PROTOCOL.md；final_readonly_20261026_014654/REPORT.md、MECHANISM_REPORT.md、LOCAL_FINAL_AUDIT.json。S使用C主干（RA+ALIGN+执行专家cost+margin路由），增加成功轨迹历史上的固定获得时教师保持。GOAL seed41每任务10训练初态20–29，共40训练采集；正式0–19每cell20回合，共200。最多128窗口/task，实际128/128/115/128=499。配对C只读历史参考，没有额外训练采集，不是等交互资源。S最终Task6/7/8/9=45/100/100/95，C=40/100/100/90。S较C早期7_6降低10pp，预登记整体支持门S-C未通过。因此仅写最终保持与学习的初步收益，不称全面优于C。成功是整轨迹标签，不证明各步最优。O/S在筛选生效前已有部分输入SHA差异，不能将整个收益独立归因于成功筛选。S seed42/43仍待独立最终验收，当前不纳入结果。',
results:'证据来源：execution/20261025_ccbp_v4_full/GOAL_THREE_SEED_BASELINES.json、OBJECT_THREE_SEED_BASELINES.json；execution/20261026_ccbp_v4_s_goal42_43/REFERENCE_RESULTS.json。三seed为41/42/43，均值±样本标准差ddof1，每格20正式episode。AUC先对每个任务的已见阶段平均，再跨任务平均；FinalAll为最后阶段4任务平均。数据均是项目复现，非论文报告原值。OBJECT原seed43采用冻结RA协议指定repeat1，非按结果选择；首次作为敏感性保留，改用首次原基线AUC均值70.59，RA73.23，RA+ALIGN80.17。历史同seed初态字节一致，动作噪声未证实逐轨迹配对。RA并非每个旧任务均提升，ALIGN的GOAL均值与保持结果未在本成果聚焦页展示，不应将OBJECT结论推广全部套件。S单seed与三seed均值分开展示。研究仍为探索性，不作显著性或硬保护保证。'
};
export async function first({presentation}){
 presentation.theme.defaultFont=FONT;const s=presentation.slides.getItem(0);base(s,'4.1 基线建立与方法改进',1);
 const xs=[40,287,534,781,1028],w=212;
 panel(s,xs[0],w,C.cyan,'MLR＋IFA','复现原有方法\n搭建实验平台','基线复现');
 panel(s,xs[1],w,C.orange,'RA-Coreset','筛选代表样本\n兼顾场景与动作','回放选样');
 panel(s,xs[2],w,'#D0A300','TASA','对齐观测与动作\n提供准确监督','监督对齐');
 panel(s,xs[3],w,C.green,'SHBR','回放成功经验\n保持已学技能','技能保持');
 panel(s,xs[4],w,'#388052','闭环验证','机器人仿真测试\n评估学习与保持','效果验证');
 for(let i=0;i<4;i++)arrow(s,xs[i]+w+5,360,25);
 summary(s,'完成基线复现，从回放选样、监督对齐与成功经验保持三个方面开展改进');s.speakerNotes.text=NOTES.base+' 本页为方法概览。TASA对应历史ALIGN，SHBR对应历史S；后续证据页保留原实验标识以便追溯。';
 return {slide:1};
}
export async function middle({presentation}){
 const s=presentation.slides.add();base(s,'4.2 RA与ALIGN：回放选样及监督对齐',2);
 text(s,48,197,1150,36,'RA-Coreset：在有限回放预算内兼顾几何覆盖与随机保留',27,true,C.orange);
 const xa=[50,350,650,950];const labs=[['原任务窗口','视觉特征＋专家动作'],['K-Center集合','优先覆盖多样样本'],['加入随机硬锚','保留原Random部分成员'],['RA回放集合','最少交换，配额不变']];
 for(let i=0;i<4;i++){rect(s,xa[i],248,275,100,'#FFF9EF',C.orange,'dashed',8);text(s,xa[i]+10,258,255,34,labs[i][0],25,true,C.dark,'center');text(s,xa[i]+10,302,255,27,labs[i][1],20,false,C.muted,'center');if(i<3)arrow(s,xa[i]+280,290,15);}
 text(s,50,366,1175,29,'固定 q＝⌈N/2⌉，其中 ⌊q/2⌋ 个为随机硬锚点；只调整入库成员，沿用原训练主干',21,false,C.muted);
 text(s,48,416,1150,36,'ALIGN：依据真实观测时序，使用与决策对应的下一步动作监督',27,true,C.blue);
 rect(s,50,468,355,77,'#F0F8FD',C.blue,'dashed',8);text(s,60,480,335,52,'真实8帧观测历史\n末帧索引 u',23,false,C.dark,'center');arrow(s,420,496,35);
 rect(s,470,468,270,77,'#F0F8FD',C.blue,'dashed',8);text(s,480,480,250,52,'监督目标 a[u＋1]\n时间身份唯一',23,false,C.dark,'center');arrow(s,755,496,35);
 rect(s,805,468,420,77,'#F0F8FD',C.blue,'dashed',8);text(s,820,480,390,52,'mask核验后计算BC\n训练人口、IFA与更新预算保留',22,false,C.dark,'center');
 summary(s,'RA在GOAL与OBJECT均提高三种子平均AUC，ALIGN进一步提高OBJECT的平均AUC');s.speakerNotes.text=NOTES.ra;
 const t=presentation.slides.add();base(t,'4.3 S：成功执行历史上的固定教师保持',3);
 const stages=[['任务获得','固定当前任务模型\n完成10次训练采集'],['成功历史筛选','仅保留成功轨迹窗口\n每任务最多128个'],['固定教师缓存','真实1–8帧输入历史\n保存执行高斯分布'],['后续任务保持','旧任务均衡抽取锚点\n约束执行分布与路由']];
 for(let i=0;i<4;i++){let x=50+i*300;rect(t,x,207,275,146,'#FFFFFF',i<2?C.green:C.blue,'dashed',11);text(t,x+10,220,255,37,stages[i][0],27,true,i<2?C.green:C.blue,'center');text(t,x+15,269,245,67,stages[i][1],22,false,C.dark,'center');if(i<3)arrow(t,x+280,270,15);}
 text(t,50,371,1180,30,'学习主干：RA＋ALIGN＋C组专家监督与原路由；新增成功历史上的固定教师保持项',22,true,C.dark,'center');
 text(t,50,419,1180,31,'GOAL seed41 · 最后阶段闭环成功率',25,true,C.green);
 table(t,50,461,1175,92,[['方法','旧任务平均','Task8','Task9','全部任务平均'],['S','81.67%','100%','95%','85.00%']],[315,215,215,215,215],24);
 text(t,52,562,1170,20,'独立验收完成。每格20正式回合；另有40次训练采集。seed42、43扩展验证正在进行。',17,false,C.muted);
 summary(t,'S在seed41上获得积极的最终保持与新任务学习结果，为成功执行历史回放提供初步依据');t.speakerNotes.text=NOTES.s;
 return {slides:[2,3]};
}
export async function last({presentation}){
 const s=presentation.slides.add();base(s,'4.4 已完成实验与前期工作成果',4);
 text(s,50,197,1170,32,'GOAL / OBJECT · seed41、42、43的已完成闭环结果',27,true,C.blue);
 table(s,50,245,1175,227,[['套件','方法','AUC（%）','最终成功率（%）'],['GOAL','原MLR＋IFA','76.35 ± 6.38','76.67 ± 9.21'],['GOAL','＋RA','81.98 ± 3.00','80.83 ± 5.05'],['OBJECT','原MLR＋IFA','68.30 ± 3.12','65.42 ± 5.20'],['OBJECT','＋RA','73.23 ± 1.78','69.58 ± 5.64'],['OBJECT','＋RA＋ALIGN','80.17 ± 3.66','75.83 ± 3.82']],[180,375,310,310],23);
 text(s,50,487,1170,35,'RA：GOAL平均AUC提高5.63个百分点；OBJECT提高4.93个百分点',25,true,C.orange,'center');
 text(s,50,529,1170,35,'ALIGN：在RA基础上，OBJECT平均AUC进一步提高6.94个百分点',25,true,C.green,'center');
 summary(s,'形成可复核的基线和跨种子比较结果，并以S的初步结果推进执行历史保持研究');
 text(s,52,666,1170,16,'均值±样本标准差，n＝3；最终成功率为最后阶段四任务平均。本表为项目复现成绩。',14,false,C.muted,'center');
 s.speakerNotes.text=NOTES.results;return {slide:4};
}
export async function realframes({presentation,input}){
 const s=presentation.slides.add();base(s,'4.5 真实闭环执行画面',5);
 const windows=['first_mean_delta','first_mode_switch','terminal'];const steps=[0,65,77];const xs=[205,435,665];
 text(s,45,195,1200,31,'GOAL Task8：将碗放到盘子上',28,true,C.blue);
 for(let j=0;j<3;j++){
  text(s,xs[j],235,165,27,`第 ${steps[j]} 步`,22,true,C.dark,'center');
  for(let i=0;i<2;i++){
   const cam=i?'robot0_eye_in_hand_image':'agentview_image';
   s.images.add({blob:await png(input.assetDir+'/'+windows[j]+'_'+cam+'.png'),contentType:'image/png',alt:`真实ALIGN GOAL Task8 seed41 Stage9 初态3 ${cam} step${steps[j]}`,fit:'contain',position:{left:xs[j],top:270+i*167,width:157,height:157}});
  }
 }
 text(s,45,310,145,68,'外部相机\nAgent view',22,true,C.dark,'center');text(s,45,479,145,68,'腕部相机\nEye-in-hand',22,true,C.dark,'center');
 text(s,905,252,322,46,'已保存成功示例',27,true,C.green);
 text(s,905,319,320,198,'方法：ALIGN\n种子：41\n模型阶段：Stage9\n任务：GOAL Task8\n初态编号：3\n回合结果：成功',23,false,C.dark);
 text(s,905,535,320,43,'来自已有诊断rollout\n原始双相机RGB关键帧',19,false,C.muted);
 rect(s,85,613,1110,48,'#FFFFFF','#2C619A','solid',12);text(s,105,620,1070,33,'真实执行画面与回合记录、时间索引和源文件哈希一一对应',24,true,'#152D49','center');
 s.speakerNotes.text='图片来源：REAL_FRAME_PROVENANCE.json及assets_private内6幅PNG。2026-10-26只读提取，逐文件源NPZ SHA与原PRIVATE_MANIFEST一致，仅导出原始RGB数组的无损PNG，不下载原NPZ/action/state。ALIGN_s41_driver9_g0，初态3，成功，77策略步；图示为0、65、77步稀疏关键帧，不是连续视频，也不是S的rollout。选取此前已审阅记录中最小成功初态编号，仅用于展示任务与执行，不估计成功率。原始图像128x128，页面等比放大，未生成、重绘、增强或改变画面。';return {slide:5};
}
export async function training({presentation,input}){
 const s=presentation.slides.add();base(s,'4.6 训练记录与实验数据',6);
 text(s,45,195,1190,33,'S · GOAL seed41 · 完整Task6–9训练',28,true,C.blue);
 s.images.add({blob:await png(input.curvePath),contentType:'image/png',alt:'从S41原始400轮训练标量生成的BC NLL和Task9保持损失曲线，未平滑',fit:'contain',position:{left:40,top:245,width:810,height:305}});
 text(s,894,247,325,35,'原始日志字段节选',25,true,C.green);
 rect(s,875,300,354,248,'#F4F8FB','#B6CCDE','solid',8);
 const log='arm: S\nstage: 9\nepoch: 99\nsteps: 76100\nbatches: 337\nvalid: 3218\ninvalid: 147\nlr_after: 0.0';
 const q=text(s,897,314,310,222,log,21,false,C.dark);q.text.typeface='Consolas';
 text(s,45,557,1180,31,'曲线由实际epoch记录复算：记录总量 ÷ 当轮batch数。原始JSONL与完整CSV随材料保存。',20,false,C.muted,'center');
 summary(s,'已完成400轮训练与76,100次主更新，形成训练日志、逐回合结果及可追溯指标表');
 s.speakerNotes.text='来源：execution/20261025_ccbp_v4_full/final_readonly_20261026_014654/EVIDENCE.json中的EPOCHS_S.json，原文本SHA e96e5d45898f2dc55da6b8ff1d6a2a06ba16e031c04ee155bd2d19b0210df609。日志节选来自check_readonly_20261026_005436/EVIDENCE.json的train_S.log，第97/98/99轮原始行逐字段核对，与最终epoch账目完全一致。epoch为0开始，99表示完成该阶段100轮。曲线为BC NLL和保持项批均值，未平滑。连续分布NLL可为负，训练loss不等同闭环成功率。仅400正式epoch与76100主更新，准入更新与额外梯度计算另列，不称等算力。';return {slide:6};
}

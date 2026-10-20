import fs from "node:fs/promises";
import path from "node:path";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";
const root=process.cwd(), dir=path.join(root,"execution/20261020_all_methods_summary");
const out=path.join(root,"outputs/all_methods_20261020");
const d=JSON.parse(await fs.readFile(path.join(dir,"ALL_RESULTS.json"),"utf8"));
await fs.mkdir(out,{recursive:true});
const wb=Workbook.create(), names=["方法汇总","单次实验","阶段任务矩阵","相对基线","原型专项指标","范围与口径","来源与哈希"];
const sheets=Object.fromEntries(names.map(n=>[n,wb.worksheets.add(n)]));
const col=n=>{let s="";for(n++;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s};
const m=["AUC","FWT","NBT","OldFinal","FinalAll"];
const rowOf=Object.fromEntries(d.runs.map((r,i)=>[r.id,i+6]));
function table(name,title,note,headers,rows,widths={}){
 const s=sheets[name],last=col(headers.length-1),end=rows.length+5;
 s.showGridLines=false;
 s.getRange(`A1:${last}${end}`).format.font={name:"Arial",size:11};
 s.getRange(`A1:${last}${end}`).format.verticalAlignment="center";
 const topEnd=name==="来源与哈希"?"C":name==="原型专项指标"||name==="范围与口径"?"E":"J";
 s.getRange(`A1:${topEnd}1`).merge();s.getRange("A1").values=[[title]];
 s.getRange("A1").format.font={name:"Arial",size:20,bold:true,color:"#15324F"};
 s.getRange("A1").format.rowHeight=38;
 s.getRange(`A2:${topEnd}3`).merge();s.getRange("A2").values=[[note]];
 s.getRange(`A2:${topEnd}3`).format.wrapText=true;
 s.getRange(`A2:${topEnd}3`).format.rowHeight=27;
 s.getRange(`A2:${topEnd}3`).format.font={name:"Arial",size:11,color:"#57657A"};
 s.getRange(`A5:${last}${end}`).values=[headers,...rows];
 s.getRange(`A5:${last}5`).format.fill="#15324F";
 s.getRange(`A5:${last}5`).format.font={name:"Arial",size:11,bold:true,color:"#FFFFFF"};
 s.getRange(`A5:${last}5`).format.wrapText=true;
 s.getRange(`A5:${last}5`).format.rowHeight=38;
 s.getRange(`A6:${last}${end}`).format.rowHeight=29;
 for(let i=0;i<headers.length;i++)s.getRange(`${col(i)}:${col(i)}`).format.columnWidth=widths[i]||15;
 for(let i=6;i<=end;i+=2)s.getRange(`A${i}:${last}${i}`).format.fill="#F1F5F9";
 const t=s.tables.add(`A5:${last}${end}`,true,"ResearchTable"+(names.indexOf(name)+1));
 t.showFilterButton=true;
 s.freezePanes.freezeRows(5);
 return s;
}
const summary=table(names[0],"Coreset 路线 · 全方法指标总览",
 "截至2026-10-20｜均值只含“seeds”列所列实验；SD为样本标准差(ddof=1)，单seed留空。不同seed组合不直接排名；首次与repeat1完全分列。",
 ["套件","方法","版本","seeds","n",...m.flatMap(k=>[k+"均值(%)",k+" SD(pp)"]),"正式记录数"],
 d.summaries.map(x=>[x.suite,x.method,x.version,x.seeds,x.n,...m.flatMap(k=>[x[k],x[k+"_SD"]]),x.episodes]),
 {0:12,1:36,2:18,3:22,4:7});
summary.getRange(`F6:O${d.summaries.length+5}`).setNumberFormat("0.00");
const single=table(names[1],"逐实验结果与计算预算",
 "51个完整四阶段实验；每个结果200条正式记录。校准方法的epoch/step仅计新增校准预算，不含只读父模型训练。空白耗时表示未在本轮统一核对，不代表0。",
 ["ID","套件","方法","seed","版本",...m.map(k=>k+"(%)"),"最终Task6(%)","最终Task7(%)","最终Task8(%)","最终Task9(%)","训练类型","epoch","逻辑step","正式记录","CP数","基线ID","训练分钟","评估分钟","状态","来源ID","备注"],
 d.runs.map(r=>[r.id,r.suite,r.method,r.seed,r.version,...m.map(k=>r.metrics[k]),...r.matrix[3],r.kind,r.epochs,r.optimizer_steps,r.episodes,r.checkpoints,r.baseline_id,r.train_minutes,r.eval_minutes,r.status,r.source,r.notes]),
 {0:9,1:11,2:36,3:8,4:18,14:25,22:17,23:11,24:95});
single.getRange("F6:N56").setNumberFormat("0.00");
single.getRange("U6:V56").setNumberFormat("0.0");
single.getRange("Y6:Y56").format.wrapText=true;
single.getRange("A6:Y56").format.rowHeight=45;
const cells=[];for(let s=0;s<4;s++)for(let t=0;t<=s;t++)cells.push([s,t]);
table(names[2],"全部阶段 × 任务成功率",
 "列名S7/T6表示学完Task7后评估Task6。每格分母20，数值单位%；未学习任务不设空占位。AUC等五项指标从这10个有效单元独立复算。",
 ["ID","套件","方法","seed","版本",...cells.map(([s,t])=>`S${s+6}/T${t+6}(%)`),"来源ID"],
 d.runs.map(r=>[r.id,r.suite,r.method,r.seed,r.version,...cells.map(([s,t])=>r.matrix[s][t]),r.source]),
 {0:9,1:11,2:36,3:8,4:18}).getRange("F6:O56").setNumberFormat("0.0");
// Formula-derived metrics remain traceable to the underlying stage/task table.
for(const r of d.runs){
 const i=rowOf[r.id],q=c=>`'阶段任务矩阵'!${c}${i}`;
 single.getRange(`F${i}:J${i}`).formulas=[[
 `=(AVERAGE(${q("F")},${q("G")},${q("I")},${q("L")})+AVERAGE(${q("H")},${q("J")},${q("M")})+AVERAGE(${q("K")},${q("N")})+${q("O")})/4`,
 `=AVERAGE(${q("F")},${q("H")},${q("K")},${q("O")})`,
 `=((${q("F")}-AVERAGE(${q("G")},${q("I")},${q("L")}))+(${q("H")}-AVERAGE(${q("J")},${q("M")}))+(${q("K")}-${q("N")}))/3`,
 `=AVERAGE(${q("L")}:${q("N")})`,`=AVERAGE(${q("L")}:${q("O")})`]];
}
for(let i=0;i<d.summaries.length;i++){
 const x=d.summaries[i];
 for(let k=0;k<5;k++){
  const refs=x.run_ids.map(id=>`'单次实验'!${col(k+5)}${rowOf[id]}`).join(",");
  summary.getRange(`${col(5+2*k)}${i+6}`).formulas=[[`=AVERAGE(${refs})`]];
  if(x.n>1)summary.getRange(`${col(6+2*k)}${i+6}`).formulas=[[`=STDEV(${refs})`]];
 }
}
table(names[3],"对应基线的配对差值",
 "全部差值=方法−所列基线，单位pp。AUC/FWT/OldFinal/FinalAll正数更好；NBT负数更好。ZR-v1 seed43主口径repeat1及首次敏感性口径同时保留。",
 ["实验ID","套件","方法","seed","版本/口径","基线ID","基线",...m.map(k=>"Δ"+k+"(pp)"),"备注"],
 d.comparisons.map(r=>[r.run_id,r.suite,r.method,r.seed,r.version,r.baseline_id,r.baseline,...m.map(k=>r[k]),r.note]),
 {0:10,1:11,2:36,3:8,4:26,5:11,6:34,12:95}).getRange(`H6:L${d.comparisons.length+5}`).setNumberFormat("+0.00;-0.00;0.00");
table(names[4],"原型、离线诊断及选择器专项指标",
 "原字段名保留，单位见字段（fraction为0–1，percent为0–100；seconds为秒）。没有完整闭环的方案不填AUC。SAZR正式12/20来自被拒绝后回退的父模型，不能算候选残差收益。",
 ["方法","套件","seed","原始指标字段","数值","文本值/状态","适用范围","来源ID"],
 d.special.map(r=>[r.method,r.suite,r.seed,r.metric,r.numeric,r.text,r.scope,r.source]),
 {0:34,1:11,2:8,3:80,4:21,5:85,6:55,7:11});
table(names[5],"纳入范围与比较边界",
 "AUC=各任务从刚学完到最终阶段成功率的时间平均，再对4任务平均；FWT=对角线平均；NBT=各旧任务相对刚学完的平均下降，再对3旧任务平均。OldFinal=最终旧3任务平均；FinalAll=最终4任务平均。",
 ["方法/范围","套件","seeds","类别","状态","说明","来源ID"],
 d.scope.map(r=>[r.method,r.suite,r.seeds,r.category,r.status,r.notes,r.source]),
 {0:50,1:17,2:24,3:25,4:35,5:105,6:28});
table(names[6],"可追溯来源与原文件保护",
 "每个源文件在汇总前后SHA256一致；sources目录保存只读轻量副本。表内来源ID对应本表；不含模型、latent、buffer载荷或原动作。仅统计已有结果，无新增训练或环境执行。",
 ["来源ID","原始相对路径","SHA256","字节数","轻量副本"],
 d.sources.map(s=>[s.id,s.relative,s.sha256,s.bytes,s.copy]),
 {0:11,1:130,2:75,3:14,4:26});
wb.recalculate();
const verification=[];
for(const r of d.runs){
 const values=single.getRange(`F${rowOf[r.id]}:J${rowOf[r.id]}`).values[0];
 for(let k=0;k<5;k++)if(Math.abs(values[k]-r.metrics[m[k]])>1e-8)throw Error(`Metric mismatch ${r.id}/${m[k]}: ${values[k]}`);
}
for(let i=0;i<d.summaries.length;i++){
 const values=summary.getRange(`F${i+6}:O${i+6}`).values[0],x=d.summaries[i];
 for(let k=0;k<5;k++){
  if(Math.abs(values[k*2]-x[m[k]])>1e-8)throw Error("Mean mismatch");
  if(x.n>1&&Math.abs(values[k*2+1]-x[m[k]+"_SD"])>1e-8)throw Error("SD mismatch");
 }
}
verification.push(await wb.inspect({kind:"table",range:"方法汇总!A5:J8",include:"values,formulas",tableMaxRows:4,tableMaxCols:10,maxChars:6000}));
verification.push(await wb.inspect({kind:"match",searchTerm:"#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!",options:{useRegex:true,maxResults:30}}));
await fs.mkdir(path.join(out,"previews"),{recursive:true});
for(const name of names){
 const range=name==="来源与哈希"?"A1:C10":name==="原型专项指标"?"A1:E11":name==="范围与口径"?"A1:E12":"A1:J13";
 const blob=await wb.render({sheetName:name,range,scale:1,format:"png"});
 await fs.writeFile(path.join(out,"previews",name+".png"),new Uint8Array(await blob.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(path.join(out,"Coreset全方法指标总表.xlsx"));
await fs.writeFile(path.join(dir,"WORKBOOK_CHECK.json"),JSON.stringify({status:"PASS",sheets:names,formula_run_metrics_checked:255,summary_mean_sd_checked:true,inspection:verification},null,2));
console.log(JSON.stringify({status:"PASS",runs:d.runs.length,sheets:names,output:path.join(out,"Coreset全方法指标总表.xlsx")}));

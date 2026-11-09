from pathlib import Path
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import re
from zipfile import ZipFile
from docx import Document
from docx.table import Table
from docx.oxml.ns import qn

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=Path('C:/Users/Lenovo/Desktop/论文/论文写作/Coverage-Aware Replay with Support-Calibrated Execution Retention for Lifelong Robot Imitation Learning20261107 - 副本.docx')
OUTPUT=SOURCE.with_name('Coverage-Aware Replay with Support-Calibrated Execution Retention for Lifelong Robot Imitation Learning20261107 - RA选样消融修订.docx')
DATA=ROOT/'execution/20261109_random_kcenter_ra_tables/RESULTS.json'
j=json.loads(DATA.read_text(encoding='utf8'))
assert j['status']=='PASS_18_RUN_READONLY_RECOMPUTATION'
for rel,h in j['source_sha256'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h
d=Document(SOURCE)
assert len(d.tables)==4
source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()

def text_replace(p,old,new):
    """Replace ordinary text without destroying OMML, fields or run formatting."""
    nodes=list(p._p.iter(qn('w:t')))
    content=''.join(n.text or '' for n in nodes)
    if old in content:
        start=content.index(old);end=start+len(old);offset=0;first=True
        for n in nodes:
            t=n.text or '';a=offset;b=a+len(t);offset=b
            if b<=start or a>=end:continue
            lo=max(start-a,0);hi=min(end-a,len(t))
            n.text=t[:lo]+(new if first else '')+t[hi:];first=False
        content=''.join(n.text or '' for n in nodes)

# Existing captions and references shift together, including references after the insertion.
number_changes=[]
for i,p in enumerate(d.paragraphs):
    before=p.text
    # Captions and references split the word and number across runs.
    # Descending order prevents replacement of a number just incremented.
    for n in [4,3,2]:text_replace(p,'表'+str(n),'表'+str(n+1))
    text_replace(p,'4.4.1 组件消融','4.4.2 组件消融')
    text_replace(p,'4.4.2 优化器预算匹配','4.4.3 优化器预算匹配')
    if before!=p.text:number_changes.append(dict(paragraph=i,before=before,after=p.text))
intro=next(p for p in d.paragraphs if p.text.startswith('本节在LIBERO-OBJECT'))
text_replace(intro,'随后通过逐步组件消融、优化器预算匹配和保持系数分析','随后通过示范记忆选样消融、逐步组件消融、优化器预算匹配和保持系数分析')
comp=next(p for p in d.paragraphs if p.text.startswith('对比方法：'))
text_replace(comp,'组件分析以RA-Coreset为起点','选样消融在MLR＋IFA基础上比较Random、K-Center与RA-Coreset；组件分析以RA-Coreset为起点')
anchor=next(p for p in d.paragraphs if p.text=='4.4.2 组件消融')

new_text=[
 ('DEBR Unified H3','4.4.1 示范记忆选样策略消融'),
 ('DEBR Unified Body','为考察RA-Coreset中随机锚定与覆盖感知交换的作用，我们在MLR＋IFA基础上比较三种示范记忆选择策略：均匀随机选择（Random）、贪心覆盖选择（K-Center）以及本文的RA-Coreset（表中记为RA）。三种方案采用相同的任务记忆配额、数据来源、模型初始化、训练预算与闭环评估协议，均不加入ALIGN、执行分量监督或SHBR。K-Center对应RA-Coreset的初始覆盖集合；RA在此基础上保护随机锚点并进行等量交换，因此该对照用于评价示范记忆选择本身的作用。'),
 ('DEBR Caption','表2  示范记忆选样策略消融。三种方案均基于MLR＋IFA，结果为seed41、42、43的均值±样本标准差。FWT与AUC以百分比表示，NBT以百分点表示；每列最佳均值加粗。'),
 ('DEBR Unified Body','相对Random，RA在OBJECT和GOAL上的AUC分别提高4.93和5.63个百分点，FWT分别提高6.67和7.08个百分点；六个套件—种子组合的AUC均得到提升。这表明，在固定记忆配额下，将随机保留成分与覆盖导向选择结合，能够改善示范回放的任务获得与全过程闭环表现。'),
 ('DEBR Unified Body','与K-Center相比，RA在GOAL上的FWT由81.25%提高至83.75%，AUC由76.35%提高至81.98%，NBT由9.81降至2.59；三个种子的AUC均有提升。该结果支持在覆盖集合中保留随机锚点的实际价值。在OBJECT上，RA的FWT与AUC分别提高5.83和0.94个百分点，而K-Center取得更低的NBT，呈现不同的任务获得—后续保持取舍。由于NBT以各配置自身的任务获得成绩为参照，需结合FWT与AUC理解该差异。总体而言，RA在两个套件上均取得三种选样方案中最高的平均FWT与AUC，其相对K-Center的综合优势在GOAL上更为一致。'),
 ('DEBR Unified Body','上述对照采用既有的同协议运行；部分OBJECT基线与RA运行的GPU型号不同，因此这些结果用于评价选样策略的经验表现，不作完全硬件受控的单因素因果解释。后续组件消融进一步考察在RA示范记忆基础上加入执行监督与成功历史保持的作用。')]
inserted=[]
for style,text in new_text:
    p=anchor.insert_paragraph_before(text,style)
    inserted.append(p)
    if style=='DEBR Unified Body':p.paragraph_format.keep_with_next=False
    if style=='DEBR Caption':
        p.clear()
        p.add_run('表2  ').bold=True
        p.add_run(text[len('表2  '):])

# Clone the manuscript's own seven-column three-line table, preserving its fonts,
# merged headers, cell padding and rule style. Keep only three method rows.
tbl=deepcopy(d.tables[1]._tbl)
template_rows=list(tbl.findall(qn('w:tr')))
last=deepcopy(template_rows[-1])
for tr in template_rows[4:]:tbl.remove(tr)
tbl.append(last)
t=Table(tbl,d._body)
def set_cell(cell,text,bold=False):
    p=cell.paragraphs[0]
    rpr=deepcopy(p.runs[0]._r.rPr) if p.runs and p.runs[0]._r.rPr is not None else None
    for child in list(p._p):
        if child.tag!=qn('w:pPr'):p._p.remove(child)
    r=p.add_run(text)
    if rpr is not None:r._r.insert(0,rpr)
    r.bold=bold
    for other in cell.paragraphs[1:]:cell._tc.remove(other._p)
set_cell(t.cell(0,0),'Method',True)
set_cell(t.cell(0,1),'OBJECT',True)
set_cell(t.cell(0,4),'GOAL',True)
order=['FWT','NBT','AUC']
for col,metric in enumerate(order+order,1):set_cell(t.cell(1,col),metric+('↓' if metric=='NBT' else '↑'),True)
def fmt(v):return str(Decimal(str(v)).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP))
for row,method in enumerate(['Random','K-Center','RA'],2):
    set_cell(t.cell(row,0),method)
    for si,suite in enumerate(['OBJECT','GOAL']):
        for ki,k in enumerate(order):
            v=j['summary'][suite][method][k]
            allmeans=[j['summary'][suite][m][k]['mean_percent'] for m in ['Random','K-Center','RA']]
            best=min(allmeans) if k=='NBT' else max(allmeans)
            set_cell(t.cell(row,1+si*3+ki),fmt(v['mean_percent'])+'±'+fmt(v['sample_sd_pp']),v['mean_percent']==best)
for rowidx,row in enumerate(t.rows):
    for cell in row.cells:
        for p in cell.paragraphs:p.paragraph_format.keep_with_next=(rowidx<len(t.rows)-1)
inserted[2]._p.addnext(tbl)

assert len(d.tables)==5 and len(t.rows)==5
assert [[t.cell(r,c).text for c in range(7)] for r in [2,3,4]][0][0]=='Random'
assert len(d._element.xpath('//m:oMath'))==len(Document(SOURCE)._element.xpath('//m:oMath'))
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
d.save(OUTPUT)
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_hash
# All binary assets and relationships are preserved; editing is restricted to document XML.
with ZipFile(SOURCE) as a,ZipFile(OUTPUT) as b:
    media=[n for n in a.namelist() if n.startswith('word/media/')]
    assert all(a.read(n)==b.read(n) for n in media)
receipt=dict(status='EDITED_PENDING_VISUAL_QA',source=str(SOURCE),source_sha256=source_hash,
    output=str(OUTPUT),output_sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    data_source=str(DATA),data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
    inserted_text=[x[1] for x in new_text],number_changes=number_changes,
    tables_before=4,tables_after=5,math_count=len(d._element.xpath('//m:oMath')),media_preserved=len(media),
    numeric_table=[[t.cell(r,c).text for c in range(7)] for r in range(5)])
(HERE/'EDIT_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
(HERE/'INSERTED_SECTION.md').write_text('\n\n'.join(x[1] for x in new_text)+'\n',encoding='utf8')
print(json.dumps(dict(output=str(OUTPUT),tables=5,math_preserved=receipt['math_count']),ensure_ascii=False))

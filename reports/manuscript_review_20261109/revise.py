from pathlib import Path
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
from zipfile import ZipFile
import json, hashlib, re
from lxml import etree
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=Path('C:/Users/Lenovo/Desktop/论文/论文写作/Coverage-Aware Replay with Support-Calibrated Execution Retention for Lifelong Robot Imitation Learning20261107 - RA选样消融修订.docx')
OUTPUT=SOURCE.with_name('DEBR - A Dual-Experience Behavioral Retention Framework for Lifelong Robot Imitation Learning - 审稿修订20261109.docx')
DATA=ROOT/'execution/20261109_random_kcenter_ra_tables/RESULTS.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
original_hash=sha(SOURCE)
d=Document(SOURCE);p=list(d.paragraphs)
assert len(p)==187 and len(d.tables)==5
changes=[]

def txt(p):return ''.join(x.text or '' for x in p._p.xpath('.//w:t|.//m:t'))
def replace(p,old,new):
    nodes=list(p._p.iter(qn('w:t')));s=''.join(n.text or '' for n in nodes)
    if old not in s:return
    start=s.index(old);end=start+len(old);offset=0;first=True
    for n in nodes:
        t=n.text or '';a=offset;b=a+len(t);offset=b
        if b<=start or a>=end:continue
        lo=max(start-a,0);hi=min(end-a,len(t))
        n.text=t[:lo]+(new if first else '')+t[hi:];first=False
    changes.append(dict(kind='replace',old=old,new=new))
def reset(p,text):
    changes.append(dict(kind='paragraph',old=txt(p),new=text))
    p.clear();p.add_run(text)
def before(anchor,text,style='DEBR Unified Body'):
    x=anchor.insert_paragraph_before(text,style)
    x.paragraph_format.keep_with_next=False
    changes.append(dict(kind='insert',new=text))
    return x
def after(anchor,text,style='DEBR Unified Body'):
    x=before(anchor,text,style);anchor._p.addnext(x._p);return x

# Editable Word mathematics through Microsoft's MathML-to-OMML transform.
transform=etree.XSLT(etree.parse('C:/Program Files/Microsoft Office/root/Office16/MML2OMML.XSL'))
def el(tag,*children):return '<'+tag+'>'+''.join(children)+'</'+tag+'>'
def mi(x):return el('mi',x)
def mn(x):return el('mn',str(x))
def mo(x):return el('mo',x)
def row(*x):return el('mrow',*x)
def sub(a,b):return el('msub',a,b)
def sup(a,b):return el('msup',a,b)
def frac(a,b):return el('mfrac',a,b)
def par(x):return row(mo('('),x,mo(')'))
def math(x):
    result=transform(etree.fromstring(('<math xmlns="http://www.w3.org/1998/Math/MathML">'+x+'</math>').encode()))
    root=result.getroot()
    om=deepcopy(root) if root.tag==qn('m:oMath') else deepcopy(root.find('.//'+qn('m:oMath')))
    # Microsoft's transform emits empty n-ary operand boxes for standalone sum
    # operators. Use native sub/sup operators with the following expression intact.
    for n in list(om.iter(qn('m:nary'))):
        e=n.find(qn('m:e'))
        if e is not None and len(e)==0:
            s=OxmlElement('m:sSubSup');base=OxmlElement('m:e')
            r=OxmlElement('m:r');t=OxmlElement('m:t');t.text='∑';r.append(t);base.append(r);s.append(base)
            for tag in ['sub','sup']:
                a=n.find(qn('m:'+tag));s.append(deepcopy(a) if a is not None else OxmlElement('m:'+tag))
            n.getparent().replace(n,s)
    return om
def inline(p,x):p._p.append(math(x))
def equation_before(anchor,x):
    e=before(anchor,'','DEBR Unified Equation')
    e.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER
    e.paragraph_format.first_line_indent=Pt(0)
    e.paragraph_format.space_before=Pt(5);e.paragraph_format.space_after=Pt(7)
    e._p.append(math(x));return e
def dimension():return math(sub(mi('d'),mi('g')))[0]

# Hard errors and consistent notation; correctly rendered cardinalities are retained.
for x in p:
    replace(x,'第3.4.1—3.4.3节','第3.4.1—3.4.2节')
    replace(x,'第3.4.4节','第3.4.3节')
    replace(x,'SBHR','SHBR')
    replace(x,'双经验行为回放框架','双经验行为保持框架')
    replace(x,'ISCIL','IsCiL')
for idx in [60,78]:
    den=p[idx]._p.xpath('.//m:den')[0]
    assert den[0].tag==qn('m:d') and ''.join(den[0].itertext())=='g'
    den.replace(den[0],dimension())
dim=p[61]._p.xpath('.//m:oMath')[1]
dim.replace(dim[0],dimension())
replace(p[61],'为该组动作维数。','为该组动作维数，平移、旋转与夹爪分别为3、3和1。')
assert p[44]._p.xpath('.//m:begChr[@m:val="⌈"]')
assert p[50]._p.xpath('.//m:begChr[@m:val="|"]')
assert p[51]._p.xpath('.//m:begChr[@m:val="|"]')
assert p[65]._p.xpath('.//m:begChr[@m:val="["]')

# Definition of the original frozen selection metric, not a projected-space distance.
reset(p[40],'其中，d(x,s)为在当前任务候选集上标定的视觉—动作联合距离。记窗口x在时间位置t、视角v的单位归一化视觉潜在特征为f̂ₓₜᵥ，窗口末端原始动作的第h维为aₓₕ。两路视觉特征与七维动作分别定义距离：')
for token,expr in [('f̂ₓₜᵥ',sub(mi('f̂'),row(mi('x'),mi('t'),mi('v')))),('aₓₕ',sub(mi('a'),row(mi('x'),mi('h'))))]:
    # Replace inline Unicode shorthand by editable math while retaining sentence order.
    s=p[40].text;a,b=s.split(token,1);p[40].clear();p[40].add_run(a);inline(p[40],expr);p[40].add_run(b)
    if token=='f̂ₓₜᵥ':break
# The remaining action notation is made native without flattening the inserted math.
for n in list(p[40]._p.iter(qn('w:t'))):
    if 'aₓₕ' in (n.text or ''):
        run=n.getparent();a,b=n.text.split('aₓₕ');n.text=a
        om=math(sub(mi('a'),row(mi('x'),mi('h'))));run.addnext(om)
        rr=OxmlElement('w:r');tt=OxmlElement('w:t');tt.text=b;rr.append(tt);om.addnext(rr)
Dv=sub(mi('D'),mi('v'));Da=sub(mi('D'),mi('a'))
sumt=el('munderover',mo('∑'),row(mi('t'),mo('='),mn(1)),mi('L'))
sumv=el('munderover',mo('∑'),row(mi('v'),mo('='),mn(1)),mn(2))
dot=row(sup(sub(mi('f̂'),mi('xtv')),mi('⊤')),sub(mi('f̂'),mi('stv')))
equation_before(p[41],row(Dv,par(row(mi('x'),mo(','),mi('s'))),mo('='),mi('max'),par(row(mn(0),mo(','),mn(1),mo('−'),frac(mn(1),row(mn(2),mi('L'))),sumt,sumv,dot)),mo(',')))
sumh=el('munderover',mo('∑'),row(mi('h'),mo('='),mn(1)),mn(7))
adiff=frac(row(sub(mi('a'),mi('xh')),mo('−'),sub(mi('a'),mi('sh'))),sub(mi('ρ'),mi('h')))
equation_before(p[41],row(Da,par(row(mi('x'),mo(','),mi('s'))),mo('='),el('msqrt',frac(mn(1),mn(7)),sumh,sup(par(adiff),mn(2))),mo(','),mi('d'),par(row(mi('x'),mo(','),mi('s'))),mo('='),frac(Dv, row(mn(2),sub(mi('m'),mi('v')))),mo('+'),frac(Da,row(mn(2),sub(mi('m'),mi('a')))),mo('.')))
before(p[41],'动作尺度ρₕ由当前任务全部原始示范动作计算，优先取1.4826×MAD；若不大于10⁻¹²，则依次回退为IQR/1.349、极差和1。mᵥ与mₐ分别为固定随机种子42抽取的至多4,096对候选窗口中、对应距离大于10⁻¹²的中位数；无此类距离时取1。两项权重均为0.5，全部距离在原始潜在特征及动作空间中以双精度计算。选样使用窗口末端原始动作，监督标签的时间对齐则按第3.3.3节单独处理。')
# Native notation inside the explanation, using ordinary Unicode for prose exponents only.
metric_explain=d.paragraphs[d.paragraphs.index(p[41])-1] if False else None
before(p[45],'该配额沿用MLR＋IFA采用的50%名义示范存储比例[7]，其预算依据是LOTUS的每任务5条示范存储设置[12]。MLR＋IFA已比较10%、20%和50%的特征保留比例；本文固定50%以比较选样策略，并在窗口数为奇数时向上取整。锚点数量取该固定配额的一半并向下取整。')

# ALIGN is a descriptive name for the implemented alignment, not an invented official acronym.
reset(p[56],'当前示范与历史示范记忆共同构成训练批次。本文将观测—动作时间对齐（observation–action temporal alignment）记为ALIGN。对于末端索引为u的历史窗口，监督目标为同一轨迹内的下一动作aᵤ₊₁。记Iᵢ表示窗口轨迹身份唯一可确定，pᵣ表示位置r为填充或不存在，ν(r)表示该位置所属轨迹，则有效性标记为：')
equation_before(p[57],row(sub(mi('v'),mi('i')),mo('='),mi('𝟙'),mo('{'),sub(mi('I'),mi('i')),mo('='),mn(1),mo('∧'),sub(mi('p'),mi('r')),mo('='),mn(0),mo('∧'),mi('ν'),par(mi('r')),mo('='),mi('ν'),par(mi('u')),mo(','),mo('∀'),mi('r'),mo('∈'),mo('{'),mi('u'),mo('−'),mi('L'),mo('+'),mn(1),mo(','),mo('…'),mo(','),mi('u'),mo('+'),mn(1),mo('}'),mo('}'),mo('.')))
before(p[57],'因此，窗口内所有历史帧及下一动作均须有效，尤其要求末端u与u+1非填充且属于同一轨迹；身份不明确、含填充帧或不存在下一动作的窗口不参与行为克隆。该规则对应本文数据导出的时间关系，不改变成功历史中同一在线输入与教师输出的配对。')

# Probe mapping: old probe is a task-balanced full fixed-set average, not four random slots.
probe=before(p[98],'在固定当前任务探针上，行为克隆指标按式（5）计算，执行误差指标取式（6）的实际执行分量误差均值；两者均不加入路由项。旧历史探针采用式（11）—（13）的支持度加权单窗保持损失，对每个旧任务的固定探针窗口先取均值，再对全部旧任务等权平均，空缓存任务贡献为零。训练时式（13）的4槽采样是该任务均衡保持目标的采样实现，门控评价则使用阶段内固定的探针集合。')

# Sources and bibliographic citations, ordered by first appearance.
replace(p[2],'模仿学习通过','模仿学习[1]通过')
replace(p[3],'经验回放是','经验回放[2]是')
replace(p[4],'M2Distill通过','M2Distill[3]通过');replace(p[4],'HReDiff则','HReDiff[4]则')
replace(p[5],'LIBERO-GOAL和LIBERO-OBJECT两个数据集','LIBERO-GOAL和LIBERO-OBJECT两个任务套件[5]')
replace(p[11],'CRIL引入','CRIL[6]引入');replace(p[11],'MLR则','MLR＋IFA[7]则')
replace(p[11],'PolyTask与M2Distill','PolyTask[8]与M2Distill[3]');replace(p[11],'HReDiff进一步','HReDiff[4]进一步')
replace(p[12],'BUDS采用','BUDS[9]采用');replace(p[12],'TAIL则','TAIL[10]则');replace(p[12],'IsCiL进一步','IsCiL[11]进一步')
replace(p[12],'同时在测试时依赖明确的阶段或任务标识，在未知任务上的泛化表现存在短板。','不同方案分别通过任务条件或状态原型检索调用技能，其额外模块和路由方式也决定了知识共享的形式。')
replace(p[22],'MLR＋IFA的','MLR＋IFA[7]的')
replace(p[66],'基础框架的增量特征调整约束','MLR＋IFA[7]的增量特征调整约束')
replace(p[119],'沿用MLR＋IFA的','沿用MLR＋IFA[7]的')
after(p[119],'为在冻结前端的条件下统一预训练起点，本文采用MLR＋IFA公开发布的OBJECT与GOAL套件权重，避免另行预训练引入额外表征差异。LIBERO-OBJECT侧重相同任务模板下的物体变化，LIBERO-GOAL侧重相同物体与场景布局下的目标变化[5]，两者分别考察物体迁移与目标迁移。')
replace(p[120],'AdamW优化器','AdamW优化器[13]')
hp=after(p[120],'其余参数固定为历史窗口长度L=8、混合分量数M=5、路由间隔δ=0.1、执行误差权重λE=0.1、执行路由权重βE=0.1及保持路由权重βR=0.1；除保持系数敏感性实验外，各配置均使用上述设置。')
# Convert the newly supplied hyperparameter notation into inline OMML.
hp.clear()
for text,expr in [('其余参数固定为历史窗口长度',row(mi('L'),mo('='),mn(8))),('、混合分量数',row(mi('M'),mo('='),mn(5))),('、路由间隔',row(mi('δ'),mo('='),mn('0.1'))),('、执行误差权重',row(sub(mi('λ'),mi('E')),mo('='),mn('0.1'))),('、执行路由权重',row(sub(mi('β'),mi('E')),mo('='),mn('0.1'))),('及保持路由权重',row(sub(mi('β'),mi('R')),mo('='),mn('0.1')))]:
    hp.add_run(text);inline(hp,expr)
hp.add_run('；除保持系数敏感性实验外，各配置均使用上述设置。')
replace(p[121],'所有实验使用三个随机种子。','本地运行统一使用三个随机种子（41、42、43）。')
replace(p[121],'均值与标准差','均值与样本标准差')
replace(p[123],'遵循参考论文[1]','遵循MLR＋IFA[7]所述的持续学习评价协议')
replace(p[128],'Sequential、ER、BUDS、LOTUS、IsCiL、M2Distill、TAIL、MLR＋IFA及DEBR','Sequential[5]、ER[2,5]、BUDS[9]、LOTUS[12]、IsCiL[11]、M2Distill[3]、TAIL[10]、MLR＋IFA[7]及DEBR')
after(p[128],'外部对比方法采用文献[7]汇总的报告结果；本文实验包括按论文所述协议进行的MLR＋IFA复现、DEBR及其消融配置。MLR＋IFA复现对应选样消融中的Random配置。SHBR另使用每任务10次、初态与正式评估分离的成功历史采集；不使用该模块的配置不进行这部分交互。')
replace(p[130],'取得最高','在表1所列配置中取得最高')
replace(p[130],'在LIBERO-OBJECT上在表1所列配置中','在表1所列配置中，在LIBERO-OBJECT上')
after(p[134],'以MLR＋IFA原论文的报告值作为补充参照，其OBJECT与GOAL的AUC分别为79.4%和77.2%[7]，DEBR的对应均值为86.5%和87.3%。由此，DEBR相对本地复现及原论文报告值均具有更高的AUC；前者为本文统一评估协议下的比较，后者为跨研究的数值参照。')

# Table 2: recompute one-decimal presentation from original summary, never rounded inputs.
j=json.loads(DATA.read_text(encoding='utf8'))
assert j['status']=='PASS_18_RUN_READONLY_RECOMPUTATION'
for rel,h in j['source_sha256'].items():assert sha(ROOT/rel)==h
fmt=lambda x:str(Decimal(str(x)).quantize(Decimal('.1'),rounding=ROUND_HALF_UP))
def setcell(cell,text,bold=None):
    pp=cell.paragraphs[0];rp=deepcopy(pp.runs[0]._r.rPr) if pp.runs else None
    pp.clear();r=pp.add_run(text)
    if rp is not None:r._r.insert(0,rp)
    if bold is not None:r.bold=bold
t=d.tables[1]
setcell(t.cell(0,1),'LIBERO-OBJECT',True);setcell(t.cell(0,4),'LIBERO-GOAL',True)
for ri,m in enumerate(['Random','K-Center','RA'],2):
    setcell(t.cell(ri,0),'RA-Coreset' if m=='RA' else m)
    for si,suite in enumerate(['OBJECT','GOAL']):
        for ki,k in enumerate(['FWT','NBT','AUC']):
            v=j['summary'][suite][m][k];vs=[j['summary'][suite][a][k]['mean_percent'] for a in ['Random','K-Center','RA']]
            best=min(vs) if k=='NBT' else max(vs)
            setcell(t.cell(ri,1+3*si+ki),fmt(v['mean_percent'])+'±'+fmt(v['sample_sd_pp']),v['mean_percent']==best)
# Exact duplicate rows are unified across tables 1, 2 and 3.
for c in range(1,7):
    setcell(d.tables[0].cell(9,c),t.cell(2,c).text)
    setcell(d.tables[2].cell(2,c),t.cell(4,c).text,False)
setcell(d.tables[0].cell(6,0),'IsCiL')
# Latest user preference: no per-row provenance labels or provenance caption in Table 1.
# The experimental-method description supplies the source distinction once.
replace(p[137],'RA-Coreset（表中记为RA）','RA-Coreset')
replace(p[137],'；RA在此基础上','；RA-Coreset在此基础上')
replace(p[138],'seed41、42、43的','三个种子的')
for idx in [139,140]:
    replace(p[idx],'RA在','RA-Coreset在');replace(p[idx],'，RA的','，RA-Coreset的')
    replace(p[idx],'OBJECT','LIBERO-OBJECT');replace(p[idx],'GOAL','LIBERO-GOAL')
    for n in re.findall(r'\d+\.\d{2}',p[idx].text):replace(p[idx],n,fmt(n))
replace(p[140],'，RA在两个','，RA-Coreset在两个')
replace(p[140],'在GOAL上更为一致','在LIBERO-GOAL上更为一致')
replace(p[141],'在RA示范记忆','在RA-Coreset示范记忆')
after(p[146],'在LIBERO-OBJECT上，DEBR的FWT与NBT标准差分别为8.5和11.6个百分点，而AUC为86.5±2.2%。这说明三个种子的获得阶段成绩及后续变化幅度存在差异，但全过程综合指标更为集中。不同种子的门控选择与学习轨迹同时变化，当前结果不能将上述离散程度单独归因于门控接受比例。')
# Brief shared boundary rather than repeated audit prose.
after(p[181],'本研究复用公开预训练权重与既有基线结果，部分运行涉及不同GPU硬件，参数敏感性结果因此反映当前实验条件下的经验变化，不作为完全硬件受控的单因素因果估计。评估套件在方法开发中被重复使用，公开预训练权重的完整数据来源尚未独立闭合核验；这些结果属于开发阶段证据，而非独立留出集上的最终确认。')

# Front matter, following the manuscript's existing typography.
p[0].style=d.styles['Title'];p[0].paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER
for r in p[0].runs:r.font.color.rgb=RGBColor(0,0,0);r.font.size=Pt(17);r.font.name='Times New Roman';r.bold=True
p[0].paragraph_format.space_after=Pt(12)
p[1].paragraph_format.page_break_before=True
zh='终身机器人模仿学习需要在有限记忆下持续获得新技能并保留旧技能。仅回放专家示范，难以直接约束策略在自主成功执行状态上的行为变化。本文提出双经验行为保持框架DEBR，协同利用专家示范与策略自主成功经验。随机锚定核心集RA-Coreset在覆盖导向选择中引入受保护的随机锚点，通过固定配额下的覆盖感知交换组织示范记忆；成功历史行为保持SHBR缓存成功执行时的潜在历史及固定高斯行为参考，并依据成功轨迹支持度调节保持强度。为协调保持与新任务学习，候选更新门控从相同训练状态比较基础学习与保持增强两类更新，仅在当前任务拟合不劣且旧历史保持更优时接受增强候选。在LIBERO-OBJECT与LIBERO-GOAL上的三个随机种子实验中，DEBR的AUC分别达到86.5%和87.3%，相对本地MLR＋IFA复现提高18.2和10.9个百分点。选样与组件消融、优化器预算匹配及保持系数分析共同支持双经验利用与更新选择的价值。新增记忆与候选比较均用于训练，部署时保持原策略推理结构不变。'
en='Lifelong robot imitation learning requires acquiring new skills while retaining previous ones under a limited memory budget. Replaying expert demonstrations alone does not directly constrain behavioral changes on states encountered during successful autonomous execution. We propose DEBR, a dual-experience behavioral retention framework that combines expert demonstrations with the policy’s own successful experience. Random-anchored coreset selection (RA-Coreset) organizes demonstration memory by incorporating protected random anchors into a coverage-oriented subset through fixed-budget, coverage-aware exchanges. Successful Historical Behavior Retention (SHBR) stores latent histories and fixed Gaussian behavior references from successful execution, and calibrates retention strength using trajectory-level support. To coordinate retention with new-task learning, a candidate-update gate compares base and retention-augmented updates from the same training state. It accepts the augmented candidate only when current-task fitting is no worse and historical retention is better on fixed training probes. Across three seeds on LIBERO-OBJECT and LIBERO-GOAL, DEBR achieves AUC scores of 86.5% and 87.3%, improving over our local MLR+IFA reproduction by 18.2 and 10.9 percentage points, respectively. Selection and component ablations, optimizer-budget matching, and retention-weight sensitivity analysis support the value of combining dual experience with update selection. The added memory and candidate comparison are confined to training, leaving the policy inference architecture unchanged.'
for title,body,kw in [('摘要',zh,'关键词：终身模仿学习；机器人操作；经验回放；核心集选择；行为保持'),('Abstract',en,'Keywords: Lifelong imitation learning; Robot manipulation; Experience replay; Coreset selection; Behavioral retention')]:
    h=before(p[1],title,'DEBR Unified H2');h.paragraph_format.keep_with_next=True
    a=before(p[1],body);a.paragraph_format.first_line_indent=Pt(0)
    if title=='Abstract':
        for r in a.runs:r.font.name='Times New Roman';r.font.size=Pt(11)
    k=before(p[1],kw);k.paragraph_format.first_line_indent=Pt(0);k.paragraph_format.space_after=Pt(10)

references=[
'A. Hussein, M.M. Gaber, E. Elyan, C. Jayne, Imitation learning: A survey of learning methods, ACM Computing Surveys 50 (2) (2017) 1–35.',
'A. Chaudhry, M. Rohrbach, M. Elhoseiny, T. Ajanthan, P.K. Dokania, P.H.S. Torr, M. Ranzato, On tiny episodic memories in continual learning, arXiv:1902.10486 (2019). https://arxiv.org/abs/1902.10486.',
'K. Roy, A. Dissanayake, B. Tidd, P. Moghadam, M2Distill: Multi-modal distillation for lifelong imitation learning, in: 2025 IEEE International Conference on Robotics and Automation (ICRA), 2025, pp. 1429–1435.',
'Z. Wei, H. Zhao, J. Zhang, S. Ke, Y. Guo, H. Ding, Hierarchically aware replay-enhanced diffusion policy for robot continual imitation learning, IEEE Transactions on Industrial Informatics (2026) 1–12. https://doi.org/10.1109/TII.2026.3708547.',
'B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, P. Stone, LIBERO: Benchmarking knowledge transfer for lifelong robot learning, Advances in Neural Information Processing Systems 36 (2023) 44776–44791.',
'C. Gao, H. Gao, S. Guo, T. Zhang, F. Chen, CRIL: Continual robot imitation learning via generative and prediction model, in: 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2021. https://arxiv.org/abs/2106.09422.',
'F. Yu, M. Tiezzi, T. Apicella, C. Beyan, V. Murino, Lifelong imitation learning with multimodal latent replay and incremental adjustment, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 6740–6749.',
'S. Haldar, L. Pinto, PolyTask: Learning unified policies through behavior distillation, arXiv:2310.08573 (2023). https://arxiv.org/abs/2310.08573.',
'Y. Zhu, P. Stone, Y. Zhu, Bottom-up skill discovery from unsegmented demonstrations for long-horizon robot manipulation, IEEE Robotics and Automation Letters 7 (2) (2022) 4126–4133.',
'Z. Liu, J. Zhang, K. Asadi, Y. Liu, D. Zhao, S. Sabach, R. Fakoor, TAIL: Task-specific adapters for imitation learning with large pretrained models, arXiv:2310.05905 (2023). https://arxiv.org/abs/2310.05905.',
'D. Lee, M. Yoo, W.K. Kim, W. Choi, H. Woo, Incremental learning of retrievable skills for efficient continual task adaptation, Advances in Neural Information Processing Systems 37 (2024) 17286–17312.',
'W. Wan, Y. Zhu, R. Shah, Y. Zhu, LOTUS: Continual imitation learning for robot manipulation through unsupervised skill discovery, in: 2024 IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 537–544.',
'I. Loshchilov, F. Hutter, Decoupled weight decay regularization, in: International Conference on Learning Representations (ICLR), 2019. https://arxiv.org/abs/1711.05101.'
]
d.add_paragraph('参考文献 References','DEBR Unified H1')
for i,r in enumerate(references,1):
    rp=d.add_paragraph(f'[{i}] {r}','DEBR Reference')
    rp.paragraph_format.first_line_indent=Inches(-.28);rp.paragraph_format.left_indent=Inches(.28)
    rp.paragraph_format.space_after=Pt(6);rp.paragraph_format.keep_together=True;rp.paragraph_format.keep_with_next=False
    rp.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.LEFT
    for run in rp.runs:run.font.name='Times New Roman';run.font.size=Pt(10)

# Remove empty spacer paragraphs, but never equations, figures, page or section breaks.
for x in list(d.paragraphs):
    if not txt(x).strip() and not x._p.xpath('.//m:oMath|.//w:drawing|.//w:br|.//w:sectPr'):
        x._p.getparent().remove(x._p)
for x in d.paragraphs:
    if x.style.name.startswith('DEBR Unified H'):
        x.paragraph_format.keep_with_next=True;x.paragraph_format.keep_together=True

d.save(OUTPUT)
assert sha(SOURCE)==original_hash
with ZipFile(SOURCE) as a,ZipFile(OUTPUT) as b:
    media=[n for n in a.namelist() if n.startswith('word/media/')]
    assert all(a.read(n)==b.read(n) for n in media)
text='\n'.join(txt(x) for x in d.paragraphs)
assert '3.4.4' not in text and 'SBHR' not in text
assert len(d.tables)==5 and len(references)==13
(HERE/'revised_text.txt').write_text('\n'.join(f'{i}: {txt(x)}' for i,x in enumerate(d.paragraphs)),encoding='utf8')
(HERE/'REFERENCES.json').write_text(json.dumps(references,ensure_ascii=False,indent=2),encoding='utf8')
receipt=dict(status='EDITED_PENDING_VISUAL_QA',source=str(SOURCE),source_sha256=original_hash,output=str(OUTPUT),output_sha256=sha(OUTPUT),data_source=str(DATA),data_sha256=sha(DATA),tables=5,media_preserved=len(media),native_math_before=len(Document(SOURCE)._element.xpath('//m:oMath')),native_math_after=len(d._element.xpath('//m:oMath')),changes=changes)
(HERE/'EDIT_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in receipt.items() if k!='changes'},ensure_ascii=False))

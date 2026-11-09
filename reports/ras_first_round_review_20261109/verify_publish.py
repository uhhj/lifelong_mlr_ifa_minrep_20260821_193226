from pathlib import Path
from decimal import Decimal
from zipfile import ZipFile
from lxml import etree
from docx import Document
import datetime, hashlib, json, re, subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1] / 'publish_mlr_branch'
SOURCE = Path('C:/Users/Lenovo/Desktop/论文/论文写作/Coverage-Aware Replay with Support-Calibrated Execution Retention for Lifelong Robot Imitation Learning20261107 - 副本.docx')
EXPECTED = 'e86d1b22477837b8209db01852aceb659bb9ae9b8a505773df661eda1648b1c3'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
doc = Document(SOURCE)
def means(t):
    return {r.cells[0].text: [Decimal(c.text.split('±')[0]) for c in (r.cells[3 if t < 2 else 4], r.cells[-1])] for r in doc.tables[t].rows[2:]}
checks = {}
for t, baseline, expected, label in [
    (0, 'MLR＋IFA', ['18.2','10.9'], 'main_auc_gain'),
    (1, '+ Support-Weighted Retention', ['1.9','3.8'], 'gate_auc_gain'),
    (2, 'Execution Supervision\n(Matched Budget)', ['5.6','3.6'], 'matched_steps_auc_gain')]:
    rows = means(t)
    gain = [a-b for a,b in zip(rows['DEBR (Ours)'], rows[baseline])]
    assert gain == list(map(Decimal, expected))
    checks[label] = list(map(str, gain))
with ZipFile(SOURCE) as z:
    xml = etree.fromstring(z.read('word/document.xml'))
    math_count = len(xml.xpath('//*[local-name()="oMath"]'))
assert math_count == 132
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
evidence = dict(status='PASS_MANUSCRIPT_ARITHMETIC_ONLY', observed=now, source_filename=SOURCE.name,
                source_sha256=EXPECTED, source_bytes=SOURCE.stat().st_size,
                comparisons=checks, omml_math_count=math_count, tables=len(doc.tables),
                limitation='No raw outcomes, running experiments, layout or statistical significance verified.')
(ROOT/'EVIDENCE.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
def git(*args):
    return subprocess.check_output(['git','-C',str(REPO),*args],text=True,encoding='utf-8').strip()
assert git('branch','--show-current') == 'MLR'
assert git('remote','get-url','origin') == 'https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
assert not git('diff','--cached','--name-only'), 'Unrelated staged changes'
assert git('rev-parse','HEAD') == git('ls-remote','origin','refs/heads/MLR').split()[0], 'Review unpublished commits before publishing'
dest = REPO/'reports/ras_first_round_review_20261109'
dest.mkdir(parents=True,exist_ok=True)
manifest = {}
secret = re.compile(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}')
for name in ['REPORT.md','EVIDENCE.json','verify_publish.py']:
    data=(ROOT/name).read_bytes()
    assert len(data)<100000 and not secret.search(data.decode('utf-8'))
    (dest/name).write_bytes(data)
    manifest[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
(dest/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8', newline='\n')
paths=[(dest/n).relative_to(REPO).as_posix() for n in [*manifest,'MANIFEST.json']]
subprocess.run(['git','-C',str(REPO),'add','--',*paths],check=True)
assert set(git('diff','--cached','--name-only').splitlines()) == set(paths)
subprocess.run(['git','-C',str(REPO),'diff','--cached','--check'],check=True)
print(git('diff','--cached','--stat'))
subprocess.run(['git','-C',str(REPO),'commit','-m','Assess RAS first-round prospects from supplied manuscript'],check=True)
subprocess.run(['git','-C',str(REPO),'push','origin','MLR'],check=True)
head=git('rev-parse','HEAD')
remote=git('ls-remote','origin','refs/heads/MLR').split()[0]
assert head==remote
receipt=dict(status='PUSH_VERIFIED',commit=head,remote=remote,observed=now,files=paths)
(ROOT/'PUBLICATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))

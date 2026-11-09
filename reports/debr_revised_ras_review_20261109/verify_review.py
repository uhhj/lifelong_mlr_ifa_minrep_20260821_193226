"""Read-only manuscript arithmetic check. Usage: python verify_review.py source.docx."""
from pathlib import Path
from decimal import Decimal
from docx import Document
from zipfile import ZipFile
from lxml import etree
import sys, json, hashlib, datetime

source = Path(sys.argv[1])
expected = 'fd03714028ce9df0872606be794194eb6e838be794149b96596d72b3806069ed'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
doc = Document(source)
assert len(doc.tables) == 5
def values(t, name):
    row = next(r for r in doc.tables[t].rows if r.cells[0].text == name)
    return [Decimal(c.text.split('±')[0].replace(',', '')) for c in row.cells[1:]]
checks = {}
for label, t, base, columns, result in [
    ('main_auc_gain', 0, 'MLR＋IFA', [2,5], ['18.2','10.9']),
    ('gate_auc_gain', 2, '+ Support-Weighted Retention', [2,5], ['1.9','3.8']),
    ('matched_budget_auc_gain', 3, 'Execution Supervision\n(Matched Budget)', [3,7], ['5.6','3.6']),
    ('full_vs_execution_auc_gain', 2, '+ Execution Supervision', [2,5], ['1.5','4.8']),
]:
    ours, other = values(t,'DEBR (Ours)'), values(t,base)
    gain = [ours[i]-other[i] for i in columns]
    assert gain == list(map(Decimal,result)), (label,gain)
    checks[label] = list(map(str,gain))
ra, random = values(1,'RA-Coreset'),values(1,'Random')
checks['ra_random_object_fwt_from_displayed_means'] = str(ra[0]-random[0])
checks['rounding_note'] = 'Displayed means imply 6.6; prose says 6.7, possibly from unrounded means. Raw means were not examined.'
with ZipFile(source) as z:
    root = etree.fromstring(z.read('word/document.xml'))
    counts = {tag:len(root.xpath(f'//*[local-name()="{tag}"]')) for tag in ['oMath','ins','del']}
    figures = {Path(n).name:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('word/media/')}
evidence = {
    'status':'PASS_MANUSCRIPT_ARITHMETIC_WITH_ROUNDING_NOTE',
    'observed_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'source_filename':source.name,'source_bytes':source.stat().st_size,'source_sha256':expected,
    'table_count':len(doc.tables),'xml_counts':counts,'figure_hashes':figures,'checks':checks,
    'limitations':'Manuscript-only arithmetic; no raw episode outcomes, training, rollouts, significance or provenance verification.',
    'review_recommendation':'Major Revision; simulated review, not an actual editorial decision'
}
out = Path(__file__).resolve().parent/'EVIDENCE.json'
out.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'status':evidence['status'],'tables':len(doc.tables),'checks':checks},ensure_ascii=True))

"""Prepare contact sheets only; no automatic labeling or human ground truth."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(pack,out):
    out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((pack/'ANNOTATION_MANIFEST.csv').open(encoding='utf-8')))
    assert len(rows)==450 and all(r['label']=='' for r in rows)
    protocol=dict(status='CONTACT_SHEETS_ONLY_NO_LABELS_GENERATED',annotator='Codex multimodal assistant',
        human_verified=False,eligible_for_original_human_calibration=False,
        evidence='provided agent-view and wrist-view RGB images; task identity for target identification',
        excluded_evidence=['visual similarity scores','expert future actions','A share/bif labels','quotas'],
        positive='Same visibly comparable target-hand-goal relation and interaction phase; object identity alone is not a mismatch',
        negative='Visible mismatch in task-relevant target position, grasp/support relation or phase despite similar robot pose',
        uncertain='Target/contact/phase cannot be resolved from these two low-resolution still images',
        temporal_limit='No dynamic state or future motion inferred from still images',
        original_manifest_sha256=sha(pack/'ANNOTATION_MANIFEST.csv'),formal_training_resumed=False)
    from .export_ai_labels import atomic_new, json_text
    atomic_new(out/'CONTACT_SHEET_PROTOCOL.json', json_text(protocol))
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    for start in range(0,len(rows),15):
        sheet=Image.new('RGB',(1200,1775),'#e8e8e8');draw=ImageDraw.Draw(sheet)
        for k,r in enumerate(rows[start:start+15]):
            im=Image.open(pack/'annotation/images'/(r['pair_id']+'.png'))
            assert im.size==(800,650)
            x=(k%3)*400;y=(k//3)*355
            draw.text((x+4,y+2),'%03d %s'%(start+k,r['pair_id']),fill='black',font=font)
            sheet.paste(im.resize((400,325),Image.Resampling.LANCZOS),(x,y+28))
        sheet.save(out/('sheet_%02d.jpg'%(start//15)),quality=95)
    (out/'ROW_ORDER.json').write_text(json.dumps([r['pair_id'] for r in rows]),encoding='utf-8')
    print('Prepared 30 sheets, 450 pairs. Original blank human labels unchanged.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pack',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();prepare(a.pack,a.out)

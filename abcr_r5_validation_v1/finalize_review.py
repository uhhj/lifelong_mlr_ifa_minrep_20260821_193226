"""Export actual AI visual decisions; does not infer labels from physics/actions."""
import argparse
from pathlib import Path
from .common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();source=Path(__file__).with_name('AI_REVIEW_DECISIONS.json');dec=read(source);blinded=read(a.out/'audit/BLINDED_MANIFEST.json');lookup={r['audit_id']:r for r in blinded['records']};assert len(dec)==len(lookup)==72
    rows=[]
    for ident,label,newphase,oldphase,reason in dec:
        assert label in PROTOCOL['A']['labels'];assert sha(a.out/'audit'/(ident+'.png'))==lookup[ident]['image_sha256']
        rows.append(dict(audit_id=ident,label=label,new_phase=newphase,old_phase=oldphase,reason=reason,reviewer_type='AI',image_sha256=lookup[ident]['image_sha256']))
    table(a.out/'audit/AI_LABELS.csv',rows)
    write(a.out/'AI_REVIEW_PROVENANCE.json',dict(source_sha=sha(source),reviewer='Codex visual review, AI NOT human',reviewed_images=72,multiframe_boards=24,previous_labels_hidden=True,action_class_and_scores_hidden=True,loss_outcomes_not_available_to_visual_review=True,independent_human_review=False,user_amendment=PROTOCOL['user_amendment'],limitations='Same assistant as prior study; new blinded identifiers reduce label visibility but this is not independent human replication. Contact and multiframe evidence inform local comparability, not knowledge sharing.'))

if __name__=='__main__':main()

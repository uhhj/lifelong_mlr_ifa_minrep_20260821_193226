"""Read-only E1 pairing evidence. Never influences selector/model or draws."""
import json
from pathlib import Path
from .io import write, sha
OLD=Path('/data/mlr_ifa/experiments/E1_protective_object_seed42_20261008')

def epoch_pairing(out,stage,epoch,rows):
    oldpath=OLD/'selection_logs'/('task%d'%stage)/('epoch%03d.jsonl'%epoch)
    old=[json.loads(line) for line in oldpath.read_text().splitlines()]
    differences=[]
    if len(old)!=len(rows):differences.append(dict(kind='row_count',old=len(old),new=len(rows)))
    for a,b in zip(old,rows):
        at=b['stage_optimizer_step']
        for k in ('stage','epoch','stage_optimizer_step'):
            if a[k]!=b[k]:differences.append(dict(step=at,kind=k))
        ar,br=a['refresh'],b['refresh']
        if (ar is None)!=(br is None):differences.append(dict(step=at,kind='refresh_cadence'))
        elif ar is not None:
            for k in ('step','references','current_reference','candidates'):
                if ar[k]!=br[k]:differences.append(dict(step=at,kind=k))
        if [(x['slot'],x['task']) for x in a['replacements']] != [(x['slot'],x['task']) for x in b['replacements']]:
            differences.append(dict(step=at,kind='attempted_positions'))
    result=dict(stage=stage,epoch=epoch,e1_source=str(oldpath),e1_sha256=sha(oldpath),
                rows=len(rows),strict_pairing=not differences,differences=differences,
                interpretation='Paired pools and gates' if not differences else 'PAIRING_GAP: retained real trajectory; no reseeding or matching redraw')
    write(Path(out)/'pairing'/('task%d_epoch%03d.json'%(stage,epoch)),result)
    return result

def buffer_pairing(out,stage,actual):
    path=OLD/('buffer_after_task%d.json'%stage)
    expected=json.loads(path.read_text())
    result=dict(stage=stage,e1_sha256=sha(path),members_and_payload_hashes_match=actual==expected,count=len(actual))
    write(Path(out)/('buffer_pairing_task%d.json'%stage),result)
    assert actual==expected, 'E1U stage buffer differs from E1'

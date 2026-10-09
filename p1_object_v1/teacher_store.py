"""Immutable stage tables and atomically published active-window manifests."""
import json
import time
from pathlib import Path

import torch
from torch.utils.data._utils.collate import default_collate
from .io import sha,save,write
from .policy_distill import fields,forward_distribution


class TeacherStore:
    def __init__(self,root,seed,manifest=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.seed=int(seed);self.cache={}
        self.manifest=Path(manifest) if manifest else None
        self.rows=json.loads(self.manifest.read_text())['windows'] if self.manifest else {}

    def validate(self,buffer):
        paths={p.name:p for p in Path(buffer).glob('*.pt')}
        assert paths.keys()==self.rows.keys(),'BUFFER_TEACHER_MEMBERSHIP_MISMATCH'
        checked=set()
        for identity,row in self.rows.items():
            assert row['seed']==self.seed and row['window_id']==identity,'TEACHER_IDENTITY_MISMATCH'
            assert sha(paths[identity])==row['input_sha256'],'TEACHER_INPUT_MISMATCH'
            if row['table'] not in checked:
                assert sha(row['table'])==row['table_sha256'],'TEACHER_TABLE_MISMATCH'
                checked.add(row['table'])
        return True

    def lookup(self,identities,tasks):
        output=[]
        for identity,task in zip(identities,tasks):
            assert identity in self.rows,'MISSING_HISTORICAL_TEACHER:'+identity
            row=self.rows[identity]
            assert row['seed']==self.seed and row['source_task']==int(task) and row['window_id']==identity,'TEACHER_IDENTITY_MISMATCH'
            if row['table'] not in self.cache:
                assert sha(row['table'])==row['table_sha256']
                self.cache[row['table']]=torch.load(row['table'],map_location='cpu')
            table=self.cache[row['table']]
            assert table['seed']==self.seed and table['ids'][row['row']]==identity
            output.append({k:table[k][row['row']] for k in ('log_weights','means','scales')})
        return {k:torch.stack([r[k] for r in output]).detach() for k in output[0]}

    def build_stage(self,model,buffer,task,checkpoint,report):
        begin=time.monotonic();paths=sorted(Path(buffer).glob('*.pt'));active={}
        new=[p for p in paths if p.name.startswith('t%d__'%task)]
        assert all(p.name in self.rows or p in new for p in paths),'MISSING_OLDER_TEACHER'
        source_hash=sha(checkpoint);table_path=self.root/('task%d_ep99.pt'%task)
        ids=[p.name for p in new]
        if table_path.exists():
            table=torch.load(table_path,map_location='cpu')
            assert table['ids']==ids and table['seed']==self.seed and table['checkpoint_sha256']==source_hash
            assert table['input_sha256']==[sha(p) for p in new]
        else:
            chunks=[]
            with torch.no_grad():
                for start in range(0,len(new),64):
                    data=default_collate([torch.load(p,map_location='cpu') for p in new[start:start+64]])
                    data={k:v.cuda() for k,v in data.items()}
                    chunks.append({k:v.detach().cpu() for k,v in fields(forward_distribution(model,data)).items()})
            assert chunks
            table=dict(seed=self.seed,source_task=task,epoch=99,ids=ids,
                checkpoint=str(checkpoint),checkpoint_sha256=source_hash,input_sha256=[sha(p) for p in new],
                **{k:torch.cat([c[k] for c in chunks]) for k in chunks[0]})
            save(table_path,table)
        table_hash=sha(table_path)
        for p in paths:
            if p.name in self.rows:active[p.name]=self.rows[p.name]
            else:
                i=ids.index(p.name)
                active[p.name]=dict(seed=self.seed,source_task=task,source_epoch=99,window_id=p.name,
                    table=str(table_path),row=i,table_sha256=table_hash,input_sha256=table['input_sha256'][i],
                    checkpoint_sha256=source_hash,checkpoint=str(checkpoint))
        manifest=self.root/('ACTIVE_task%d.json'%task)
        value=dict(seed=self.seed,stage=task,windows=active)
        if manifest.exists():assert json.loads(manifest.read_text())==value,'IMMUTABLE_TEACHER_MANIFEST_MISMATCH'
        else:write(manifest,value)
        self.manifest=manifest;self.rows=active;self.validate(buffer)
        cost=dict(task=task,teacher_count=len(new),active_count=len(active),seconds=time.monotonic()-begin,
            numeric_payload_bytes=300*len(active),latent_bytes=sum(p.stat().st_size for p in paths),
            active_teacher_bytes=sum(Path(p).stat().st_size for p in set(r['table'] for r in active.values())),
            index_bytes=manifest.stat().st_size,source_checkpoint_sha256=source_hash)
        with (Path(report)/'teacher_build_log.jsonl').open('a') as f:f.write(json.dumps(cost,allow_nan=False)+'\n')
        return manifest

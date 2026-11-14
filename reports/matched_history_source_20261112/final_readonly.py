"""Independently audit the completed paired experiment without starting rollouts."""
import ast
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


BASE = Path('/data/mlr_ifa/experiment_workspaces')
ORIGINAL = BASE / 'MATCHED_HISTORY_SOURCE_GOAL_OBJECT_20261112_v1'
ACTIVE = ORIGINAL.with_name(ORIGINAL.name + '_engineering02')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def local_time():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def check_exit(directory, *names):
    for name in names:
        assert (directory / (name + '.exit')).read_text().strip() == '0', (directory, name)


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit_case(directory):
    from runtime import Dataset, O, S, POP, VALID, OLD, SEED, TEACHERS, CASE_SPEC
    from runtime import MATCHED_MEMORY, COMMON_INITIAL, acquisition_path, verify, equal
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    import torch.distributions as D
    from transformers import GPT2Config, GPT2Model
    import robomimic.utils.tensor_utils as TensorUtils
    from match_histories import expert_candidates, match

    def array_sha(array):
        return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

    def tensor_sha(tensor):
        return array_sha(tensor.detach().cpu().numpy())

    def symbols():
        space = dict(nn=nn, F=F, D=D, torch=torch, GPT2Config=GPT2Config,
                     GPT2Model=GPT2Model, TensorUtils=TensorUtils, math=math)
        for name, src, cls in (
            ('OfficialGPT2Decoder', 'modules/transformer_modules.py', None),
            ('GMMHead', 'policy_head.py', None),
            ('temporal_encode', 'brief.py', 'BriefPolicy'),
        ):
            path = S / 'libero/lifelong/models' / src
            nodes = ast.parse(path.read_text()).body
            if cls:
                nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls).body
            node = next(n for n in nodes if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), space)
        return space

    code = symbols()

    class Policy(nn.Module):
        temporal_encode = code['temporal_encode']

        def __init__(self):
            super().__init__()
            self.temporal_transformer = code['OfficialGPT2Decoder'](512, 6, 8, .15, 32, train=True)
            self.policy_head = code['GMMHead'](512, 7, num_modes=5, min_std=1e-4)

    def strict_model(state):
        normalized = {k[7:] if k.startswith('module.') else k:v for k,v in state.items()}
        model = Policy().eval().requires_grad_(False)
        assert len(normalized) == len(model.state_dict()) == 100
        assert all(normalized[k].dtype == v.dtype and normalized[k].shape == v.shape and
                   torch.isfinite(normalized[k]).all() for k, v in model.state_dict().items())
        model.load_state_dict(normalized, strict=True)
        assert all(torch.equal(normalized[k], v) for k, v in model.state_dict().items())
        return model

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    contract = verify()
    check_exit(directory, 'match', 'initial_E', 'initial_S', 'train_E', 'train_S', 'evaluate')
    acq = directory.parent.parent / directory.name
    check_exit(acq, 'acquire', 'collect_6', 'collect_7', 'collect_8')
    split = read(directory / 'INITIAL_SPLIT.json')
    actual_initials = {}
    for task, row in split.items():
        assert digest(row['path']) == row['file_sha256']
        states = torch.load(row['path'], map_location='cpu')
        hashes = {str(i):array_sha(np.asarray(states[i])) for i in range(30)}
        assert all(hashes[i] == h for kind in ('training','evaluation') for i,h in row[kind].items())
        assert not set(row['training'].values()) & set(row['evaluation'].values())
        actual_initials[int(task)] = hashes
    assert len(actual_initials) == 4

    def check_episode(row, task, index, cp, collection):
        assert row['task'] == task and row['initial_index'] == index and row['attempt'] == 1
        assert row['cp_sha256'] == cp and row['env_seed'] == (SEED+100 if collection else SEED+index//10)
        assert row['initial_file_sha256'] == split[str(task)]['file_sha256']
        assert row['initial_sha256'] == actual_initials[task][str(index)]
        base = 20271025 if collection else 20261024
        eps = np.random.RandomState(base+SEED*1000+task*100+index).standard_normal((600,7)).astype(np.float32)
        assert row['noise_sha256'] == array_sha(eps)
        assert 0 < row['steps'] <= 600 and isinstance(row['success'], bool)
        assert row['termination'] == ('success' if row['success'] else 'timeout600')
        assert row['success'] or row['steps'] == 600

    metadata = read(directory / 'MATCHED_MEMORY.json')
    assert metadata['status'] == 'PASS_MATCHED_HISTORY_CACHE' and metadata['sha256'] == digest(MATCHED_MEMORY)
    matched = torch.load(MATCHED_MEMORY, map_location='cpu')
    assert set(matched) == {'E','S','probe'} and all(set(parts)=={6,7,8} for parts in matched.values())
    probe = read(directory/'PROBES.json')
    assert probe['shared_cache_sha256'] == metadata['sha256']
    collection_summary = {}
    teacher_max_abs = 0.
    replayed_windows = 0

    def row_proof(row):
        measured = {k:v for k,v in row.items() if k not in ('feature','mu','sd')}
        measured.update({k+'_sha256':tensor_sha(row[k]) for k in ('feature','mu','sd')})
        return measured

    for task in (6,7,8):
        teacher = TEACHERS[str(task)]
        assert digest(teacher['path']) == teacher['sha256']
        policy = strict_model(torch.load(teacher['path'], map_location='cpu')['state_dict'])
        outcomes_path = acq/('COLLECTION_OUTCOMES_S_%d.json'%task)
        outcomes = read(outcomes_path)
        assert len(outcomes)==10
        for j,row in enumerate(outcomes):
            check_episode(row,task,20+j,teacher['sha256'],True)
        m = read(acq/('MEMORY_S_%d.json'%task))
        cache_path = acquisition_path(task)
        assert digest(cache_path)==m['cache_sha256'] and digest(outcomes_path)==m['collection_outcomes_sha256']
        cache = torch.load(cache_path,map_location='cpu')
        assert cache['source_sha256']==m['source_sha256']==teacher['sha256']
        assert len(cache['rows'])==m['count']<=128 and [row_proof(x) for x in cache['rows']]==m['rows']
        expected = []
        for j,row in enumerate(outcomes):
            if not row['success']:
                continue
            gen=np.random.RandomState(90100+SEED+task*100+20+j)
            size=13 if j<8 else 12
            pool=[]
            for step in range(row['steps']):
                slot=step if step<size else int(gen.randint(step+1))
                if step<size:
                    pool.append(step)
                elif slot<size:
                    pool[slot]=step
            expected.extend((j,step,min(step+1,8)) for step in sorted(pool))
        assert [(row['trajectory'],row['step'],row['length']) for row in cache['rows']]==expected
        ds=Dataset(task)
        candidates=expert_candidates(ds,task)
        yy=torch.stack([torch.as_tensor(x['next_action']) for x in ds.rows if x['task']==task and x['bc_valid']]).float()
        scale=torch.stack([yy[:,lo:hi].square().mean().sqrt().clamp_min(.1) for lo,hi in ((0,3),(3,6),(6,7))])
        assert torch.equal(cache['scales'],scale)
        records=metadata['tasks'][str(task)]
        assert records['original_online_cache_sha256']==m['cache_sha256'] and records['teacher_sha256']==teacher['sha256']
        online={(row['trajectory'],row['step']):row_proof(row) for row in cache['rows']}
        groups=collections.defaultdict(list)
        for row in cache['rows']:
            groups[row['trajectory']].append(row)
        rng_base=20261112+SEED*1000+task*100
        expected_pairs, expected_probes, expected_trajectories=match(groups,candidates,rng_base)
        assert records['rng_base']==rng_base and len(records['pair_records'])==len(expected_pairs)
        assert records['raw_windows']==len(cache['rows']) and records['raw_trajectories']==len(groups)
        train={k:iter(matched[k][task]['rows']) for k in ('E','S')}
        held=iter(matched['probe'][task]['rows'])
        expert_seen=set()
        online_seen=set()
        sources={}
        all_expert=[]
        n_probe=0
        for i,(record,(_,online_row,expert_row)) in enumerate(zip(records['pair_records'],expected_pairs)):
            assert record['pair']==i and record['probe']==(i in expected_probes)
            assert record['online']==row_proof(online_row)
            assert all(record['expert'][key]==value for key,value in expert_row.items())
            if record['probe']:
                e,s=next(held),next(held)
                n_probe+=1
            else:
                e,s=next(train['E']),next(train['S'])
            assert row_proof(e)==record['expert'] and row_proof(s)==record['online']
            assert e['trajectory']==s['trajectory'] and e['length']==s['length']
            key=(s['trajectory'],s['step']); ek=(e['demo'],e['u'],e['length'])
            assert key not in online_seen and ek not in expert_seen and online[key]==record['online']
            online_seen.add(key);expert_seen.add(ek)
            assert sources.setdefault(e['trajectory'],e['demo'])==e['demo']
            assert any(all(e[k]==candidate[k] for k in candidate)
                       for candidate in candidates[e['demo']][e['length']])
            feature=ds.native[e['source_index']][0]['feature'].squeeze(0).float()[:e['length']]
            assert torch.equal(feature,e['feature'])
            all_expert.append(e)
        assert len(set(sources.values()))==len(sources) and n_probe<=4
        assert all(next(it,None) is None for it in (train['E'],train['S'],held))
        ids=sorted({x['trajectory'] for x in matched['E'][task]['rows']})
        assert set(ids)==set(sources) and ids==expected_trajectories
        assert records['dropped_online_windows']==len(cache['rows'])-len(expected_pairs)
        assert records['dropped_online_trajectories']==sorted(set(groups)-set(ids))
        for arm in matched:
            part=matched[arm][task]
            assert part['support']==probe['support'][str(task)]==len(ids)/10
            assert part['support_trajectories']==probe['support_trajectories'][str(task)]==ids
            assert torch.equal(part['scales'],scale) and part['source_sha256']==teacher['sha256']
        assert probe['old_indices'][str(task)]==list(range(2*n_probe))
        assert len(matched['E'][task]['rows'])==len(matched['S'][task]['rows'])<=128-n_probe
        buckets=collections.defaultdict(list)
        for row in all_expert+cache['rows']:
            buckets[row['length']].append(row)
        for group in buckets.values():
            for offset in range(0,len(group),16):
                chunk=group[offset:offset+16]
                inp=torch.stack([x['feature'] for x in chunk])
                mu,sd,logits=policy.policy_head.forward_fn(policy.temporal_encode(inp)[:,-1])
                for i,row in enumerate(chunk):
                    mode=int(logits[i].argmax())
                    assert mode==row['mode']
                    for val in ('mu','sd'):
                        pred=(mu if val=='mu' else sd)[i,mode]
                        teacher_max_abs=max(teacher_max_abs,float((pred-row[val]).abs().max()))
                        assert torch.allclose(pred,row[val],atol=2e-4,rtol=2e-4)
                    replayed_windows+=1
        collection_summary[task]=dict(successes=sum(row['success'] for row in outcomes),
            paired_trajectories=len(ids),train_windows_per_arm=len(matched['E'][task]['rows']),
            probes_per_arm=n_probe,teacher_sha256=teacher['sha256'],cache_sha256=m['cache_sha256'])

    ds=Dataset(9)
    initial=read(directory/'INITIAL_E.json')
    assert initial==read(directory/'INITIAL_S.json') and initial['common_initial_sha256']==digest(COMMON_INITIAL)
    first=torch.load(COMMON_INITIAL,map_location='cpu')
    start=CASE_SPEC['common_initial']
    assert digest(start['path'])==initial['teacher_checkpoint_sha256']==start['sha256']
    assert equal(first['model'],torch.load(start['path'],map_location='cpu')['state_dict'])
    assert initial['dataset_manifest_sha256']==digest(O/'ra_coreset/dataset_manifests/task9.json')
    assert first['step']==initial['global_step']==start['global_step']
    assert not first['optimizer']['state'] and all(g is None for g in first['grads'].values())
    assert set(first['rng'])=={'python','numpy','cpu','cuda'} and len(first['rng']['cuda'])==1
    new_ids=[i for i,x in enumerate(ds.rows) if x['task']==9 and x['bc_valid']]
    expected_new=sorted(int(i) for i in np.random.RandomState(20261027+SEED*100+9).choice(new_ids,min(64,len(new_ids)),replace=False))
    assert probe['new_indices']==expected_new and probe['new_population']==len(new_ids)
    assert sum(len(matched[arm][t]['rows']) for arm in ('E','S') for t in (6,7,8))<=768

    expected=CASE_SPEC['accepted_per_arm']; nb=(POP[9]+9)//10
    training={}
    for arm in 'ES':
        tr=read(directory/('TRAIN_%s.json'%arm))
        history=read(directory/('EPOCHS_%s.json'%arm))
        gates=iter(lines(directory/('GATES_%s.jsonl'%arm)))
        assert tr['epochs']==len(history)==100
        done=blocks=q_blocks=0
        for i,row in enumerate(history):
            sampler=torch.randperm(POP[9],generator=torch.Generator().manual_seed(i)).numpy().astype(np.int64)
            assert row['epoch']==i and row['stage']==9 and row['batches']==nb and row['sampler_sha256']==array_sha(sampler)
            assert row['totals']['valid']==VALID[9] and row['totals']['invalid']==POP[9]-VALID[9]
            assert row['totals']['old_eligible']==OLD[9]
            accepted_slots=0
            for batch in range(0,nb,10):
                gate=next(gates)
                size=min(10,nb-batch)
                assert (gate['epoch'],gate['batch'],gate['steps'],gate['start_step'],gate['end_step'])==(i,batch,size,initial['global_step']+done,initial['global_step']+done+size)
                c,q=gate['C'],gate['Q']
                check=dict(bc_new=q['bc_new']<=c['bc_new'],exec_new=q['exec_new']<=c['exec_new'],retain_improved=q['retain_support']<c['retain_support'])
                assert all(math.isfinite(x[k]) for k in ('bc_new','exec_new','retain_support') for x in (c,q))
                is_q=all(check.values())
                assert gate['checks']==check and gate['accepted']==('Q' if is_q else 'C')
                q_blocks+=is_q; blocks+=1; done+=size
                accepted_slots+=4*size*is_q
            assert row['totals']['anchor_slots']==accepted_slots
            assert row['totals']['anchor_actual']+row['totals']['anchor_empty']==accepted_slots
            assert row['physical_optimizer_calls']==2*done and row['global_step']==initial['global_step']+done
            assert row['gate_blocks']==blocks and row['gate_Q_accepts']==q_blocks
        assert next(gates,None) is None and done==expected==tr['accepted_updates']
        assert tr['physical_optimizer_calls']==2*done and tr['gate_blocks']==blocks and tr['gate_Q_accepts']==q_blocks
        path=Path(tr['checkpoint'])
        assert digest(path)==tr['sha256'] and os.path.samefile(path,path.with_name('recovery.pth'))
        recovery=torch.load(path,map_location='cpu')
        assert recovery['stage']==9 and recovery['next_epoch']==100 and recovery['history']==history
        assert recovery['contract']==contract and recovery['arm']==arm and recovery['physical_optimizer_calls']==2*done
        assert recovery['matched_cache_sha256']==metadata['sha256'] and all(recovery[k]==v for k,v in initial.items() if k!='status')
        state=recovery['state']
        assert set(state)==set(first)=={'model','optimizer','scheduler','grads','modes','requires','rng','step'}
        assert set(state['rng'])==set(first['rng'])
        assert state['optimizer']['state'] and state['scheduler'] and state['requires']==first['requires']
        assert state['step']==initial['global_step']+done and set(state['grads'])==set(first['grads'])
        assert set(state['modes'])==set(first['modes']) and all(type(v) is bool for v in state['modes'].values())
        for optimizer_state in state['optimizer']['state'].values():
            assert all(torch.isfinite(v).all() for v in optimizer_state.values() if torch.is_tensor(v))
        assert all(v is None or torch.isfinite(v).all() for v in state['grads'].values())
        assert equal(state['model'],recovery['state_dict'])
        strict_model(state['model'])
        admission=read(directory/('ADMISSION_%s.json'%arm))
        assert admission['status']=='PASS_OWN_GPU_ADMISSION' and admission['optimizer_calls']==29
        training[arm]=dict(epochs=100,accepted_updates=done,production_optimizer_calls=2*done,
            checkpoint_sha256=tr['sha256'],strict_keys=100,recovery_all_fields=True,
            gate_blocks=blocks,gate_Q_accepts=q_blocks,admission_optimizer_calls=29,
            epoch_compute_s=sum(row['seconds'] for row in history))

    end=read(directory/'EVALUATION_COMPLETE.json')
    rows=read(directory/'OUTCOMES.json')
    def key(row):return row['arm'],row['task'],row['initial_index']
    index={key(row):row for row in rows}
    shards=[row for path in (directory/'outcomes').glob('*.json') for row in read(path)]
    assert len(index)==len(rows)==len(shards)==end['episodes']==160 and {key(x):x for x in shards}==index
    events=lines(directory/'EVAL_EVENTS.jsonl')
    assert [(e['run'],e['kind']) for e in events]==[
        ('%s_s9_t%d_g%d'%(arm,t,g),event)
        for arm in 'ES' for t in range(6,10) for g in range(2) for event in ('start','complete')]
    counts={a:{} for a in 'ES'}
    paired={}
    for task in range(6,10):
        pairs=[]
        for i in range(20):
            e,s=(index[a,task,i] for a in 'ES')
            for row in (e,s):
                check_episode(row,task,i,training[row['arm']]['checkpoint_sha256'],False)
                assert row['stage']==9
            pairs.append((e['success'],s['success']))
        counts['E'][task]=sum(e for e,_ in pairs)
        counts['S'][task]=sum(s for _,s in pairs)
        paired[task]=dict(E_only=sum(e and not s for e,s in pairs),S_only=sum(s and not e for e,s in pairs))
    return dict(case=directory.name,training=training,collection=collection_summary,counts=counts,
                paired=paired,formal_episodes=160,collection_episodes=30,initial_state_bytes_checked=120,
                cpu_teacher_replay_windows=replayed_windows,cpu_teacher_replay_max_abs=teacher_max_abs,
                outcome_sha256=digest(directory/'OUTCOMES.json'),
                collection_outcome_sha256={t:digest(acq/('COLLECTION_OUTCOMES_S_%d.json'%t)) for t in (6,7,8)},
                evaluation_completed_epoch_s=end['time'])


def main():
    check_exit(ACTIVE, 'controller')
    assert read(ACTIVE/'STATUS.json')['phase']=='complete_pending_final_audit'
    assert not (ACTIVE/'FAILURE.json').exists()
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not gpu
    for process in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv=process.read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            continue
        assert not any(x.startswith(str(root).encode()+b'/') for root in (ORIGINAL,ACTIVE) for x in argv), process
    path=ACTIVE/'PAIR_CONTRACT.json'
    assert digest(path)=='bc3db9e6f252cc5167f7bb043ae420a7daeb88b7b6e4959d5c9a8cfa6281b397'
    contract=read(path)
    assert all(digest(f)==h for f,h in contract['files'].items())
    assert read(ORIGINAL/'controller.exit')==1 and contract['inherited_completed_cases']==['GOAL_41']
    retry=read(ACTIVE/'GOAL_42'/'TASK7_REPEAT_RECEIPT.json')
    assert retry['repeated_episodes']==10 and retry['actual_attempt_ordinal']==2
    prior=ORIGINAL/'GOAL_42'/'COLLECTION_OUTCOMES_S_7.json'
    new=ACTIVE/'GOAL_42'/'COLLECTION_OUTCOMES_S_7.json'
    assert digest(prior)==retry['original_sha256'] and digest(new)==retry['replacement_sha256']
    fields=('task','initial_index','initial_sha256','initial_file_sha256','noise_sha256','cp_sha256','env_seed')
    assert len(read(prior))==len(read(new))==10
    assert all(all(x[k]==y[k] for k in fields) for x,y in zip(read(prior),read(new)))
    assert retry['identical_success_and_steps']==sum((x['success'],x['steps'])==(y['success'],y['steps']) for x,y in zip(read(prior),read(new)))==10
    assert retry['identical_action_streams']==sum(x['action_stream_sha256']==y['action_stream_sha256'] for x,y in zip(read(prior),read(new)))
    return dict(controller_exit=0,residual_experiment_processes=0,frozen_files=len(contract['files']),
                contract_sha256=digest(path),inherited_completed_cases=contract['inherited_completed_cases'],
                case_names=contract['cases'],retry=retry,observed_CST=local_time())


if __name__ == '__main__':
    directory=Path.cwd()
    print(json.dumps(audit_case(directory) if directory.parent.name=='pairs' else main()))

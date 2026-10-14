"""Run the user-supplied screen unchanged; add provenance and missing batch audit.

python -m abcr_pair_availability_a_v1.run --root REPOSITORY
Only existing CPU-loaded feature/actions assets are used. No model imports.
"""
import argparse
import ast
import collections
import csv
import datetime
import json
import math
import os
import platform
import random
import resource
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import torch
from . import original_probe as probe
from .batches import compare
from .io import sha, write, table, read_table
from .cache import CacheMismatch, freeze_many, make_binding, verify_complete
from .settings import FROZEN_PARAMETERS, native_args, assert_consistent

NAME = 'abcr_pair_availability_a_v1'
SEEDS = (42, 43, 44)
TASKS = (6, 7, 8, 9)


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def execution_sources(root):
    package = Path(__file__).resolve().parent
    result = {NAME+'/'+p.name:sha(p) for p in sorted(package.glob('*.py'))}
    for name in ('libero/lifelong/algos/mlr.py', 'libero/lifelong/models/complete_policy.py',
                 'libero/lifelong/models/base_policy.py', 'libero/lifelong/save_feature.py'):
        result[name] = sha(root/name)
    return result


def runtime_identity():
    return dict(python=platform.python_version(), pytorch=torch.__version__,
        cpu_threads=torch.get_num_threads(), omp=os.environ.get('OMP_NUM_THREADS'),
        mkl=os.environ.get('MKL_NUM_THREADS'), openblas=os.environ.get('OPENBLAS_NUM_THREADS'),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))


def native_selection(root, feature_root, task, seed):
    # Execute only the actual repository's two pure file-selection helpers.
    # Do not import LIBERO (which would initialize model/environment dependencies).
    path = root / 'libero/lifelong/algos/mlr.py'
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_list_pt', '_sample')]
    assert len(nodes) == 2
    import glob
    import typing
    scope = dict(os=os, glob=glob, random=random, List=typing.List, Sequence=typing.Sequence)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    paths = scope['_list_pt'](str(feature_root / str(task)))
    return [Path(p) for p in scope['_sample'](paths, max(1, math.ceil(len(paths)*0.5)), random.Random(seed))]


def audit(root, out):
    entries = []
    configurations = {}
    for seed in SEEDS:
        p = root / ('runs/own_reference_control/ifa/seed%d/run_001/config.json' % seed)
        cfg = json.loads(p.read_text())
        assert cfg['seed'] == seed and cfg['start_task'] == 6 and cfg['end_task'] == 10
        assert cfg['past_data_rate'] == 0.5
        configurations[str(seed)] = dict(path=str(p), sha256=sha(p), seed=seed,
            original_feature_dir=cfg['feature_dir'], original_buffer_dir=cfg['buffer_dir'],
            start_task=6, end_task=10, past_data_rate=0.5)
        for task in TASKS:
            paths = sorted((root / ('features/object_seed%d/%d' % (seed, task))).glob('*.pt'))
            assert paths, 'Missing current task cache'
            for p in paths:
                payload = torch.load(str(p), map_location='cpu', weights_only=True)
                assert set(payload) == {'feature', 'actions'}, 'Unexpected metadata: inspect before using'
                h, a = payload['feature'], payload['actions']
                assert tuple(h.shape) == (1, 8, 4, 512) and tuple(a.shape) == (1, 8, 7)
                assert h.dtype == a.dtype == torch.float32
                assert bool(torch.isfinite(h).all() and torch.isfinite(a).all())
                assert bool((h[0, -1, 1:3].norm(dim=1) > 1e-8).all())
                entries.append(dict(seed=seed, task=task, file=str(p), size=p.stat().st_size, sha256=sha(p),
                    trajectory_id=None, window_index=None, source_record_index=int(p.stem)))
    freeze_many({out / 'INPUT_MANIFEST.json': dict(files=entries, configurations=configurations,
        trajectory_mapping='NOT_AVAILABLE: file stems are exported record IDs, not proven trajectory/timestep IDs',
        all_finite=True, no_dropped_files=True)})
    # Existing native selection is append-only at this scale; verify source semantics.
    source = (root/'libero/lifelong/algos/mlr.py').read_text()
    assert 'BUFFER_CAPACITY = 59660000' in source
    assert 'if prev_total + len(curr_10pct) <= capacity:' in source
    assert 'rng = random.Random(seed)' in source
    hashes = {x['file']: x['sha256'] for x in entries}
    stages = []
    for seed in SEEDS:
        fr = root / ('features/object_seed%d' % seed)
        stages.append(dict(seed=seed, task=6, status='NO_OLD_BUFFER', n_current=len(list((fr/'6').glob('*.pt'))),
            n_old=0, note='Native incremental replay starts empty; no base-task replay invented'))
        for task in (7, 8, 9):
            selected = []
            for old in range(6, task):
                selected.extend((old, p) for p in native_selection(root, fr, old, seed))
            want = {'t%d__%s' % (tid, p.name): p for tid, p in selected}
            assert len(want) < 59660000 and len(want) == len(selected)
            if seed == 42:
                tx = root / ('runs/a1_object_v1/seed42/transactions/task%d' % (task-1))
                assert (tx/'COMMITTED.json').exists()
                plan = json.loads((tx/'PLAN.json').read_text())
                assert set(plan['files']) == set(want)
                bd = tx/'buffer'
                assert set(p.name for p in bd.glob('*.pt')) == set(want)
                for name, src in want.items():
                    assert sha(bd/name) == hashes[str(src)], 'Historical replay payload differs from feature source'
                evidence = dict(kind='EXISTING_COMMITTED_NATIVE_SELECTION', plan=str(tx/'PLAN.json'),
                    plan_sha256=sha(tx/'PLAN.json'), committed_sha256=sha(tx/'COMMITTED.json'),
                    all_buffer_bytes_equal_original_feature_files=True,
                    note='A1 did not alter replay selection; every member verified against actual native sampling helpers')
            else:
                # Recreate only path views, not features or models; no access to future tasks.
                bd = root / ('runs/%s/seed%d/task%d/old_view' % (NAME, seed, task))
                bd.mkdir(parents=True, exist_ok=True)
                for name, src in want.items():
                    p = bd/name
                    if not p.exists():
                        p.symlink_to(src)
                    assert p.resolve() == src.resolve()
                assert set(p.name for p in bd.glob('*.pt')) == set(want)
                evidence = dict(kind='DETERMINISTIC_NATIVE_RECONSTRUCTION',
                    note='Original corrected-baseline buffer directory no longer present. Exact native selection replayed from surviving per-seed features and original config; not a surviving historical buffer snapshot.')
            stages.append(dict(seed=seed, task=task, status='READY', feature_dir=str(fr), buffer_dir=str(bd),
                n_current=len(list((fr/str(task)).glob('*.pt'))), n_old=len(want),
                old_counts=dict(collections.Counter(str(tid) for tid, _ in selected)),
                buffer_evidence=evidence, old_files=sorted(want)))
    freeze_many({out/'STAGE_MANIFEST.json': dict(stages=stages)})
    return stages


def run_stage(stage, out, code, runtime, parameters=None):
    parameters = dict(FROZEN_PARAMETERS if parameters is None else parameters)
    key = 'object_seed%d_task%d' % (stage['seed'], stage['task'])
    final = out/key
    binding = make_binding(stage, parameters, code, runtime)
    if verify_complete(final, binding):
        print('CACHE_REUSED_BOUND', key, binding['fingerprints']['binding_sha256'], flush=True)
        return
    temp = out/(key+'.incomplete_'+str(time.time_ns()))
    temp.mkdir()
    start = time.perf_counter()
    write(temp/'STAGE_BINDING.json', binding)
    args = native_args(stage, temp, parameters)
    probe.run(args)
    report = json.loads((temp/'manifest.json').read_text())
    assert report['dropped_current'] == report['dropped_old'] == 0
    new_paths = sorted((Path(args.feature_dir)/str(args.current_task)).glob('*.pt'))
    old_paths = sorted(Path(args.buffer_dir).glob('*.pt'))
    nv, na, nk, nd = probe._load(new_paths, parameters['past_q'])
    ov, oa, ok, od = probe._load(old_paths, parameters['past_q'])
    assert not nd and not od
    tasks = torch.tensor([int(Path(p).name.split('__')[0][1:]) for p in ok])
    assert bool((tasks < args.current_task).all())
    scale = probe._action_stats(na, oa)
    assert scale.tolist() == report['scale_motion_iqr_floor_0p1']
    tick = time.perf_counter()
    assert_consistent(args, parameters)
    rows, batch_summaries = compare(nv, ov, na, oa, scale, tasks, seed=stage['seed'], parameters=parameters)
    table(temp/'batch_trials.csv', rows)
    table(temp/'batch_comparison.csv', batch_summaries)
    summary = read_table(temp/'summary.csv')
    # Classification counts must partition visual matches; no inferred new labels.
    for r in summary:
        r['uncertain_pairs'] = int(r['visual_candidates']) - int(r['share_pairs']) - int(r['bif_pairs'])
        assert r['uncertain_pairs'] >= 0
    table(temp/'summary_with_uncertain.csv', summary)
    report['stage_provenance'] = stage['buffer_evidence']
    report['execution'] = dict(cpu_only=True, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        cuda_initialized=torch.cuda.is_initialized(), torch_threads=torch.get_num_threads(),
        seconds_total=time.perf_counter()-start, supplemental_batch_seconds=time.perf_counter()-tick)
    assert not report['execution']['cuda_initialized']
    report['parameters']['out_dir'] = str(final)
    write(temp/'manifest.json', report)
    # Empty candidates are a valid result, so always create both schema-only CSVs.
    fields = ['visual_threshold','class','new_file','old_file','old_task','visual_cosine','action_distance','both_idle']
    for name in ('retrieved_candidates.csv', 'audit_sample.csv'):
        if not (temp/name).exists():
            table(temp/name, [], fields)
    if make_binding(stage, parameters, code, runtime) != binding:
        raise CacheMismatch('INPUT_CHANGED_DURING_STAGE: no completion record published')
    files = {p.name:sha(p) for p in temp.iterdir() if p.is_file()}
    write(temp/'COMPLETE.json', dict(status='COMPLETE', outputs_sha256=files, time=stamp(),
        seconds=time.perf_counter()-start, binding=binding['fingerprints'], formal_training_resumed=False))
    assert not final.exists()
    os.replace(temp, final)
    print('STAGE_COMPLETE',key,round(time.perf_counter()-start,2),'seconds',flush=True)


def summarize(out, stages):
    all_rows, old_rows, batches, times = [], [], [], []
    for s in stages:
        if s['status'] != 'READY':
            continue
        p = out/('object_seed%d_task%d'%(s['seed'], s['task']))
        prefix = dict(suite='LIBERO_OBJECT',seed=s['seed'],current_task=s['task'])
        all_rows += [dict(prefix,**r) for r in read_table(p/'summary_with_uncertain.csv')]
        old_rows += [dict(prefix,**r) for r in read_table(p/'by_old_task.csv')]
        batches += [dict(prefix,**r) for r in read_table(p/'batch_comparison.csv')]
        times.append(dict(prefix,seconds=json.loads((p/'COMPLETE.json').read_text())['seconds']))
    table(out/'summary.csv',all_rows)
    table(out/'by_old_task.csv',old_rows)
    table(out/'batch_comparison.csv',batches)
    write(out/'EFFICIENCY.json',dict(stages=times,peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        torch_threads=torch.get_num_threads(),cuda_initialized=torch.cuda.is_initialized()))
    write(out/'SUMMARY.json',dict(status='EXPERIMENT_A_COMPLETE_ON_AVAILABLE_OBJECT_ASSETS',
        completed_stage_seed_units=len(times),seeds=list(SEEDS),tasks_with_pairs=[7,8,9],
        no_history_stage=6,unavailable_suites=['LIBERO_GOAL','LIBERO_50'],
        physical_state_matching='NOT_VALIDATED',human_audit='NOT_RUN',experiment_b='NOT_RUN',
        abcr_training='NOT_RUN',formal_training_resumed=False,
        trajectory_deduplicated_sample_size='UNAVAILABLE',
        scope_limit='Seed42 committed native buffers verified. Seed43/44 deterministic native reconstruction, not historical snapshot evidence.',
        main_metrics=all_rows))
    lines = ['# ABCR 跨任务候选样本对可用性：实验 A（CPU）','',
        '本次只运行现有 feature/actions 的筛查，不训练模型、不运行实验 B。候选 share/bif 不是经过物理状态或人工核验的知识共享/动作分叉标签。','',
        '## 范围与固定规则','',
        '- OBJECT seeds42/43/44，各 task7/8/9 对应已到达的历史增量任务；task6 原生缓存为空。GOAL、LIBERO-50 在已有 features 根目录无可用资产，未运行。',
        '- seed42 使用已提交阶段缓存，逐文件核验与原始特征及原生采样选择一致。seed43/44 原始 buffer 已不在，因此从原配置及原源码纯采样函数确定性重建只读路径视图；不伪称存留快照。',
        '- 原生 replay 是每任务50%窗口，容量未触发淘汰；没有新增基座任务回放。',
        '- 直接运行用户原脚本；q=1，视觉阈值0.70/0.80/0.90/0.95，动作阈值0.5/1.5，gripper按 >0 比较，双侧静止为原动作前6维 RMS<0.05。',
        '- 逐阶段当前+旧动作 IQR，逐维下限0.1；这遵循提供脚本，非全阶段同一固定尺度。属于探索性筛查，不是已校准物理尺度。',
        '- 采用方案举例的500次批次模拟（脚本默认300以命令参数覆盖）；原脚本统计保留，另以同5个当前锚点配对比较随机和检索。','',
        '## 严格视觉筛查（0.90；所有阈值见 summary.csv）','',
        '|seed|当前任务|候选对|share|bif|Top5 share覆盖|Top5 bif覆盖|随机批次双类|检索批次双类|',
        '|---|---|---|---|---|---|---|---|---|']
    for r in all_rows:
        if float(r['visual_threshold']) != .9:
            continue
        bs = [x for x in batches if x['seed']==r['seed'] and x['current_task']==r['current_task'] and float(x['visual_threshold'])==.9]
        bdict = {x['strategy']:float(x['fraction_batches_with_both']) for x in bs}
        lines.append('|%s|%s|%s|%s|%s|%.1f%%|%.1f%%|%.1f%%|%.1f%%|'%(r['seed'],r['current_task'],r['visual_candidates'],r['share_pairs'],r['bif_pairs'],
            100*float(r['anchor_share_coverage_retrieved']),100*float(r['anchor_bif_coverage_retrieved']),100*bdict['RANDOM_5_PLUS_5'],100*bdict['RETRIEVE_TOP5_PER_ANCHOR']))
    lines += ['', '## 有效样本量与比较边界','',
        '特征导出只保存 feature/actions；record编号不能直接当作轨迹ID或时间。报告逐批去重 window pair 和旧任务覆盖，但轨迹级有效对数、重叠窗口去重、轨迹bootstrap均不可得。500次模拟也不是500条独立轨迹。',
        '随机5+5与每锚点Top5均打分25对，但检索可用最多25个不同旧窗口，并先扫描5×N_old个余弦；候选预算和检索成本不同。结果反映检索可用性，不证明等成本算法优势。',
        '原生训练使用混合数据集 DistributedSampler，并不固定每批5+5；此处是附件预设的平衡批次可用性模拟。',
        '两视角特征已被语言FiLM调制，不能仅凭高余弦证明状态相同。末时刻7维标签是当前/历史动作，未构造未来动作。',
        '', '## 后续边界','',
        '候选对数量和覆盖率仅能说明本设置下有无可检索的阈值标签。是否值得训练ABCR仍需独立实验B的物理状态与人工抽检；本轮没有该证据，不输出ABCR收益或Go结论。',
        '', '`formal_training_resumed = false`','']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output',type=Path,help='Use a NEW directory for schema2; legacy unbound results are read-only')
    args=ap.parse_args()
    root=args.root.resolve();out=args.output.resolve() if args.output else root/'reports'/NAME
    out.mkdir(parents=True,exist_ok=True)
    # Serialize checks/publication for this output; a second runner fails closed.
    import fcntl
    lock=(out/'.runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sources=execution_sources(root)
    parameters=dict(FROZEN_PARAMETERS)
    runtime=runtime_identity()
    protocol=dict(name=NAME,seeds=list(SEEDS),tasks=list(TASKS),parameters=parameters,cache_schema=2,
        both_idle_raw_rms=.05,scale='per-stage pooled current+old IQR, min0.1',
        subsample_files=False,cpu_threads=1,runtime=runtime,
        sources=sources,user_protocol_sha256=sha(Path(__file__).resolve().parent/'USER_PROTOCOL.md'),
        training=False,formal_training_resumed=False,experiment_b=False)
    if (out/'PROTOCOL.json').exists():
        if json.loads((out/'PROTOCOL.json').read_text()) != protocol:
            raise CacheMismatch('FROZEN_PROTOCOL_CODE_MISMATCH: use new output directory, do not relabel legacy results')
    else:
        write(out/'PROTOCOL.json',protocol)
    context=dict(time=stamp(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),
        dirty=subprocess.check_output(['git','status','--short','-uno'],cwd=root,text=True),
        python=platform.python_version(),pytorch=torch.__version__,cpu=platform.processor(),
        cpu_count=os.cpu_count(),cuda_visible_devices='',cuda_initialized=torch.cuda.is_initialized(),
        disk_free_bytes={str(p):shutil.disk_usage(p).free for p in (Path('/data'),Path('/root'))},
        feature_roots=sorted(p.name for p in (root/'features').iterdir() if p.is_dir()),
        formal_training_resumed=False,formal_sources_modified=False,
        source_sha256={str(p):sha(root/p) for p in ['libero/lifelong/algos/mlr.py','libero/lifelong/models/complete_policy.py','libero/lifelong/models/base_policy.py','libero/lifelong/save_feature.py']})
    # Do not regenerate missing audit records around previously completed stages.
    if list(out.glob('object_seed*_task*/COMPLETE.json')):
        for name in ('INPUT_MANIFEST.json','STAGE_MANIFEST.json','EXPERIMENT_CONTEXT.json'):
            if not (out/name).is_file():
                raise CacheMismatch('COMPLETED_RUN_AUDIT_RECORD_MISSING:'+name)
    # Existing context/status/manifests remain untouched if any input check fails.
    stages=audit(root,out)
    if not (out/'EXPERIMENT_CONTEXT.json').exists():
        write(out/'EXPERIMENT_CONTEXT.json',context)
    ready=[s for s in stages if s['status']=='READY']
    reused=[]
    for s in ready:
        final=out/('object_seed%d_task%d'%(s['seed'],s['task']))
        reused.append(verify_complete(final,make_binding(s,parameters,sources,runtime)))
    if all(reused):
        print('ALL_BOUND_STAGES_REUSED',len(ready),'no report overwritten',flush=True)
        return
    for s in stages:
        if s['status']=='READY':
            write(out/'STATUS.json',dict(status='RUNNING',seed=s['seed'],task=s['task'],time=stamp(),formal_training_resumed=False))
            if execution_sources(root) != sources:
                raise CacheMismatch('SOURCE_CHANGED_DURING_RUN')
            run_stage(s,out,sources,runtime,parameters)
    summarize(out,stages)
    write(out/'STATUS.json',dict(status='COMPLETE_AVAILABLE_ASSETS',time=stamp(),formal_training_resumed=False))


if __name__=='__main__':
    main()

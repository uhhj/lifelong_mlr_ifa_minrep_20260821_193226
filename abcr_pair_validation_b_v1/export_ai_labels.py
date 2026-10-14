"""Package actual AI image-review decisions with honest, input-bound provenance.

This module does not infer labels, train a model or calibrate a threshold.
It deliberately leaves the original human-label pack and protocol untouched.
"""
import argparse
import csv
import hashlib
import html
import io
import json
import os
from collections import Counter
from pathlib import Path

SOURCE = 'AI_VISUAL_ANNOTATION'
VERSION = 'ABCR_B_V2_AI_LABELS'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_new(path, text):
    """Resume identical exports, refuse to overwrite changed evidence."""
    path = Path(path)
    data = text.encode('utf-8')
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Existing output differs: %s' % path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def json_text(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def csv_text(rows, fields):
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows({k: r[k] for k in fields} for r in rows)
    return buf.getvalue()


def validate_decisions(decisions, n):
    if len(decisions) != n or [d['index'] for d in decisions] != list(range(n)):
        raise ValueError('Missing, duplicate or reordered review indices')
    for d in decisions:
        if d['label'] not in ('0', '1', 'U'):
            raise ValueError('Invalid decision')
        if d['confidence'] not in ('low', 'medium', 'high') or not d['reason'].strip():
            raise ValueError('Missing confidence/reason')


def trajectory(record_id):
    return record_id.rsplit('/t', 1)[0]


def export(pack, decisions_path, protocol, out, receipt):
    manifest_path = pack / 'ANNOTATION_MANIFEST.csv'
    manifest = list(csv.DictReader(manifest_path.open(encoding='utf-8')))
    if len(manifest) != 450 or any(r['label'] for r in manifest):
        raise ValueError('Expected original 450-pair blank annotation pack')
    source_complete = json.loads(receipt.read_text(encoding='utf-8'))
    if sha(manifest_path) != source_complete['outputs']['ANNOTATION_MANIFEST.csv']:
        raise ValueError('Input manifest changed')
    decisions = json.loads(decisions_path.read_text(encoding='utf-8'))
    validate_decisions(decisions, len(manifest))
    protocol_sha = sha(protocol)
    rows = []
    for original, decision in zip(manifest, decisions):
        rel_image = 'annotation/images/' + original['pair_id'] + '.png'
        image_sha = sha(pack / rel_image)
        if image_sha != source_complete['outputs'][rel_image]:
            raise ValueError('Image changed: ' + rel_image)
        row = {k: original[k] for k in ('pair_id', 'role', 'id_a', 'id_b')}
        row.update(index=decision['index'], decision=decision['label'],
                   label='' if decision['label'] == 'U' else decision['label'],
                   uncertain=decision['label'] == 'U', code=decision['code'],
                   confidence=decision['confidence'], note=decision['reason'],
                   label_source=SOURCE, human_verified=False,
                   annotator='Codex multimodal assistant', protocol_version=VERSION,
                   protocol_sha256=protocol_sha, image_sha256=image_sha)
        rows.append(row)
    trajectories = {role: {trajectory(r[k]) for r in rows if r['role'] == role
                          for k in ('id_a', 'id_b')}
                    for role in ('calibration', 'validation')}
    if trajectories['calibration'] & trajectories['validation']:
        raise ValueError('Calibration/validation trajectory leakage')
    counts = {role: {label: sum(r['role'] == role and r['decision'] == label for r in rows)
                     for label in ('1', '0', 'U')}
              for role in ('calibration', 'validation')}
    if sum(counts['calibration'].values()) != 300 or sum(counts['validation'].values()) != 150:
        raise ValueError('Role membership changed')
    summary = dict(status='AI_ANNOTATION_COMPLETE_NOT_EXPERIMENT_B_PASS',
                   protocol_version=VERSION, label_source=SOURCE, human_verified=False,
                   pairs_total=len(rows), counts=counts,
                   binary_label_count=sum(r['decision'] != 'U' for r in rows),
                   uncertain_count=sum(r['decision'] == 'U' for r in rows),
                   trajectory_counts={k: len(v) for k, v in trajectories.items()},
                   trajectory_overlap=0,
                   minimum_calibration_label_counts_satisfied=(counts['calibration']['1'] >= 20
                                                               and counts['calibration']['0'] >= 40),
                   tau_v=None, visual_calibration='NOT_RUN', incremental_screen='NOT_RUN',
                   unmodulated_features='PENDING_EXPORT',
                   eligible_for_v2_ai_calibration=(counts['calibration']['1'] >= 20
                                                  and counts['calibration']['0'] >= 40),
                   eligible_for_original_human_calibration=False,
                   next_step='Export and verify frozen pre-FiLM features, then apply v2 calibration',
                   formal_training_resumed=False,
                   limitations=['Single AI annotator, no independent human verification or inter-rater evidence',
                                'Static RGB supports visual comparability judgments, not dynamic physical equivalence',
                                'Task/demo/time identifiers and split roles were visible; score/action evidence was not used',
                                'Uncertain pairs remain in coverage denominators and are excluded from binary calibration',
                                'Label counts do not establish visual gate accuracy or authorize ABCR training'])
    binding = dict(protocol_sha256=protocol_sha, decisions_sha256=sha(decisions_path),
                   original_manifest_sha256=sha(manifest_path),
                   original_pack_complete_sha256=sha(receipt),
                   exporter_sha256=sha(__file__), label_source=SOURCE,
                   human_verified=False, annotation_date='2026-10-14',
                   evidence='30 contact sheets reviewed pair-by-pair; original image enlarged for ambiguous detail',
                   rubric='PROTOCOL_V2_AI_LABELS.md',
                   score_blind=True, action_blind=True, role_blind=False,
                   labeler_independence='independent of calibrated scores, not independent human ground truth')
    files = {'AI_LABELS.csv': csv_text(rows, list(rows[0])),
             'DECISIONS.json': json_text(decisions),
             'SUMMARY.json': json_text(summary), 'PROVENANCE.json': json_text(binding),
             'PROTOCOL_V2_AI_LABELS.md': protocol.read_text(encoding='utf-8')}
    for role in ('calibration', 'validation'):
        files[role.upper() + '_AI_LABELS.csv'] = csv_text(
            [r for r in rows if r['role'] == role and r['decision'] != 'U'], list(rows[0]))
    files['UNRESOLVED_PAIRS.csv'] = csv_text([r for r in rows if r['decision'] == 'U'], list(rows[0]))
    strata = []
    for role in ('calibration', 'validation'):
        pairs = sorted({r['pair_id'].rsplit('_', 1)[0] for r in rows if r['role'] == role})
        for pair in pairs:
            subset = [r for r in rows if r['pair_id'].rsplit('_', 1)[0] == pair]
            c = Counter(r['decision'] for r in subset)
            strata.append(dict(role=role, task_pair=pair, total=len(subset),
                               positive=c['1'], negative=c['0'], uncertain=c['U']))
    files['LABEL_COUNTS_BY_PAIR.csv'] = csv_text(strata, list(strata[0]))
    lines = ['# 实验 B：v2 AI 视觉标注交付', '',
             '本次完成450对图像审阅。标签来源为AI，human_verified=false；不冒称人工标签。', '',
             '| 集合 | 正例 | 负例 | 不确定 | 总数 |', '|---|---:|---:|---:|---:|']
    for role in counts:
        c = counts[role]
        lines.append('| %s | %d | %d | %d | %d |' % (role, c['1'], c['0'], c['U'], sum(c.values())))
    lines.extend(['', '校准集20正/40负数量下限：%s。该下限只检查可计算性，不证明标签正确。' % summary['minimum_calibration_label_counts_satisfied'],
                  '全部%d个不确定项保留，导出的二分类表仅含明确0/1，不把U改为负例。' % summary['uncertain_count'], '',
                  '校准与验证轨迹零重叠；每条标签绑定原图SHA、原始清单SHA、协议SHA和理由。',
                  '450张图像与原始PACK_COMPLETE逐项核验；原人工空白表和历史报告未改。', '',
                  '## 协议变更与证据边界', '',
                  '用户授权以AI视觉标注替代本轮人工标签来源；数值物理门、阈值计算公式、基础任务限定和轨迹隔离规则保持不变。',
                  '本次单一AI评审没有独立人工复核，也没有标注者间一致性证据。图片可见task/demo/time和集合名称，',
                  '故不声称对这些元数据盲审；标注没有使用视觉余弦、专家未来动作或类别配额。',
                  '图像以接触表逐对查看，疑难处放大原图；原始相机低分辨率及静态截图限制仍然存在。',
                  '篮筐阶段或抓取关系相近标正，目标相对手的位置/支撑关系不同标负，物体类别不同本身不判负。',
                  '这些是可复核的视觉判断，不是完整物理状态等价证明；正例也不代表共享动作或共享知识。', '',
                  '## 实验状态', '',
                  '标签环节完成；pre-FiLM特征导出与输入对齐核验仍待完成，视觉阈值未计算，增量SHARE/BIF筛查未执行。',
                  '后续校准须同时携带PROVENANCE和SUMMARY，并将指标称为“相对于AI视觉标签的一致性”。',
                  '不得把这些CSV冒充原v1所要求的独立人工真值。', '',
                  '`formal_training_resumed=false`。'])
    files['FINAL_REPORT.md'] = '\n'.join(lines) + '\n'
    for name, content in files.items():
        atomic_new(out / name, content)
    complete = dict(status='AI_LABEL_EXPORT_COMPLETE', label_source=SOURCE, human_verified=False,
                    bindings=binding, outputs={name: sha(out / name) for name in sorted(files)},
                    formal_training_resumed=False)
    atomic_new(out / 'COMPLETE.json', json_text(complete))
    return summary


def render_review(pack, report, path):
    rows = list(csv.DictReader((report / 'AI_LABELS.csv').open(encoding='utf-8')))
    cards = []
    for r in rows:
        image = os.path.relpath(pack / 'annotation/images' / (r['pair_id'] + '.png'), path.parent).replace('\\', '/')
        # The old PNG header is historical blank-pack text, explicitly superseded in the page.
        cards.append('<article data-label="{label}"><h3>{idx} · {pair}</h3>'
                     '<p class="decision">AI判断：{label} · 置信度：{confidence}</p>'
                     '<p>{note}</p><a href="{image}" target="_blank"><img loading="lazy" src="{image}"></a></article>'.format(
                         label=html.escape(r['decision']), idx=r['index'], pair=html.escape(r['pair_id']),
                         confidence=r['confidence'], note=html.escape(r['note']), image=html.escape(image)))
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ABCR B v2 · AI视觉标签</title>'
    page += '<style>body{font:16px system-ui;max-width:1700px;margin:auto;padding:24px;background:#f4f6f8}header{position:sticky;top:0;background:#fff;padding:16px;z-index:1;border-bottom:2px solid #2563eb}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}article{background:white;padding:16px;border-radius:10px}img{width:100%;height:auto}.decision{font-weight:bold}select{font-size:16px;padding:5px}</style>'
    page += '<header><h1>450对 AI 视觉标注 · v2</h1><p>来源：AI_VISUAL_ANNOTATION；human_verified=false。1=可比，0=不可比，U=不确定。</p><p>原图顶部的“LABEL BLANK / human review”属于原始空白包，当前判断以本页AI记录为准。图像不可单独证明动态状态等价。</p><label>筛选 <select id="filter"><option value="all">全部</option><option>1</option><option>0</option><option>U</option></select></label> <span id="count"></span></header><main>'
    page += ''.join(cards) + '</main><script>const sel=document.querySelector("#filter");function update(){let n=0;document.querySelectorAll("article").forEach(a=>{const show=sel.value==="all"||a.dataset.label===sel.value;a.hidden=!show;if(show)n++;});document.querySelector("#count").textContent=n+" 对";}sel.addEventListener("change",update);update();</script></html>'
    atomic_new(path, page)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('pack', 'decisions', 'protocol', 'out', 'receipt'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--review-html', type=Path)
    args = parser.parse_args()
    print(json_text(export(args.pack, args.decisions, args.protocol, args.out, args.receipt)))
    if args.review_html:
        render_review(args.pack, args.out, args.review_html)

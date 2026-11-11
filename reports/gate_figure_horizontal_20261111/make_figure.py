"""Restyle the user's gate figure using unchanged original scalar evidence."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
EXEC = HERE.parent
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)
sources = {
    'OBJECT': EXEC / '20261028_support_gate_full_s_object/final_readonly_20261029_201235',
    'GOAL': EXEC / '20261029_support_gate_full_s_goal/final_readonly_20261030_111546',
}
for name in ['times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf']:
    font_manager.fontManager.addfont(str(Path('C:/Windows/Fonts') / name))
plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 10.5,
    'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman', 'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold', 'axes.unicode_minus': True,
})
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {}
audit = {'status': 'PASS_DATA_AND_LAYOUT', 'suites': {}, 'new_training_calls': 0,
         'new_environment_episodes': 0, 'font': 'Times New Roman', 'layout': '1x4',
         'legend_and_description_position': 'below_all_panels',
         'panel_titles_position': 'below_each_panel'}
fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.5),
                         gridspec_kw={'width_ratios': [1, 1.05, 1, 1.05]})
fig.subplots_adjust(left=.047, right=.987, top=.97, bottom=.39, wspace=.65)
accept, reject = '#007F75', '#D66A46'
order = [0, 1, 2, 4, 3, 5, 6, 7]
names = ['All pass', 'BC', 'Exec', 'Ret.', 'BC + Exec', 'BC + Ret.', 'Exec + Ret.', 'All three']
def rel(q, c):
    return 200 * (q-c) / (abs(q)+abs(c)) if abs(q)+abs(c) else 0.

for si, (suite, folder) in enumerate(sources.items()):
    rows = []
    for seed in [41, 42, 43]:
        n = 0
        for stage in [7, 8, 9]:
            p = folder / f'light/seed{seed}/GATES_stage{stage}.jsonl'
            manifest[str(p.relative_to(EXEC))] = sha(p)
            raw = [json.loads(line) for line in p.read_text(encoding='utf8').splitlines() if line]
            assert len({(r['epoch'], r['batch']) for r in raw}) == len(raw)
            for r in raw:
                c, q = r['C'], r['Q']
                checks = dict(bc_new=q['bc_new'] <= c['bc_new'],
                              exec_new=q['exec_new'] <= c['exec_new'],
                              retain_improved=q['retain_support'] < c['retain_support'])
                assert checks == r['checks']
                selected = all(checks.values())
                assert selected == (r['accepted'] == 'Q')
                mask = int(not checks['bc_new']) + 2*int(not checks['exec_new']) + 4*int(not checks['retain_improved'])
                rows.append((rel(q['exec_new'], c['exec_new']),
                             rel(q['retain_support'], c['retain_support']), selected, mask))
            n += len(raw)
        assert n == {'OBJECT': 9100, 'GOAL': 6700}[suite]
    ax = axes[2*si]
    for selected, color in [(False, reject), (True, accept)]:
        points = [r for r in rows if r[2] == selected]
        ax.scatter([r[0] for r in points], [r[1] for r in points], s=2,
                   color=color, alpha=.22, edgecolors='none', rasterized=True)
    for axis in ['x', 'y']:
        getattr(ax, f'set_{axis}scale')('symlog', linthresh=.01, linscale=.65, base=10)
        getattr(ax, f'set_{axis}lim')(-210, 210)
        getattr(ax, f'set_{axis}ticks')([-100, -1, 0, 1, 100])
        getattr(ax, f'set_{axis}ticklabels')(['−100', '−1', '0', '1', '100'])
    ax.axhline(0, color='#6B7280', lw=.7)
    ax.axvline(0, color='#6B7280', lw=.7)
    ax.set_xlabel('Relative Δ execution error (%)')
    ax.set_ylabel('Relative Δ retention error (%)')
    ax.text(.5, -.24, f'({chr(97+2*si)}) {suite} | Candidates',
            transform=ax.transAxes, ha='center', va='top', fontsize=12)
    ax.text(.025, .97, f'n = {len(rows):,} blocks', transform=ax.transAxes, va='top',
            bbox=dict(facecolor='white', alpha=.9, edgecolor='none', pad=2))
    ax = axes[2*si+1]
    counts = Counter(r[3] for r in rows)
    vals = [100*counts[m]/len(rows) for m in order]
    assert abs(sum(vals)-100) < 1e-10
    ax.barh(range(8), vals, color=[accept]+[reject]*7, height=.67, zorder=3)
    ax.set_yticks(range(8), names)
    ax.invert_yaxis()
    ax.set(xlim=(0, 60), xticks=[0, 20, 40, 60], xlabel='Share of all blocks (%)')
    ax.text(.5, -.24, f'({chr(98+2*si)}) Violated conditions',
            transform=ax.transAxes, ha='center', va='top', fontsize=12)
    ax.grid(axis='x', color='#DCE2E5', lw=.55)
    ax.set_axisbelow(True)
    for j, v in enumerate(vals):
        ax.text(v+.8, j, f'{v:.1f}', va='center', fontsize=9.5)
    audit['suites'][suite] = dict(blocks=len(rows), percentages=vals,
                                all_decisions_recomputed=True, all_points_retained=True)

fig.legend(handles=[Line2D([], [], ls='none', marker='o', color=color, label=label)
                    for color, label in [(accept, 'Retention candidate accepted'),
                                         (reject, 'Retention candidate rejected')]],
           loc='center', bbox_to_anchor=(.5, .165), ncol=2, frameon=False,
           columnspacing=3.2, handletextpad=.4, fontsize=11)
fig.text(.5, .095, 'DEBR (λR = 0.1) | All blocks | Retention − Base', ha='center', fontsize=10, color='#444444')
fig.text(.5, .045, 'Symmetric-log scatter axes', ha='center', fontsize=10, color='#555555')
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
assert all(t.get_fontproperties().get_name() == 'Times New Roman' for t in texts)
outside = [t.get_text() for t in texts if not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[0])
           or not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[1])]
assert not outside, outside
audit['all_text_font_verified'] = True
audit['text_outside_canvas'] = outside
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(OUT / f'gate_tradeoffs_horizontal_times.{ext}', dpi=600, facecolor='white')
svg = OUT / 'gate_tradeoffs_horizontal_times.svg'
svg.write_bytes(('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf8').splitlines())+'\n').encode('utf8'))
fig.savefig(OUT / 'preview.png', dpi=150, facecolor='white')
assert all(sha(EXEC/p) == h for p, h in manifest.items())
(HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
(HERE/'DATA_LAYOUT_AUDIT.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf8')
print(json.dumps(audit, indent=2))

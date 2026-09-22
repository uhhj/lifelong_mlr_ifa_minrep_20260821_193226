"""Render existing offline results only; no solver or training execution."""
import csv,json
from pathlib import Path
from .numeric import write,sha
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/qp_workingset_refinement_diag_v1'
def read(name):return json.loads((OUT/name).read_text())
def main():
    g=read('GEOMETRY.json');r=read('REGRESSION.json');q=read('B58_REFINEMENT.json')
    original=read('B58_RECOMPUTATION.json');audit=read('INDEPENDENT_AUDIT.json')
    assert audit['status']=='PASS'
    lines=['# Task8 epoch12 Python batch58: offline working-set refinement',
    '', '## Scope', '',
    'Original `ec18946` QP arrays and JSON were verified against the Git blobs. All existing inputs, formal runtime sources and the recovery index remained unchanged.',
    '`training_active=false`. No model was loaded, no parameter writeback, no formal v5 and no training restart.',
    'KKT stays at1e-8. QP, normalization, constraints and lambda mapping are untouched. Precision is80 decimal digits from exact binary64 values.',
    '', '## Failure mechanism', '',
    '| Stored candidate | Float64 KKT | Dominant coordinate/component |',
    '|---|---:|---|',
    '| Standard30k | 1.8797007229e-7 | coordinate29 complementarity; coordinate15 also violates primal positivity |',
    '| Original L-BFGS-B | 1.8797004880e-7 | same complementarity bottleneck at29 |',
    '| Trust final | 6.7897659823e-8 | primal positivity at40; complementarity also fails |',
    '| Trust minimum-KKT iterate56 | 1.9770926601e-8 | complementarity at29; still FAIL |',
    '', 'Indices are0-based. Full exceeding-coordinate lists, x and primal values are in `B58_RECOMPUTATION.json` and `B58_COORDINATE_RESIDUALS.csv`.',
    '',
    'The standard result already has the same33-variable positive support as the certified refined solution. One direct stationary solve on this support passes all64 original coordinates without adding or removing a variable. Thus this instance is consistent with insufficient stationarity accuracy inside an already correct working set, not an unattainable1e-8 gate.',
    'At coordinate29, x≈13.2243 multiplies a stationarity residual≈-1.4214e-8 into complementarity≈1.8797e-7. The trust final point has a different dominant residual; its internal xtol success does not certify the project KKT.',
    '', '## Geometry of this instance', '',
    f"- A symmetry max error: {g['symmetry_max_abs']:.16g}.",
    f"- cond(A): {g['cond_A']:.12g}; eigenvalue range: [{g['eig_min']:.12g}, {g['eig_max']:.12g}].",
    f"- Numerical ranks: default{g['rank_default']}; relative1e-12/1e-10/1e-8: {g['rank_relative']}.",
    f"- Maximum unique off-diagonal absolute correlation: {g['max_off_diagonal_abs']:.12g}.",
    f"- Unique pair counts: {g['high_correlation_unique_pairs']}.",
    '- Standard/L-BFGS-B33-variable working-set condition number:1.4554027486e7.',
    '- Trust start support contains all64 variables, condition number4.2561752286e7;31 boundary removals recover the same33-variable support.',
    '- All actual working-set matrices and condition numbers are recorded in candidate TRACE.json. No real fixture encountered a singular linear solve.',
    '- Spectrum uses the symmetric part for reporting only; actual linear solves and KKT use original unsymmetrized A.',
    '', '## Four fixed B58 starts', '',
    '| Start | Working-set iterations | Linear solves | Seconds | Float64 KKT | Decimal80 KKT |',
    '|---|---:|---:|---:|---:|---:|']
    for name,m in q.items():
        lines.append(f"| {name} | {m['iterations']} | {m['linear_solves']} | {m['seconds']:.6f} | {m['parts']['kkt']:.6g} | {float(m['decimal80']['kkt']):.6g} |")
    lines += ['', 'The minimum-KKT trust iterate was used only as an input; it was never accepted as-is. All four starts reached byte-identical saved refined x.',
    '', '## Same-rule regression', '',
    'These are diagnostic refinements from the original standard path, not a newly approved formal fallback order. F1 is already accepted by that path and is returned bit-for-bit unchanged.',
    '', '| Fixture | Standard KKT | Refinement iterations | Seconds | Float64 KKT | Decimal80 KKT | Result |',
    '|---|---:|---:|---:|---:|---:|---|']
    csv_rows=[]
    for name,m in r.items():
        lines.append(f"| {name} | {m['standard_input_kkt']:.6g} | {m['iterations']} | {m['seconds']:.6f} | {m['parts']['kkt']:.6g} | {float(m['decimal80']['kkt']):.6g} | {m['status']} |")
        csv_rows.append(dict(fixture=name,standard_kkt=m['standard_input_kkt'],iterations=m['iterations'],
            linear_solves=m['linear_solves'],seconds=m['seconds'],kkt64=m['parts']['kkt'],
            kkt80=m['decimal80']['kkt'],strict_pass=m['strict_pass'],status=m['status']))
    lines += ['', 'Timing includes project KKT/Decimal checks inside refinement, but excludes input reading, source hashing, geometry reporting and prior standard solving. F1 standard-path regeneration separately cost3.848931seconds; F1 refinement cost is verification-only. These are single CPU timing observations, not end-to-end training speedups.',
    '', '## Verification and limits', '',
    '-8 unit tests passed, including exact-active-set enumeration for12 synthetic4D strictly convex QPs, removal/re-entry, singular handling, deterministic replay, budget and cycle detection.',
    '- Formal float64 checker matches independently implemented float64 components for every saved refined candidate.',
    '-9 saved candidates additionally passed an independent exact-rational KKT audit. All original64 coordinates were checked; Decimal80 results agree to less than1e-65.',
    '- No ridge, pseudoinverse, rank truncation, eigenvalue clipping or permanently removed constraints.',
    '- No unresolved real fixture in this set. Singular/degenerate working systems remain an explicit fail-stop case; only synthetic singular handling was exercised.',
    '- Working-set re-entry is implemented and unit-tested; these real starts needed zero re-entries. Five selected historical fixtures do not prove robustness on all future QPs.',
    '- No actual float32 writeback, optimizer/scheduler/RNG parity or continuation validation was performed in this round.',
    '', '`training_active=false`.', '']
    with (OUT/'regression.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines))
    write(OUT/'POSTPROCESS_PROVENANCE.json',dict(files={str(p.relative_to(ROOT)):sha(p)
        for p in (Path(__file__),Path(__file__).with_name('audit.py'))},training_active=False))
if __name__=='__main__':main()

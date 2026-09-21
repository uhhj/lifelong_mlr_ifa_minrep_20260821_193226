"""Offline regression for the v3 dual-QP solver (L-BFGS-B fallback).

Design (environment-aware):
  A. Forced-path checks on the captured iterates (environment independent):
     - F1: 20k continuation from captured post-10k x must reach KKT <= 1e-8
           without any fallback (matches historical 6.9231e-9).
     - F2/B185: L-BFGS-B fallback from the captured 30k x must reach
           float64 KKT <= 1e-8 and 80-digit KKT <= 1e-8.
  B. Full v3 solve() on each instance:
     - final float64 KKT <= 1e-8 via the project checker;
     - 80-digit KKT of the accepted solution <= 1e-8;
     - determinism (two runs bit-identical lam);
     - lambda mapping consistency.
  C. Normal path: b<=0, u_norm=0, well-conditioned QP identical to v2 solver.
  D. With --strict-parity (server run, same SciPy that produced the captures):
     - the v3 standard path must reproduce the captured x within 1e-12
       (F2/B185) and the fallback trigger status must match history
       (F1: none, F2/B185: trigger).
"""
import argparse
import json
import os
import sys
import time
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solver import FALLBACK_OPTIONS, dual_kkt, solve  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

ROOT = Path(os.environ.get('V3_REGRESSION_ROOT',
                           'C:/Users/Lenovo/Documents/ChatGPT/MLR'))
V2_SOLVER = os.environ.get('V3_V2_SOLVER', str(ROOT/'tmp/v2_solver.py'))
ASSETS = {
    'F1': dict(path=Path(os.environ.get('V3_F1_PATH', str(ROOT/'failing_qp_trace.npz'))),
               x_key='x', captured_stage='post_10k',
               historical_standard_kkt=1.5633119541946794e-08,
               expect_fallback_server=False),
    'F2': dict(path=Path(os.environ.get('V3_F2_PATH', str(ROOT/'tmp/qp_assets/F2_real_before_extended.npz'))),
               x_key='x', captured_stage='post_30k',
               historical_standard_kkt=1.0038755857838533e-08,
               expect_fallback_server=True),
    'B185': dict(path=Path(os.environ.get('V3_B185_PATH', str(ROOT/'tmp/qp_assets/b185_before_extended.npz'))),
                 x_key='x_before', captured_stage='post_30k',
                 historical_standard_kkt=4.864608038612772e-08,
                 expect_fallback_server=True),
}
KKT_TOL = 1e-8
getcontext().prec = 80


def decimal_kkt(A, b, x):
    Ad = [[Decimal.from_float(float(v)) for v in row] for row in A]
    bd = [Decimal.from_float(float(v)) for v in b]
    xd = [Decimal.from_float(float(v)) for v in x]
    n = len(xd)
    primal_pos = Decimal(0); nonneg = Decimal(0); comp = Decimal(0)
    for i in range(n):
        acc = Decimal(0)
        Ai = Ad[i]
        for j in range(n):
            acc += Ai[j] * xd[j]
        r = bd[i] - acc
        if r > 0 and r > primal_pos:
            primal_pos = r
        if xd[i] < 0 and -xd[i] > nonneg:
            nonneg = -xd[i]
        c = abs(xd[i] * r)
        if c > comp:
            comp = c
    return float(primal_pos), float(nonneg), float(comp)


def coordinate_polish(A, b, x, sweeps):
    x = x.copy()
    for _ in range(sweeps):
        for i in range(len(x)):
            x[i] = max(0., x[i] + (b[i] - float(A[i] @ x)) / A[i, i])
    return x


def forced_path_check(name, spec):
    """A: run the v3 continuation/fallback directly from the captured iterate."""
    z = np.load(spec['path'], allow_pickle=True)
    A, b, x0 = z['A'], z['b'], z[spec['x_key']]
    out = dict(captured_stage=spec['captured_stage'],
               historical_standard_kkt=spec['historical_standard_kkt'])
    if spec['captured_stage'] == 'post_10k':
        # F1: 20k continuation from the capture must pass without fallback.
        x = coordinate_polish(A, b, x0, 20000)
        pp, nv, cp = dual_kkt(A, b, x)
        kkt = max(pp, nv, cp)
        hp = decimal_kkt(A, b, x)
        out.update(action='continuation_20k_from_capture', kkt_float64=float(kkt),
                   kkt_decimal80=float(max(hp)),
                   passed=bool(kkt <= KKT_TOL and max(hp) <= KKT_TOL))
    else:
        # F2/B185: L-BFGS-B fallback from the captured 30k iterate.
        objective = lambda x: .5 * float(x @ A @ x) - float(b @ x)
        jac = lambda x: A @ x - b
        t0 = time.perf_counter()
        fb = minimize(objective, x0, jac=jac, bounds=[(0, None)] * len(x0),
                      method='L-BFGS-B', options=dict(FALLBACK_OPTIONS,
                                                      maxcor=len(x0)))
        sec = time.perf_counter() - t0
        cand = np.maximum(np.asarray(fb.x, dtype=np.float64), 0.0)
        pp, nv, cp = dual_kkt(A, b, cand)
        kkt = max(pp, nv, cp)
        hp = decimal_kkt(A, b, cand)
        out.update(action='lbfgsb_fallback_from_capture', kkt_float64=float(kkt),
                   kkt_decimal80=float(max(hp)), nit=int(fb.nit), nfev=int(fb.nfev),
                   njev=int(fb.njev), seconds=round(sec, 4),
                   optimizer_success=bool(fb.success), optimizer_message=str(fb.message),
                   objective=float(objective(cand)),
                   passed=bool(kkt <= KKT_TOL and max(hp) <= KKT_TOL))
    return out


def full_solve_check(name, spec, strict_parity):
    """B: full v3 solve() on the instance."""
    z = np.load(spec['path'], allow_pickle=True)
    K, f, u_norm = z['K'], z['f'], float(z['u_norm'])
    valid, scale = z['valid'], z['scale']
    A, b = z['A'], z['b']
    x_saved = z[spec['x_key']]

    t0 = time.perf_counter()
    sol = solve(K, f, u_norm)
    seconds = time.perf_counter() - t0
    lam = np.asarray(sol['lam'])
    fb = sol['fallback']
    x_out = lam[valid] / u_norm * scale
    hp = decimal_kkt(A, b, x_out)
    hp_kkt = max(hp)

    sol2 = solve(K, f, u_norm)
    det = float(np.max(np.abs(np.asarray(sol2['lam']) - lam)))

    rec = dict(fallback_triggered=fb['triggered'],
               expected_fallback_on_server=spec['expect_fallback_server'],
               final_kkt_float64=float(sol['kkt_error']),
               final_kkt_decimal80=float(hp_kkt),
               base_sweeps=sol['base_sweeps'],
               continuation_sweeps=sol['continuation_sweeps'],
               slsqp_iterations=sol['slsqp_iterations'],
               seconds=round(seconds, 3),
               determinism_max_abs_lam_diff=det,
               lam_mapping_max_err=float(np.max(np.abs(
                   lam[valid] - x_out * u_norm / scale))) if u_norm > 0 else 0.0,
               rates_finite=bool(np.isfinite(sol['rates']).all()))
    checks = dict(float64_kkt_within_threshold=bool(sol['kkt_error'] <= KKT_TOL),
                  decimal80_kkt_within_threshold=bool(hp_kkt <= KKT_TOL),
                  deterministic=bool(det == 0.0),
                  rates_finite=rec['rates_finite'])
    if strict_parity:
        if fb['triggered']:
            diff = float(np.max(np.abs(np.asarray(fb['standard_x']) - x_saved)))
            checks['standard_path_parity_with_capture'] = bool(diff <= 1e-12)
            checks['fallback_trigger_matches_history'] = bool(
                fb['triggered'] == spec['expect_fallback_server'])
            rec['standard_path_max_abs_diff_vs_capture'] = diff
        else:
            checks['fallback_trigger_matches_history'] = bool(
                fb['triggered'] == spec['expect_fallback_server'])
    rec['checks'] = checks
    rec['status'] = 'PASS' if all(checks.values()) else 'FAIL'
    return rec


def normal_path_checks():
    out = {}
    K = np.diag([1.0, 2.0, 3.0]); f = -np.ones(3)
    sol = solve(K, f, 1.0)
    out['b_nonpositive'] = dict(
        passed=bool(sol['kkt_error'] == 0.0 and np.abs(sol['lam']).max() == 0.0
                    and not sol['fallback']['triggered']))
    sol = solve(np.diag([1.0, 2.0]), np.ones(2), 0.0)
    out['zero_u_norm'] = dict(
        passed=bool(sol['kkt_error'] == 0.0 and np.abs(sol['lam']).max() == 0.0))
    rng = np.random.RandomState(7)
    M = rng.randn(50, 12)
    K = M.T @ M + np.eye(12)
    u = rng.randn(50)
    f = M.T @ u
    import importlib.util
    spec = importlib.util.spec_from_file_location('v2solver', V2_SOLVER)
    v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
    s2 = v2.solve(K, f, float(np.linalg.norm(u)))
    s3 = solve(K, f, float(np.linalg.norm(u)))
    out['well_conditioned_vs_v2'] = dict(
        v3_fallback_triggered=bool(s3['fallback']['triggered']),
        passed=bool(not s3['fallback']['triggered']
                    and np.array_equal(np.asarray(s2['lam']), np.asarray(s3['lam']))
                    and s2['kkt_error'] == s3['kkt_error']))
    out['status'] = 'PASS' if all(
        v['passed'] for v in out.values() if isinstance(v, dict)) else 'FAIL'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strict-parity', action='store_true',
                    help='enforce bit-level standard-path parity with captures '
                         '(only valid in the environment that produced them)')
    ap.add_argument('--output', default=str(ROOT/'tmp/v3_dev/offline_regression.json'))
    args = ap.parse_args()

    report = dict(solver='v3 L-BFGS-B fallback', fallback_options=FALLBACK_OPTIONS,
                  kkt_threshold=KKT_TOL, strict_parity=bool(args.strict_parity),
                  forced_path={}, full_solve={}, normal_path=normal_path_checks())
    ok = True
    for name, spec in ASSETS.items():
        forced = forced_path_check(name, spec)
        full = full_solve_check(name, spec, args.strict_parity)
        report['forced_path'][name] = forced
        report['full_solve'][name] = full
        ok = ok and forced['passed'] and full['status'] == 'PASS'
    ok = ok and report['normal_path']['status'] == 'PASS'
    report['status'] = 'PASS' if ok else 'FAIL'
    out = Path(args.output)
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    tmp.replace(out)
    print(json.dumps(report, indent=2, allow_nan=False))
    print('OFFLINE_REGRESSION', report['status'])
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

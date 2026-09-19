# Runtime map

HEAD: `c74249d0663021c5c1f105a9176eb37b71954e92`

Formal solver: `sample_qp_optimized_v1_continuation_v2/solver.py`; caller `sample_qp_optimized_v1_continuation_v2/core.py::project`; training writeback `sample_qp_lifelong_v1_continuation_v2/step.py::update`. The normalized QP uses `scale=sqrt(diag(K))`, `A=K/(scale_i scale_j)`, `b=f/(scale*u_norm)`, and `primal=b-Ax`. KKT is `max(max(primal,0), max(-x,0), max(abs(x*primal)))`; this is normalized dual-space residual. Original polish is 10,000 coordinate sweeps; continuation adds at most 20,000 sweeps, each sweep visiting every coordinate once. SLSQP iterations are separate. Native AdamW runs before `project`; correction is written by subtracting the returned displacement and no second optimizer step occurs.

Latest F2 traceback reports kkt=1.0038755857838533e-08 after the continuation solver call; this calibration must verify whether its 20k continuation path was actually executed from a real replay.

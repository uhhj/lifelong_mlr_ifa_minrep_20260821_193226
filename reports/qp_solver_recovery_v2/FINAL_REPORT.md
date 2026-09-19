# QP solver recovery v2

F2 was captured from the real step41490 recovery boundary. The pre-extension path reproduced KKT 1.0038755857838533e-8 after 29,999 existing sweeps. The bounded extension reached 5.9600765746348205e-9; an 80-digit Decimal recomputation of the final dump was 9.906086978023084e-12.

G2/G3 checks were started. Unit tests pass. An independent SLSQP restart from the captured F2 point returned the same pre-extension point with KKT 1.0038755857838533e-8, so it is not an independently certified reference solution. No real H/model state is available for the required 0/32/64/96 writeback parity. Therefore G2 is FAIL_REFERENCE_UNRESOLVED and G3 is FAIL_WRITEBACK_PARITY_NOT_RUN.

No formal task8/task9 continuation or closed-loop evaluation was resumed. Strict KKT <=1e-8 remains unchanged.

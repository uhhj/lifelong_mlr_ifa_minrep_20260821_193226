"""Evaluation process bootstrap: apply the existing, unchanged package binding.

Needed by multiprocessing spawn before it unpickles libero.libero worker types.
No model, environment, RNG, or evaluation-rule changes.
"""
from protective_e1.bindings import bind
bind()

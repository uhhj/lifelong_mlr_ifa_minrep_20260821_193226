"""Reuse the migrated server's established sibling-package resolution."""
from pathlib import Path


def bind():
    import libero
    root=Path(__file__).resolve().parents[1]
    assert root in Path(libero.__file__).resolve().parents
    # Same binding as existing cb_ifa_experiment.gpu_runtime.bind(), no install/edit.
    sibling=str(root/'libero')
    if sibling not in libero.__path__:libero.__path__.insert(0,sibling)

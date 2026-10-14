"""Fail-closed cache bindings. Old unbound results are never upgraded implicitly."""
import hashlib
import json
from pathlib import Path
from .io import sha, write


class CacheMismatch(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def freeze_many(values):
    """Compare *all* existing records first; never replace them on mismatch."""
    for path, value in values.items():
        path = Path(path)
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
            raise CacheMismatch('IMMUTABLE_MANIFEST_MISMATCH:' + str(path))
    for path, value in values.items():
        if not Path(path).exists():
            write(path, value)


def file_inventory(paths):
    rows = []
    for path in paths:
        path = Path(path)
        # Hash bytes actually consumed through the current path/symlink.
        rows.append(dict(path=str(path), size=path.stat().st_size, sha256=sha(path)))
    return rows


def make_binding(stage, parameters, code, runtime):
    current = sorted((Path(stage['feature_dir']) / str(stage['task'])).glob('*.pt'))
    old = sorted(Path(stage['buffer_dir']).glob('*.pt'))
    if len(current) != stage['n_current'] or len(old) != stage['n_old']:
        raise CacheMismatch('INPUT_MEMBERSHIP_COUNT_MISMATCH')
    if sorted(p.name for p in old) != stage['old_files']:
        raise CacheMismatch('OLD_INPUT_MEMBERSHIP_MISMATCH')
    inputs = dict(seed=stage['seed'], task=stage['task'], current=file_inventory(current), old=file_inventory(old))
    body = dict(inputs=inputs, code=code, parameters=parameters, runtime=runtime,
        provenance=stage['buffer_evidence'])
    fingerprints = dict(schema=2, input_manifest_sha256=digest(inputs), code_sha256=digest(code),
        parameters_sha256=digest(parameters), runtime_sha256=digest(runtime),
        provenance_sha256=digest(stage['buffer_evidence']))
    fingerprints['binding_sha256'] = digest(fingerprints)
    return dict(fingerprints=fingerprints, **body)


def verify_complete(final, binding):
    final = Path(final)
    if not (final/'COMPLETE.json').exists():
        return False
    record = json.loads((final/'COMPLETE.json').read_text())
    saved = record.get('binding')
    if not saved or saved.get('schema') != 2:
        raise CacheMismatch('LEGACY_CACHE_UNBOUND: preserve old results; use a new output directory')
    if saved != binding['fingerprints']:
        raise CacheMismatch('STAGE_BINDING_MISMATCH: input/code/parameters/runtime/provenance changed')
    outputs = record.get('outputs_sha256', {})
    if 'STAGE_BINDING.json' not in outputs:
        raise CacheMismatch('STAGE_BINDING_ARTIFACT_MISSING')
    for name, expected in outputs.items():
        path = final/name
        if Path(name).name != name or not path.is_file() or sha(path) != expected:
            raise CacheMismatch('CACHED_OUTPUT_MISMATCH:' + name)
    if json.loads((final/'STAGE_BINDING.json').read_text()) != binding:
        raise CacheMismatch('STAGE_BINDING_CONTENT_MISMATCH')
    return True

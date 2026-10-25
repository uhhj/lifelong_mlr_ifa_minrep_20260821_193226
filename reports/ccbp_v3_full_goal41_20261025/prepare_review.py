"""Extract only reviewed text, verify upload manifest; never deploy or train."""
from pathlib import Path
import ast, hashlib, json, zipfile

R = Path(__file__).resolve().parent
P = R.parent / '20261024_ccbp_v2_full'
Z = Path('C:/Users/Lenovo/Downloads/ccbp_v3_directional_prototype.zip')
digest = lambda b: hashlib.sha256(b).hexdigest()
assert digest(Z.read_bytes()) == '093ac51bfa414178d333c4ed1d5299b4d4a28d5b11826d936ec37b274a0761e8'
out = R / 'uploaded_reviewed_text'
assert not out.exists(), 'Do not overwrite uploaded evidence'
with zipfile.ZipFile(Z) as z:
    prefix = 'ccbp_v3_directional_prototype/'
    man = json.loads(z.read(prefix + 'MANIFEST.json'))
    assert set(z.namelist()) == {prefix + n for n in [*man, 'MANIFEST.json']}
    for n, m in man.items():
        assert Path(n).name == n and Path(n).suffix in ('.py', '.md', '.json')
        raw = z.read(prefix + n)
        assert len(raw) == m['bytes'] and digest(raw) == m['sha256']
        if n.endswith('.py'): ast.parse(raw.decode('utf8'))
    out.mkdir()
    for n in [*man, 'MANIFEST.json']: (out/n).write_bytes(z.read(prefix+n))
test = R/'cpu_review'; test.mkdir()
for n in ['losses_ccbp_v3.py', 'test_ccbp_v3.py']:
    (test/n).write_bytes((out/n).read_bytes())
for name in ['legacy_losses.py', 'losses.py']:
    dest = 'original_C_losses.py' if name == 'legacy_losses.py' else 'original_v2_losses.py'
    (test/dest).write_bytes((P/name).read_bytes())
# Resolve legacy import within the copied v2 source, while preserving original bytes above.
(test/'legacy_losses.py').write_bytes((P/'legacy_losses.py').read_bytes())
receipt = dict(status='UPLOAD_TEXT_HASH_AND_AST_PASS_NOT_RUNTIME', zip_sha256=digest(Z.read_bytes()),
               entries=man, archived_C_sha256=digest((P/'legacy_losses.py').read_bytes()),
               archived_v2_sha256=digest((P/'losses.py').read_bytes()),
               real_checkpoint_loads=0, updates=0, environments=0)
(R/'UPLOAD_REVIEW.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
print(json.dumps(receipt, indent=2))

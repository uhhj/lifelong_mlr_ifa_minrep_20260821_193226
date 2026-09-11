"""Verify every delivered file against MANIFEST.sha256.json."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'MANIFEST.sha256.json').read_text(encoding='utf-8'))
    for name,expected in manifest.items():
        path=(root/name).resolve()
        if root not in path.parents:
            raise ValueError('Manifest path outside bundle: '+name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Bundle content differs: '+name)
    print('PASS: %d bundled files match SHA256 manifest'%len(manifest))


if __name__=='__main__': main()

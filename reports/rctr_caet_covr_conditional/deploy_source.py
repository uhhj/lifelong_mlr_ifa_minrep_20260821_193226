"""Install reviewed text source with exact hashes, never overwrite a workspace."""
import hashlib
import json
import sys
from pathlib import Path

bundle = json.loads(Path(sys.argv[1]).read_text())
target = Path(sys.argv[2]).resolve()
assert not target.exists()
target.mkdir(parents=True)
for rel, item in bundle.items():
    dest = (target/rel).resolve()
    assert target in dest.parents
    assert dest.suffix in (".py", ".yaml", ".yml", ".sh", ".md", ".json")
    content = item["text"].encode("utf8")
    assert hashlib.sha256(content).hexdigest() == item["sha256"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
print(json.dumps(dict(status="INSTALLED", files=len(bundle), target=str(target))))

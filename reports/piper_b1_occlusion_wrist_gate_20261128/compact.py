"""Keep frame-level gate decisions as masks without camera or motion payloads."""
from pathlib import Path
import json,sys
def compress(evidence):
    out={}
    for ep,r in evidence.items():
        frames=r['window'];assert [f['i'] for f in frames]==list(range(-8,9))
        names=('intent','coarse','height','seed','edge','size','jaw_x','jaw_depth','metric','aligned')
        strings={k:''.join('?' if f[k] is None else '1' if f[k] else '0' for f in frames) for k in names}
        f=r['expert']
        out[ep]=dict(masks=strings,history_at_expert=f['history'],height_at_expert_mm=f['z'],coarse_distance_at_expert_mm=f['gap_mm'])
    assert len(out)==15
    return out
if __name__=='__main__':
    source=Path(sys.argv[1])
    print(json.dumps(compress(json.loads(source.read_bytes())),indent=2))

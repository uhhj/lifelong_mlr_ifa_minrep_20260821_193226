"""In-memory display transfer only; no image file is written."""
import sys,json,urllib.request,urllib.error,base64,datetime
from pathlib import Path
port=int(sys.argv[1]);kind=sys.argv[2];q={'kind':kind}
if kind=='sheet':q.update(seed=int(sys.argv[3]),method=sys.argv[4],indices=[int(x) for x in sys.argv[5].split(',')],window=sys.argv[6])
try:r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:'+str(port),data=json.dumps(q).encode(),headers={'Content-Type':'application/json'}),timeout=180)
except urllib.error.HTTPError as e:
 detail=e.read().decode();p=Path(__file__).parent/('VIEW_FAILURE_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json');p.write_text(json.dumps({'request':q,'detail':detail,'updates':0,'environments':0},indent=2));print(detail);sys.exit(1)
b=r.read();print(base64.b64encode(b).decode() if kind=='sheet' else b.decode())

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import json, socket, struct, time
def now():
    return datetime.now().astimezone().isoformat(timespec='milliseconds')
ports = sorted(p.name for p in Path('/sys/class/net').iterdir() if (p/'type').read_text().strip()=='280')
def stats():
    return {p:{k:int((Path('/sys/class/net')/p/'statistics'/k).read_text()) for k in ('rx_packets','tx_packets','rx_errors','tx_errors','rx_dropped','tx_dropped')} for p in ports}
counts, local, errors = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
last, codes, bounds = {}, defaultdict(Counter), {}
s=socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
s.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_ERR_FILTER,struct.pack('=I',0x1fffffff))
s.bind(('',))
before=stats()
start=now()
deadline=time.monotonic()+9
while time.monotonic()<deadline:
    s.settimeout(max(.0001,deadline-time.monotonic()))
    try: data, anc, flags, addr=s.recvmsg(72)
    except socket.timeout: break
    p=addr[0]
    cid,n=struct.unpack_from('=IB',data)
    if cid & 0x20000000:
        errors[p][hex(cid & 0x1fffffff)]+=1
        continue
    b=data[8:8+n]
    counts[p][cid]+=1
    if flags & socket.MSG_DONTROUTE: local[p][cid]+=1
    last[p,cid]=b
    if cid in (0x151,0x2a1) or 0x261<=cid<=0x266:
        if len(b)>=6: codes[p,cid][b[5:6].hex() if 0x261<=cid<=0x266 else b.hex()]+=1
    if cid in (0x155,0x156,0x157,0x2a5,0x2a6,0x2a7) and len(b)==8:
        for j,v in enumerate(struct.unpack('>ii',b)):
            k=(p,cid,j)
            lo,hi=bounds.get(k,(v,v))
            bounds[k]=min(lo,v),max(hi,v)
end=now()
after=stats()
s.close()
print('WINDOW',start,end)
for p in ports:
    print('PORT',p,'delta',{k:after[p][k]-v for k,v in before[p].items()},'frames',sum(counts[p].values()),'error_frames',dict(errors[p]),'local_tx',dict(local[p]))
    print('IDS',p,{hex(k):v for k,v in sorted(counts[p].items())})
    for id0 in [0x151,0x2a1,*range(0x261,0x267),0x2a8,0x159]:
        if (p,id0) in last: print('DATA',p,hex(id0),last[p,id0].hex(),dict(codes[p,id0]))
    for name,base in [('target_deg',0x155),('feedback_deg',0x2a5)]:
        keys=[(p,c,j) for c in range(base,base+3) for j in (0,1)]
        if all(k in bounds for k in keys):
            print(name,p,'last',[struct.unpack('>ii',last[p,c])[j]/1000 for _,c,j in keys],'span_deg',[round((bounds[k][1]-bounds[k][0])/1000,3) for k in keys])

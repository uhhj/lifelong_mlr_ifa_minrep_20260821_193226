"""Observe the existing SocketCAN interface; never transmit or reconfigure it."""
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta
import json,select,socket,struct,subprocess,time
def wall(t):return datetime.fromtimestamp(t,timezone(timedelta(hours=8))).isoformat(timespec='milliseconds')
def link():
    proc=subprocess.run(['ip','-j','-s','-d','link','show','dev','can0'],capture_output=True,text=True)
    return json.loads(proc.stdout)[0] if not proc.returncode else {'error':proc.stderr.strip()}
ids=[0x2a1,0x150,0x151,0x470,0x471,*range(0x155,0x15a),*range(0x2a5,0x2a9),*range(0x251,0x257),*range(0x261,0x267)]
before=link();start=time.time();counts=Counter();states=defaultdict(Counter);latest={};limits={};errors=Counter();issue=None
if 'UP' in before.get('flags',[]):
    try:
        with socket.socket(socket.PF_CAN,socket.SOCK_RAW,socket.CAN_RAW) as rx:
            rx.setsockopt(socket.SOL_CAN_RAW,socket.CAN_RAW_FILTER,b''.join(struct.pack('=II',i,0xC00007FF) for i in ids))
            rx.setsockopt(socket.SOL_CAN_RAW,socket.CAN_RAW_ERR_FILTER,struct.pack('=I',0x1fffffff))
            rx.bind(('can0',))
            until=time.monotonic()+3
            while time.monotonic()<until:
                if not select.select([rx],[],[],min(.1,max(0,until-time.monotonic())))[0]:continue
                raw,_,flags,_=rx.recvmsg(16)
                ident,n,d=struct.unpack('=IB3x8s',raw);d=d[:n]
                if ident&0x20000000:
                    errors[f'{ident:08x}:{d.hex()}']+=1
                    continue
                origin='host' if flags&socket.MSG_DONTROUTE else 'bus'
                key=f'{origin}_{ident:03x}';counts[key]+=1;latest[key]=d.hex()
                if ident in (0x2a1,0x150,0x151,0x470,0x471) or 0x261<=ident<=0x266:
                    code=d[5:6].hex() if 0x261<=ident<=0x266 else d.hex()
                    states[key][code]+=1
                if 0x261<=ident<=0x266 and n==8:
                    voltage,driver_c,motor_c,status,bus_current=struct.unpack('>HhbBH',d)
                    vals=(voltage/10,driver_c,motor_c,bus_current)
                    if key not in limits:limits[key]=[[v,v] for v in vals]
                    for pair,v in zip(limits[key],vals):
                        pair[0],pair[1]=min(pair[0],v),max(pair[1],v)
    except OSError as exc:issue=str(exc)
print(json.dumps(dict(start=wall(start),end=wall(time.time()),interface_before=before,interface_after=link(),counts=counts,states=states,latest=latest,voltage_temperature_current_ranges=limits,can_error_frames=errors,issue=issue)))

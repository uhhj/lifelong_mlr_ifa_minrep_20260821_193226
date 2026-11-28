"""Summarize the read-only evidence and cross-check status fields against raw frames."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import struct
import sys

def summarize(directory):
    def read(name):return json.loads((directory/name).read_bytes())
    samples=[]
    for name in ('snapshot.json','archive.json'):
        d=read(name)
        assert not d['issue']
        status=d['states']['bus_2a1']
        codes=[]
        for payload,count in status.items():
            b=bytes.fromhex(payload)
            assert len(b)==8
            codes.append(dict(raw=payload,frames=count,control_mode=b[0],arm_status=b[1],
                              comm_axes=[i+1 for i in range(6) if b[7]&(1<<i)],
                              angle_limit_axes=[i+1 for i in range(6) if b[6]&(1<<i)]))
        drives=[]
        for axis in range(1,7):
            k=f'bus_{0x260+axis:03x}'
            v,t,m,flag,_=struct.unpack('>HhbBH',bytes.fromhex(d['latest'][k]))
            assert f'{flag:02x}' in d['states'][k] and sum(d['states'][k].values())==d['counts'][k]
            drives.append(dict(axis=axis,status_histogram=d['states'][k],last_raw=d['latest'][k],
                               last_enabled=bool(flag&64),range_V=d['voltage_temperature_current_ranges'][k][0],
                               last_voltage_V=v/10,last_driver_degC=t,last_motor_degC=m,
                               high_speed_frames=d['counts'].get(f'bus_{0x250+axis:03x}',0)))
        assert sum(status.values())==d['counts']['bus_2a1']
        before=d['interface_before']['linkinfo'];after=d['interface_after']['linkinfo']
        samples.append(dict(start=d['start'],end=d['end'],archive_sha256=hashlib.sha256((directory/name).read_bytes()).hexdigest(),
                            statuses=codes,drives=drives,bitrate=after['info_data']['bittiming']['bitrate'],
                            can_state=[i['info_data']['state'] for i in (before,after)],
                            can_error_frames=sum(d['can_error_frames'].values()),bus_off_counters=[i['info_xstats']['bus_off'] for i in (before,after)],
                            enable_frames=sum(v for k,v in d['counts'].items() if k.endswith('_471'))))
    log=read('logs.json');ui=read('details.json');last=log['sessions'][0];watch=log['watch'][0]
    return dict(samples=samples,latest_policy=dict(session=last['id'],last_event=last['last'],mtime=last['mtime'],bundle=last['bundle']),
                checked_at=log['now'],last_rolling_can_archive=watch['last'],manual_ui_instances=len(ui['ui_process']),
                can_or_enable_logfiles_in_checked_paths=len(ui['files']),onsite_enable_event_time='unknown')
if __name__=='__main__':
    print(json.dumps(summarize(Path(sys.argv[1])),ensure_ascii=False,indent=2))

"""Extract evidence for the startup failure from read-only snapshots."""
import hashlib
import json
from pathlib import Path
import sys

def compare(d):
    def read(n): return json.loads((d/(n+'.json')).read_bytes())
    dev=read('archive');log=read('logs');can=read('snapshot');history=read('details')
    trace=dev['trace']
    stages=['guard(link,args.task,args.python)','gripper = B1Gripper()','self.device = open_gripper()','with serial.Serial(']
    assert all(any(part in l for l in trace) for part in stages)
    assert 'Permission denied' in trace[-1] and dev['symlink'] in trace[-1]
    assert dev['node']==log['devices'][dev['symlink']]
    assert dev['group'] not in dev['account_groups'] and all(dev['group'] not in p['groups'] for p in dev['desktop_processes'])
    assert log['weights']['B1-ode50-wrist-20261127']['receipt'] and dev['ssh_os_access']==[False,False]
    assert can['states']['bus_2a1']=={'0000000000000000':600}
    assert all(can['states'][f'bus_{0x260+j:03x}']=={'40':60} for j in range(1,7))
    grants=[g for g in history['prior_targeted_permission_changes'] if g['target']==[dev['node']] and g['acl']=='u:ding-101:rw']
    assert grants and history['last_run']['matching_grip_samples']==532
    k=history['kernel']
    disconnect=next(v['time'] for v in k if 'disconnected from ttyUSB0' in v['message'])
    reconnect=next(v['time'] for v in k if 'attached to ttyUSB0' in v['message'])
    assert history['last_run']['end'][:19]<=disconnect[:19]<reconnect[:19]<dev['unique_denial_seconds'][0][:19]
    return dict(observed=dev['time'],denied_seconds=dev['unique_denial_seconds'],error='SerialException / EACCES (13)',
                prior_node_grant=dict(time=grants[-1]['time'],tool='setfacl',scope='one ttyUSB0 node',permission='rw'),
                successful_usb_session=history['last_run'],serial_disconnect=disconnect,serial_reattach=reconnect,node_ctime=history['node_ctime'],
                stack=['runner.guard','B1Gripper.__init__','open_gripper','serial.Serial.open'],
                device=dev['node'],mode=dev['mode'],owner=dev['user'],group=dev['group'],acl=dev['acl'],
                account_in_dialout=dev['group'] in dev['account_groups'],
                desktop_in_dialout=any(dev['group'] in p['groups'] for p in dev['desktop_processes']),
                pre_repair_read_write_access=dev['ssh_os_access'],visible_owners_of_devices=bool(log['holders']),
                candidate=log['weights']['B1-ode50-wrist-20261127'],last_run=log['runs'][0]['session'],
                can_sample=dict(start=can['start'],end=can['end'],state=can['interface_after']['linkinfo']['info_data']['state'],
                            controller_frames=can['states']['bus_2a1'],six_enable_bits=[1]*6,can_errors=sum(can['can_error_frames'].values())),
                input_hashes={n:hashlib.sha256((d/(n+'.json')).read_bytes()).hexdigest() for n in ('logs','details','ui_journal','snapshot','archive')})
if __name__=='__main__':
    print(json.dumps(compare(Path(sys.argv[1])),ensure_ascii=False,indent=2))

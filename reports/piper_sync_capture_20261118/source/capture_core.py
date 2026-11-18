"""Native capture contract and crash-preserving episode writer; no robot control."""
import os,json,time,math,uuid,hashlib,shutil
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parent/'datasets'/'piper_real'
PURPOSES={'commissioning','demonstration','autonomous_training','formal_evaluation'}
def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def valid_sample(state,command,cameras,now_ns,required=('external','wrist'),max_age_ns=120000000,max_skew_ns=60000000):
    reasons=[]; stamps=[]
    if state is None:reasons.append('missing_follower_state')
    else:
        try:
            if len(state['joints_rad'])!=6 or not all(math.isfinite(v) for v in state['joints_rad']):raise ValueError()
            if not math.isfinite(state['gripper_mm']):raise ValueError()
            ts=state['sample_monotonic_ns'];stamps.append(ts)
            if not 0<=now_ns-ts<=max_age_ns:reasons.append('stale_follower_state')
        except (KeyError,TypeError,ValueError):reasons.append('invalid_follower_state')
    if command is None:reasons.append('missing_sent_command')
    else:
        try:
            if command['origin']!='local_can_tx':reasons.append('command_not_local_can_tx')
            if len(command['joints_rad'])!=6 or not all(math.isfinite(v) for v in command['joints_rad']):raise ValueError()
            if not math.isfinite(command['gripper_mm']):raise ValueError()
            ts=command['sample_monotonic_ns']
            if ts>now_ns:reasons.append('future_sent_command')
            if ts!=max(command['joint_sent_mono_ns'],command['gripper_sent_mono_ns']):
                reasons.append('invalid_command_time')
        except (KeyError,TypeError,ValueError):reasons.append('invalid_sent_command')
    for role in required:
        c=cameras.get(role)
        if c is None:reasons.append('missing_camera_'+role);continue
        try:
            ts=c['host_monotonic_ns']
            if not isinstance(ts,int) or not isinstance(c['sequence'],int):raise ValueError()
            stamps.append(ts)
            if not 0<=now_ns-ts<=max_age_ns:reasons.append('stale_camera_'+role)
            a=c.get('image')
            if not isinstance(a,np.ndarray) or a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3 or a.size==0:reasons.append('invalid_image_'+role)
        except (KeyError,TypeError,ValueError):reasons.append('invalid_camera_'+role)
    if stamps and max(stamps)-min(stamps)>max_skew_ns:reasons.append('sensor_time_skew')
    return sorted(set(reasons))
class EpisodeWriter:
    def __init__(self,metadata,data_root=DATA):
        self.metadata=dict(metadata)
        purpose=metadata.get('purpose')
        if purpose not in PURPOSES:raise ValueError('Explicit valid purpose required')
        if not metadata.get('task'):raise ValueError('Task description required')
        if purpose!='commissioning' and not metadata.get('initial_condition_id'):raise ValueError('Initial condition ID required')
        if purpose in ['autonomous_training','formal_evaluation']:
            digest=metadata.get('policy_checkpoint_sha256','')
            if len(digest)!=64 or any(x not in '0123456789abcdef' for x in digest):raise ValueError('Policy SHA256 required')
        data_root=Path(data_root).resolve()
        if not data_root.is_relative_to(ROOT.parent.resolve()):raise ValueError('Dataset must remain under ZPWang')
        if shutil.disk_usage(ROOT).free<2*1024**3:raise OSError('Storage reserve below 2 GiB')
        self.id=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
        self.folder=data_root/purpose/(self.id+'.partial')
        self.folder.mkdir(parents=True,exist_ok=False)
        self.records=(self.folder/'samples.jsonl').open('x',encoding='utf8')
        self.events=(self.folder/'events.jsonl').open('x',encoding='utf8')
        self.commands=(self.folder/'commands.jsonl').open('x',encoding='utf8');self.command_count=0
        self.raw=(self.folder/'teleop_events.jsonl').open('x',encoding='utf8')
        self.count=0;self.rejects=0;self.last_ns=None;self.last_camera_seq={};self.first_sample_ns=None;self.max_gap_ns=0
        self.started=metadata.get('started_monotonic_ns',time.monotonic_ns());self.closed=False
        self.metadata.update({'schema':'piper_native_capture_v2','episode_id':self.id,'started_wall_ns':time.time_ns(),
          'started_monotonic_ns':self.started,'clock':'host_monotonic_ns_for_alignment',
          'action_semantics':'absolute_joint_targets_rad_and_gripper_width_mm_sent_on_follower_CAN',
          'sample_command_semantics':'echo_confirmed_target_active_at_observation; raw commands retain actual send times',
          'state_semantics':'measured_joints_rad_and_gripper_width_mm',
          'simulation_action_compatible':False,'camera_clock_note':'host delivery times are not exposure synchronization',
          'formal_result':False})
        atomic_json(self.folder/'metadata.json',self.metadata)
    def append_raw(self,row):
        self.raw.write(json.dumps(row,separators=(',',':'))+'\n')
    def append_commands(self,rows,dropped=0):
        if self.closed:raise RuntimeError('Episode closed')
        if dropped:self.reject(['command_queue_overflow_'+str(dropped)],time.monotonic_ns())
        for row in rows:
            if row['origin']!='local_can_tx' or row['id'] not in [0x151,0x155,0x156,0x157,0x159] or len(bytes.fromhex(row['data_hex']))!=8:raise ValueError('Invalid command frame')
            self.commands.write(json.dumps(row,allow_nan=False)+'\n');self.command_count+=1
        self.commands.flush()
    def reject(self,reasons,now_ns):
        self.rejects+=1
        self.events.write(json.dumps({'event':'sample_rejected','time_monotonic_ns':now_ns,'reasons':reasons})+'\n')
        self.events.flush()
    def append(self,state,command,cameras,now_ns,leader_intent=None):
        if self.closed:raise RuntimeError('Episode closed')
        reasons=valid_sample(state,command,cameras,now_ns)
        if self.last_ns is not None and now_ns<=self.last_ns:reasons.append('non_increasing_timestamp')
        for role,c in cameras.items():
            if c and self.last_camera_seq.get(role)==c.get('sequence'):reasons.append('duplicate_camera_'+role)
        if reasons:self.reject(reasons,now_ns);return False
        imeta={}
        for role in ['external','wrist']:
            c=cameras[role]
            dst=self.folder/'images'/role/('%06d.jpg'%self.count);dst.parent.mkdir(parents=True,exist_ok=True)
            ok=cv2.imwrite(str(dst),c['image'],[cv2.IMWRITE_JPEG_QUALITY,95])
            if not ok:raise OSError('Image write failed')
            imeta[role]={k:v for k,v in c.items() if k!='image'}
            imeta[role].update({'path':str(dst.relative_to(self.folder)),'sha256':sha(dst)})
        record={'index':self.count,'time_monotonic_ns':now_ns,'time_wall_ns':time.time_ns(),'state':state,
          'command':command,'leader_intent':leader_intent,'images':imeta}
        self.records.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');self.records.flush()
        if self.first_sample_ns is None:self.first_sample_ns=now_ns
        if self.last_ns is not None:self.max_gap_ns=max(self.max_gap_ns,now_ns-self.last_ns)
        self.count+=1;self.last_ns=now_ns
        self.last_camera_seq={k:v['sequence'] for k,v in cameras.items()}
        return True
    def finish(self,outcome,end_ns=None):
        if self.closed:raise RuntimeError('Already closed')
        if outcome not in ['success','failure','aborted']:raise ValueError('Explicit outcome required')
        for f in [self.records,self.events,self.commands,self.raw]:f.flush();os.fsync(f.fileno());f.close()
        self.closed=True
        end_ns=time.monotonic_ns() if end_ns is None else end_ns
        coverage_gap=max(self.max_gap_ns, self.first_sample_ns-self.started, end_ns-self.last_ns) if self.count else end_ns-self.started
        passed=self.count>1 and self.rejects==0 and coverage_gap<=120_000_000 and outcome!='aborted'
        summary={'outcome':outcome,'outcome_source':'operator_annotation','samples':self.count,'command_frames':self.command_count,'rejected_samples':self.rejects,
          'mean_recording_hz':(self.count-1)*1e9/(self.last_ns-self.first_sample_ns) if self.count>1 else 0,
          'max_record_gap_ms':self.max_gap_ns/1e6,
          'max_coverage_gap_ms':coverage_gap/1e6,
          'finished_monotonic_ns':end_ns,'sampling_checks_pass':passed,
          'purpose':self.metadata['purpose'],'formal_result':False,
          'training_candidate':self.metadata['purpose'] in ['demonstration','autonomous_training'] and outcome=='success' and passed and self.command_count>0}
        atomic_json(self.folder/'summary.json',summary)
        manifest={str(p.relative_to(self.folder)):sha(p) for p in self.folder.rglob('*') if p.is_file()}
        atomic_json(self.folder/'sha256.json',manifest)
        target=self.folder.with_suffix('.complete')
        self.folder.rename(target);self.folder=target
        return target
def audit_episode(folder):
    folder=Path(folder);meta=json.loads((folder/'metadata.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    hashes=json.loads((folder/'sha256.json').read_text())
    files={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name!='sha256.json'}
    if set(hashes)!=files:raise ValueError('Manifest file set mismatch')
    if not all((folder/p).resolve().is_relative_to(folder.resolve()) and (folder/p).is_file() and sha(folder/p)==d for p,d in hashes.items()):raise ValueError('Hash mismatch')
    rows=[json.loads(x) for x in (folder/'samples.jsonl').read_text().splitlines()]
    if len(rows)!=summary['samples']:raise ValueError('Frame count mismatch')
    commands=[json.loads(x) for x in (folder/'commands.jsonl').read_text().splitlines()]
    if len(commands)!=summary['command_frames']:raise ValueError('Command frame count mismatch')
    if any(b['host_monotonic_ns']<a['host_monotonic_ns'] for a,b in zip(commands,commands[1:])):raise ValueError('Command clock ordering')
    prev=-1;last_seq={}
    for i,row in enumerate(rows):
        if row['index']!=i or row['time_monotonic_ns']<=prev:raise ValueError('Invalid time/index')
        prev=row['time_monotonic_ns']
        if set(row['images'])!={'external','wrist'}:raise ValueError('Camera roles mismatch')
        if row['command']['sample_monotonic_ns']>prev:
            raise ValueError('Control target is later than its observation')
        for role,x in row['images'].items():
            if x['sequence']<=last_seq.get(role,0) or x['host_monotonic_ns']>prev:
                raise ValueError('Invalid camera frame sequence or time')
            last_seq[role]=x['sequence']
            p=folder/x['path']
            if not p.resolve().is_relative_to(folder.resolve()):raise ValueError('Image path escape')
            if sha(p)!=x['sha256'] or cv2.imread(str(p)) is None:raise ValueError('Image invalid')
    return {'integrity_pass':True,'samples':len(rows),'purpose':meta['purpose'],'outcome':summary['outcome'],
      'formal_result':False,'simulation_action_compatible':False}


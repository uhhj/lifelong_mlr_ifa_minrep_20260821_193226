"""Reviewed script transport; password via getpass only, or existing SSH agent/key."""
import argparse, datetime, getpass, hashlib, json
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent

def run(name,agent=False):
    assert Path(name).name==name and name.endswith('.py')
    out=R/(Path(name).stem+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
    c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
    pw=None;authenticated=False;commands=0
    try:
        if not agent:pw=getpass.getpass('SSH password (memory only): ')
        c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=agent,allow_agent=agent,timeout=20,banner_timeout=20,auth_timeout=20)
        pw=None;authenticated=True
        script=(R/name).read_text(encoding='utf8')
        if name=='deploy_remote.py':script='BUNDLE='+ascii(json.loads((R/'DEPLOY_TEXT_BUNDLE.json').read_text(encoding='utf8')))+'\n'+script
        script='# -*- coding: utf-8 -*-\n'+script
        i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -B -',timeout=1800);commands=1
        i.write(script.encode('utf8'));i.flush();i.channel.shutdown_write()
        raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status()
        (out/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err,script_sha256=hashlib.sha256(script.encode()).hexdigest(),authenticated=True,remote_commands=1),indent=2)+'\n',encoding='utf8')
        assert rc==0,err
        data=json.loads(raw)
        if isinstance(data.get('diagnostics'),dict):
            files={}
            for key,rows in data.pop('diagnostics').items():
                assert key.replace('_','').isalnum();p=out/'diagnostics'/(key+'.json');p.parent.mkdir(exist_ok=True)
                encoded=(json.dumps(rows,indent=2)+'\n').encode();p.write_bytes(encoded)
                files[str(p.relative_to(out))]=dict(sha256=hashlib.sha256(encoded).hexdigest(),rows=len(rows),bytes=len(encoded))
            data['diagnostic_files']=files
        (out/'EVIDENCE.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
        print(out);print(json.dumps({k:v for k,v in data.items() if k not in ('light','log_tails','diagnostics','unclassified_intermediate_name_inventory')})[:6000])
    except Exception as ex:
        if not (out/'ATTEMPT.json').exists():
            (out/'ATTEMPT.json').write_text(json.dumps(dict(status='FAILED_NO_COMPLETION_CLAIM',error_type=type(ex).__name__,error=str(ex),authenticated=authenticated,remote_commands=commands,observation=datetime.datetime.now().astimezone().isoformat()),indent=2)+'\n',encoding='utf8')
        raise
    finally:pw=None;c.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('script');p.add_argument('--agent',action='store_true');a=p.parse_args();run(a.script,a.agent)

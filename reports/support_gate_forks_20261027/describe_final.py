from pathlib import Path
import json,sys
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);assert json.loads((D/'LOCAL_FINAL_AUDIT.json').read_text())['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
v=json.loads((D/'EVIDENCE.json').read_text());cases=json.loads((R/'CASES.json').read_text());summary={}
for name,c in cases.items():
    records=json.loads(v['light'][name+'/OUTCOMES.json']['text']);methods={};pairs={}
    by={arm:{(r['task'],r['initial_index']):r for r in records if r['arm']==arm} for arm in c['arms']}
    for arm in c['arms']:
        scores={str(t):sum(by[arm][t,i]['success'] for i in range(20))/20 for t in range(6,c['stage']+1)}
        methods[arm]=dict(tasks=scores,new=scores[str(c['stage'])],old_mean=sum(scores[str(t)] for t in range(6,c['stage']))/(c['stage']-6))
    for old,new in [('C','S'),('S','Q'),('Q','G'),('C','G')]+([('U','Q')] if 'U' in c['arms'] else []):
        per={}
        for t in range(6,c['stage']+1):
            sf=[i for i in range(20) if by[old][t,i]['success'] and not by[new][t,i]['success']];fs=[i for i in range(20) if not by[old][t,i]['success'] and by[new][t,i]['success']]
            per[str(t)]=dict(SF=sf,FS=fs,delta=(len(fs)-len(sf))/20)
        a,b=methods[old],methods[new];support=(b['new']>=a['new'] and b['old_mean']>a['old_mean']) or (b['new']>a['new'] and b['old_mean']>=a['old_mean'])
        pairs[new+'-'+old]=dict(tasks=per,support_gate=support)
    summary[name]=dict(methods=methods,pairs=pairs)
(R/'COMPARISON.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

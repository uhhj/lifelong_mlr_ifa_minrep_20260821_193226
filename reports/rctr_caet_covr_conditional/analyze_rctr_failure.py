"""Post-hoc read-only analysis of existing committed RCTR outcomes. No model/env calls."""
import collections
import hashlib
import json
from pathlib import Path

R=Path(__file__).resolve().parent
OUT=R/'rctr_failure_analysis';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def outcome(a,b):return ('S' if a else 'F')+('S' if b else 'F')
def pair_stats(pairs):
    c=collections.Counter(outcome(a['success'],b['success']) for a,b in pairs)
    return {k:c[k] for k in ('FF','FS','SF','SS')}
def qualifies(p):
    c=pair_stats(p);return c['FS']>=1 and c['SF']==0

def main():
    source=R/'progress_20261021_094302/DETAILS.json';d=read(source)
    contract=read(R/'rctr42_admission/CONTRACT.json')
    checked={}
    for name in ('rctr_exp/collect.py','rctr_exp/loss.py'):
        assert sha(R/'source'/name)==contract['source_sha256'][name]
        checked[name]=sha(R/'source'/name)
    results={};all_light=[]
    for task,data in d['collections_by_task'].items():
        rows=data['committed'];assert all(r['status']=='COMMITTED' for r in rows)
        assert len(rows)==len({r['key'] for r in rows})
        snapshots=[];confirm=[];all_candidate_pairs=[];perturb_pairs=[];rolepairs=collections.defaultdict(list)
        for initial,at in sorted({(r['initial'],r['snapshot_step']) for r in rows}):
            table={(r['phase'],r['candidate']['id'],r['repeat']):r for r in rows if (r['initial'],r['snapshot_step'])==(initial,at)}
            normal=[table['exploration','normal',rep] for rep in range(2)]
            candidates=[];bases=[];children=[];allpairs=[]
            for axis in range(3):
                for sign in (-1,1):
                    base=[table['exploration',f'a{axis}_{sign:+d}_original',rep] for rep in range(2)]
                    bases+=base;perturb_pairs+=list(zip(normal,base))
                    for role in ('1','2'):
                        cid=f'a{axis}_{sign:+d}_{role}'
                        child=[table['exploration',cid,rep] for rep in range(2)]
                        pairs=list(zip(base,child));allpairs+=pairs;children+=child
                        assert all(a['noise_sha256']==b['noise_sha256'] for a,b in pairs)
                        rolepairs[role]+=pairs
                        if qualifies(pairs):candidates.append((cid,axis,sign,pair_stats(pairs)))
            all_candidate_pairs+=allpairs
            snapshots.append(dict(initial=initial,step=at,normal_success=sum(r['success'] for r in normal),
                perturbed_default_success=sum(r['success'] for r in bases),perturbed_default_total=len(bases),
                alternative_success=sum(r['success'] for r in children),alternative_total=len(children),
                candidate_pairs=pair_stats(allpairs),eligible_candidates=len(candidates),
                all_default_succeed=all(r['success'] for r in bases),
                all_branches_fail=not any(r['success'] for r in normal+bases+children)))
            if candidates:
                cid,axis,sign,exp=candidates[0]
                cb=[table['confirmation',f'a{axis}_{sign:+d}_original',rep] for rep in range(2)]
                cc=[table['confirmation',cid,rep] for rep in range(2)]
                pairs=list(zip(cb,cc));assert all(a['noise_sha256']==b['noise_sha256'] for a,b in pairs)
                stats=pair_stats(pairs)
                reason=('reverse_harm' if stats['SF'] else 'both_succeed_no_rescue' if stats['SS']==2 else
                        'both_fail_no_rescue' if stats['FF']==2 else 'no_net_qualifying_rescue')
                confirm.append(dict(initial=initial,step=at,candidate=cid,exploration=exp,
                    exploration_eligible_candidates=len(candidates),confirmation=stats,
                    passes=qualifies(pairs),reason=reason))
        groups={}
        for name,predicate in [('normal',lambda r:r['phase']=='exploration' and r['candidate']['id']=='normal'),
                ('perturbed_default',lambda r:r['phase']=='exploration' and r['candidate']['axis'] is not None and r['candidate']['role']=='original'),
                ('alternative',lambda r:r['phase']=='exploration' and r['candidate']['role']!='original'),
                ('confirmation',lambda r:r['phase']=='confirmation')]:
            rr=[r for r in rows if predicate(r)]
            groups[name]=dict(n=len(rr),successes=sum(r['success'] for r in rr),
                policy_steps=sum(r['steps'] for r in rr),seconds=sum(r['seconds'] for r in rr))
        assert sum(x['n'] for x in groups.values())==len(rows)
        assert sum(x['passes'] for x in confirm)==data['acceptance']['records']==0
        results[task]=dict(groups=groups,snapshots=snapshots,confirmations=confirm,
            candidate_vs_default=pair_stats(all_candidate_pairs),candidate_by_role={k:pair_stats(v) for k,v in rolepairs.items()},
            perturb_vs_normal=pair_stats(perturb_pairs),
            eligible_candidate_count=sum(s['eligible_candidates'] for s in snapshots),
            snapshots_with_default_failure=sum(not s['all_default_succeed'] for s in snapshots),
            all_failure_snapshots=sum(s['all_branches_fail'] for s in snapshots),
            total_seconds=sum(r['seconds'] for r in rows),total_steps=sum(r['steps'] for r in rows))
        all_light.extend(dict(task=int(task),**{k:r[k] for k in ('key','initial','snapshot_step','phase','repeat','candidate','noise_seed','noise_sha256','success','steps','seconds')}) for r in rows)
    result=dict(status='READ_ONLY_RECOMPUTATION_PASS',source_sha256=sha(source),verified_source_files=checked,
        tasks=results,total_branches=len(all_light),total_steps=sum(r['steps'] for r in all_light),
        total_branch_seconds=sum(r['seconds'] for r in all_light),
        optimizer_calls=0,environment_steps=0,
        caveat='Repeated outcomes within snapshot/noise are dependent; counts are descriptive, not independent trials or a significance test.')
    (OUT/'ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'BRANCH_OUTCOMES.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_light))
    print(json.dumps({k:{kk:v for kk,v in val.items() if kk!='snapshots'} for k,val in results.items()},indent=2))

if __name__=='__main__':main()

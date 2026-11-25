"""Verify window selection and pair scores, and write a human-readable index of selected frames."""
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_selection(selection):
    chosen=[]
    lines=['# B1 横向对准训练窗口','',
           '帧号是原始示范 alignment.jsonl 的零起始行号，区间含首尾。起始秒数从本条示范第一个保留的反馈时刻算起。距离均为桌面 XY 毫米。','',
           '| 示范编号 | 起止帧 | 起始秒 | 起始距离 | 八步后距离 | 净缩短 | 到最终夹持位置的净缩短 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    skipped=[]
    nraw=0
    nscan=0
    for ep,group in selection['episodes'].items():
        rows=group['qualifying']
        nraw+=len(rows)
        nscan+=group['scanned']
        assert group['scanned']==max(0,group['stop_exclusive']-15)
        assert [r['t'] for r in rows]==sorted({r['t'] for r in rows})
        by_t={r['t']:r for r in rows}
        selected=[r for r in rows if r['selected']]
        for a,b in zip(selected,selected[1:]):assert a['t']+8<=b['t']
        for r in rows:
            assert 7<=r['t'] and r['t']+8<group['stop_exclusive']
            assert r['source_start']==group['first_source_row']+r['t'] and r['source_end']==r['source_start']+8
            assert abs(r['initial_mm']-r['final_mm']-r['advance_mm'])<1e-8
            assert r['initial_mm']<=60 and min(r['advance_mm'],r['contact_advance_mm'])>=5
            assert r['advance_mm']>=0.6*r['path_mm'] and r['planar_chord_mm']>=r['height_span_mm']
        # Independent backward dynamic program by source frame, not by candidate index.
        stop=group['stop_exclusive']
        dp=[(0,0.) for _ in range(stop+9)]
        for t in range(stop-1,6,-1):
            dp[t]=dp[t+1]
            if t in by_t:
                r=by_t[t]
                count,gain=dp[t+8]
                dp[t]=max(dp[t],(count+1,gain+min(r['advance_mm'],r['contact_advance_mm'])))
        assert len(selected)==dp[7][0]
        np.testing.assert_allclose(sum(min(r['advance_mm'],r['contact_advance_mm']) for r in selected),dp[7][1],atol=1e-8)
        if not selected:skipped.append(ep)
        for r in selected:
            lines.append(f"| {ep} | {r['source_start']}–{r['source_end']} | {r['start_sec']:.2f} | {r['initial_mm']:.1f} | {r['final_mm']:.1f} | {r['advance_mm']:.1f} | {r['contact_advance_mm']:.1f} |")
        chosen.extend(selected)
    assert nscan==selection['n_scanned'] and nraw==selection['n_qualifying'] and len(chosen)==selection['n_selected']
    assert selection['represented']==len(selection['episodes'])-len(skipped)
    lines+=['','以下示范没有符合筛选规则的窗口。','',', '.join(skipped),'']
    return chosen,'\n'.join(lines)


def paired(rows,pairs,mode):
    entries=[r for r in rows if r['mode']==mode]
    assert len(entries)==2*len(pairs)
    scores=[]
    for a,b in pairs:
        group=[r for r in entries if (r['recipient'],r['donor']) in ((a,b),(b,a))]
        assert len(group)==2 and {r['recipient'] for r in group}=={a,b}
        scores.append(np.mean([r['signed_mm'] for r in group]))
    return np.array(scores)


def metrics(s,reference,boot):
    diff=s-reference
    return dict(n=len(s),positive=int(sum(s>1e-8)),positive_over_1=int(sum(s>1)),negative_over_1=int(sum(s< -1)),
        median_signed_mm=np.median(s),mean_signed_mm=np.mean(s),min_mm=min(s),max_mm=max(s),
        pairwise_median_difference_mm=np.median(diff),
        paired_ci95_mm=np.percentile(np.median(diff[boot],axis=1),[2.5,97.5]).tolist())


def main(out):
    selected=read(out/'selection.json')
    fit=read(out/'fit.json')
    inputs=read(out/'audit_inputs.json')
    results=read(out/'outcomes.json')
    last=out.parent/'piper_b1_feedback_xy_20261125'
    older=read(last/'summary.json')
    original=read(out.parent/'piper_b1_vision_contrasts_20261124/final.json')
    pairs=inputs['pairs']
    assert pairs==original['geometry']['pairs']==older['pairs']
    assert len(pairs)==16 and len({e for p in pairs for e in p})==32
    assert set(selected['episodes'])==set(read(last/'audit_inputs.json')['training_episodes'])
    assert set(selected['episodes']).isdisjoint(inputs['heldout_episodes'])
    assert inputs['original_windows_sha256']==read(last/'audit_inputs.json')['windows_sha256']
    assert selected['checkpoint_sha256']==inputs['checkpoint_sha256']==original['checkpoint_ids']['baseline']
    assert sha(out/'protocol.md')==selected['protocol_sha256']==inputs['protocol_sha256']
    assert sha(out/'selection.json')==inputs['selection_sha256']
    chosen,index=inspect_selection(selected)
    assert inputs['independent_training_window_rescan']==selected['n_scanned']
    assert inputs['rescan_max_advance_difference_mm']<1e-4
    assert fit['n_action_training']==len(chosen) and fit['n_demos']==selected['represented']
    assert fit['matched_reference_readouts'] and fit['readout_training']==[50,33]
    verification=fit['image_to_joint_verification']
    assert verification['natural']==33 and verification['swapped']==32 and verification['max_scalar_difference_mm']<1e-5
    ik=results['pose_reconstruction']
    assert ik['poses']==2072 and max(ik['xy_max_mm'],ik['height_max_mm'],ik['angle_max_deg'])<1e-5
    scores={}
    statistics={}
    bootstrap=np.random.default_rng(1125).integers(len(pairs),size=(20000,len(pairs)))
    for r in results['directions']:
        np.testing.assert_allclose(np.hypot(r['signed_mm'],r['orthogonal_mm']),r['amplitude_mm'],atol=1e-7)
    for mode in ('base','xy','fixed','permuted','wood_only','jaw_only'):
        s=paired(results['directions'],pairs,mode)
        scores[mode]=s.tolist()
        statistics[mode]=metrics(s,scores['base'],bootstrap)
    np.testing.assert_allclose(scores['base'],np.array(older['pair_means']['base'])[:,0],rtol=0,atol=1e-7)
    fixed_window=np.array(older['pair_means']['xy'])[:,0]
    vs_fixed=metrics(np.array(scores['xy']),fixed_window,bootstrap)
    natural={}
    for mode in ('base','xy','fixed','permuted','measured'):
        rs=[r for r in results['natural'] if r['mode']==mode]
        assert len(rs)==33 and {r['episode'] for r in rs}==set(inputs['heldout_episodes'])
        natural[mode]={k:np.median([r[k] for r in rs]) for k in ('feedback_endpoint_error_mm','advance_mm','travel_mm')}
    context=inputs['fixed_anchor_context']
    assert {r['episode'] for r in context}==set(inputs['heldout_episodes'])
    match=inputs['phase_matched']
    assert match['policy_sha256']==fit['policy_sha256']
    eligible=[p for p in pairs if all(match['ticks'][ep] is not None for ep in p)]
    assert match['pairs']==eligible
    supplement={}
    if eligible:
        boot=np.random.default_rng(1125).integers(len(eligible),size=(20000,len(eligible)))
        ref=paired(match['directions'],eligible,'base')
        for mode in ('base','xy'):
            ss=paired(match['directions'],eligible,mode)
            supplement[mode]=metrics(ss,ref,boot)
            supplement[mode]['natural_error_mm']=np.median([r['error_mm'] for r in match['natural'] if r['mode']==mode])
    res=dict(pair_signed_mm=scores,primary=statistics,vs_previous_xy=vs_fixed,natural=natural,
        median_selected_advance_mm=np.median([r['advance_mm'] for r in chosen]),
        anchor_matches=sum(r['qualifies'] for r in context if r['paired']),
        phase_matched_pairs=len(eligible),phase_matched=supplement)
    (out/'summary.json').write_bytes((json.dumps(res,indent=1,allow_nan=False)+'\n').encode())
    (out/'windows.md').write_bytes(index.encode())


if __name__=='__main__':
    main(Path(__file__).resolve().parent)

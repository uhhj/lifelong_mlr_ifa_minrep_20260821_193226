"""Independent metadata-only reconstruction of completed COVR decisions."""
def audit_decisions(stage, collection):
    decisions = collection['decisions']
    branches = {r['key']: r for r in collection['branches']}
    normals = {(n['task'], n['initial']): n for n in collection['normal']}
    reports = {}
    expected_keys = set()
    certified = windows = 0
    for task in range(6, stage):
        selected = 0
        for initial in range(20, 26):
            key = f't{task}_i{initial}'
            if key not in decisions:
                continue
            d = decisions[key]
            n = normals[task, initial]
            assert n == d['normal']
            should_select = not n['success'] and n['snapshot_available'] and selected < 3
            assert d['selected'] == should_select
            if not should_select:
                assert d['reason'] == 'NOT_ELIGIBLE'
                continue
            selected += 1
            def rows(phase, mode):
                keys = [f's{stage}_t{task}_i{initial}_{phase}_r{r}_{mode}' for r in range(3)]
                expected_keys.update(keys)
                return [branches[k] for k in keys]
            def changes(before, after):
                return [sum(not a['success'] and b['success'] for a,b in zip(before,after)),
                        sum(a['success'] and not b['success'] for a,b in zip(before,after))]
            def passes(before, after):
                fs,sf = changes(before,after)
                return fs >= 2 and sf == 0
            def distinct(child, normal, argmax):
                return all(c['first_action_sha256'] not in (n['first_action_sha256'],a['first_action_sha256'])
                           for c,n,a in zip(child,normal,argmax))
            normal = rows('exploration','normal')
            winner = normal[0]['winner']
            means = {m:rows('exploration',m) for m in range(5)}
            meta = d['certificate']
            assert meta['winner'] == winner
            eligible = []
            comparisons = {}
            for m, child in means.items():
                comparisons[str(m)] = dict(vs_normal=changes(normal,child),vs_argmax=changes(means[winner],child))
                if m != winner and passes(normal,child) and passes(means[winner],child) and distinct(child,normal,means[winner]):
                    eligible.append((-(changes(normal,child)[0]+changes(means[winner],child)[0]),m))
            if not eligible:
                assert meta['status'] == 'NO_EXPLORATION_CANDIDATE'
            else:
                chosen = min(eligible)[1]
                assert meta['chosen'] == chosen
                nc,mc,cc = rows('confirmation','normal'),rows('confirmation',winner),rows('confirmation',chosen)
                for field, before, after in [('exploration_vs_normal',normal,means[chosen]),
                    ('exploration_vs_argmax',means[winner],means[chosen]),
                    ('confirmation_vs_normal',nc,cc),('confirmation_vs_argmax',mc,cc)]:
                    assert meta[field] == changes(before,after)
                status = ('CONFIRMATION_FAILED_NO_REPLACEMENT' if not (passes(nc,cc) and passes(mc,cc)) else
                          'CONFIRMATION_ACTION_NOT_DISTINCT' if not distinct(cc,nc,mc) else 'CERTIFIED')
                assert meta['status'] == status
                if status == 'CERTIFIED':
                    repeat = next(r for r in range(3) if cc[r]['success'] and not nc[r]['success'] and not mc[r]['success'])
                    assert meta['selected_confirmation_repeat'] == repeat and meta['training_branch_key'] == cc[repeat]['key']
                    assert 1 <= meta['windows'] <= min(8,cc[repeat]['steps'])
                    certified += 1
                    windows += meta['windows']
            reports[key] = dict(status=meta['status'],winner=winner,comparisons=comparisons,
                                eligible_in_priority_order=sorted(eligible),certificate=meta)
    acceptance = collection['acceptance']
    if acceptance is not None:
        assert set(decisions) == {f't{t}_i{i}' for t in range(6,stage) for i in range(20,26)}
        assert expected_keys == set(branches)
        assert acceptance['status'] == 'PASS' and acceptance['optimizer_calls'] == 0
        assert acceptance['unique_branches'] == len(branches)
        assert acceptance['selected_snapshots'] == len(reports)
        assert acceptance['certified_snapshots'] == certified and acceptance['training_windows'] == windows
        assert acceptance['normal_episodes'] == len(normals)
        assert acceptance['invocation_counts']['screening_steps'] == sum(b['steps'] for b in branches.values())
    return dict(status='PASS',completed_decisions=len(decisions),selected=len(reports),
                certified=certified,training_windows=windows,decisions=reports,
                scope='Outcome/selection/confirmation metadata; full payload structure remains server final audit responsibility')

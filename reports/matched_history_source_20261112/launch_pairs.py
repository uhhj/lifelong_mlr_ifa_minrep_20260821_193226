"""Build exact case packages and submit the finite queue using existing SSH transport."""
import ast
import json
import sys

from launch_collection import COPY, HERE, send


def source(name):
    return (HERE / name).read_text(encoding='utf-8')


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def evaluation(text):
    for old, new in [('Exactly 200 episodes per seed', 'Exactly 160 paired Task9-final episodes per seed'),
                     ("p=load('S',6,c,shape)", "p=load('E',9,c,shape)"),
                     ('budget=200', 'budget=160'), ('for stage in range(6,10):', 'for stage in (9,):'),
                     ('len(records)==200', 'len(records)==160'),
                     ('episodes=200,vectors=20,cells=10', 'episodes=160,vectors=16,cells=8')]:
        text = replace_once(text, old, new)
    return replace_once(text, 'initial_sha=sha(initial)',
        "initial_sha=sha(initial);split=js(R/'INITIAL_SPLIT.json')[str(taskid)];assert initial_sha==split['file_sha256'];assert all(ah(np.asarray(initials[i]))==split['evaluation'][str(i)] for i in range(20))")


def case_layout(original):
    path = original / 'layout.py'
    names = ('POP', 'VALID', 'OLD')
    values = {}
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                values[target.id] = ast.literal_eval(node.value)
    assert set(values) == set(names) and all(set(v) == set(range(6, 10)) for v in values.values()), path
    return source('pair_layout.py') + '\n' + ''.join('%s = %r\n' % (name, values[name]) for name in names)


def bundle():
    result = {'controller.py': source('controller.py'), 'PROTOCOL.md': source('PROTOCOL.md'),
              'INPUT_PLAN.json': source('INPUT_PLAN.json')}
    cases = json.loads(result['INPUT_PLAN.json'])['cases']
    versions = {'GOAL': '20261029_support_gate_full_s_goal', 'OBJECT': '20261028_support_gate_full_s_object'}
    for case, binding in cases.items():
        suite, seed = binding['suite'], binding['seed']
        original = HERE.parent / versions[suite] / ('seed%d' % seed)
        common = {name: (original / name).read_text(encoding='utf-8') for name in COPY
                  if name != 'candidate.py' or suite == 'GOAL'}
        common['CASE.json'] = json.dumps(binding, indent=2) + '\n'
        layout = case_layout(original)
        if case != 'GOAL_41':
            collection_layout = layout + '\n' + '''
ARMS = 'S'
OUTPUT = {'S': DATA_PRIVATE / CASE / 'acquisition'}
def memory_path(arm, task):
    assert arm == 'S'
    return acquisition_path(task)
'''
            collect = dict(common, **{'layout.py': collection_layout, 'acquire.py': source('acquire.py')})
            result.update({case + '/' + name: text for name, text in collect.items()})
        training = dict(common)
        for name in ('fork_tools.py', 'support_memory.py', 'legacy_memory.py', 'legacy_updates.py',
                     'gate_admission.py', 'updates.py'):
            training[name] = (original / name).read_text(encoding='utf-8')
        training['evaluate.py'] = evaluation(training['evaluate.py'])
        training['layout.py'] = layout
        for name in ('match_histories.py', 'train_pair.py'):
            training[name] = source(name)
        result.update({'pairs/' + case + '/' + name: text for name, text in training.items()})
    return result


if __name__ == '__main__':
    assert sys.argv[1:] in [['launch'], ['status'], ['recover'], ['recovery_status']]
    action = sys.argv[1]
    files = bundle() if action in ('launch', 'recover') else {}
    if action == 'recover':
        old = 'MATCHED_HISTORY_SOURCE_GOAL_OBJECT_20261112_v1'
        for name in files:
            if name.endswith('/layout.py'):
                files[name] = replace_once(files[name], old, old + '_engineering02')
    sys.exit(send('pair_' + action, 'ACTION = ' + repr(action) + '\nBUNDLE = ' + ascii(files) + '\n' + source('pairs_remote.py')))

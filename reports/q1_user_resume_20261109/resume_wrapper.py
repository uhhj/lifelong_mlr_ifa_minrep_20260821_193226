import os,sys,json,io,time,hashlib,traceback
from pathlib import Path
ENG=Path(__file__).resolve().parent
ROOT=ENG.parent.parent/'S_Q1_OBJECT41_42_43_20261109_rerun_v1'
case=sys.argv[1];assert case in ['Q1_41','Q1_42']
sys.argv=[str(ROOT/'train_wrapper.py'),case]
source=(ROOT/'train_wrapper.py').read_text();assert source.endswith('tr.main()\n')
ns={'__file__':str(ROOT/'train_wrapper.py'),'__name__':'resume_wrapper'}
exec(compile(source[:-len('tr.main()\n')],str(ROOT/'train_wrapper.py'),'exec'),ns)
rt=ns['rt'];tr=ns['tr'];torch=ns['torch']
binding=json.loads((ENG/'RECOVERY_BINDING.json').read_text())['cases'][case]

import ast
assert all(rt.sha(p)==h for p,h in rt.js(ENG/'CONTRACT.json')['files'].items())
assert json.loads((ENG/'VALIDATION.json').read_text())['status']=='VALIDATION_COMPLETE'
for n,h in binding['light_hashes'].items():assert rt.sha(rt.R/n)==h
assert rt.sha(binding['checkpoint'])==binding['checkpoint_sha256']
tr.ENG=ENG;tr.case=case;tr.binding=binding
main=ast.parse((ENG/'resume_main.py').read_text()).body[0]
change=ns['Change']();change.visit(main);assert change.copies==3 and change.checks==1
exec(compile(ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[])),str(ENG/'resume_main.py'),'exec'),tr.__dict__)
tr.main()

from pathlib import Path
import json,hashlib,datetime,ast
R=Path(__file__).resolve().parent
old=R/'engineering01_preserved';old.mkdir(exist_ok=False)
for n in ['paired_worker.py','supervisor.py','deploy.py','CONTRACT.json','ACTIVE_ROOT.json']:(old/n).write_bytes((R/n).read_bytes())
t=(R/'paired_worker.py').read_text()
t=t.replace("MUJOCO_GL='egl',OMP_NUM_THREADS", "MUJOCO_GL='egl',LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HOME='/data/mlr_ifa/cache/huggingface',MPLBACKEND='Agg',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS")
ast.parse(t);(R/'paired_worker.py').write_text(t,encoding='utf8')
t=(R/'deploy.py').read_text().replace("Q='/data/mlr_ifa/experiment_workspaces/RA_ALIGN_PAIRED_GOAL_20261024_v1'", "Q='/data/mlr_ifa/experiment_workspaces/RA_ALIGN_PAIRED_GOAL_20261024_v1_engineering02'")
# Bind stopped parent and its full frozen source before allowing new version.
t=t.replace("q=Path(%r)\nassert not q.exists()", "q=Path(%r)\nold=Path('/data/mlr_ifa/experiment_workspaces/RA_ALIGN_PAIRED_GOAL_20261024_v1')\nassert (old/'worker.exit').read_text().strip()=='1' and (old/'supervisor.exit').read_text().strip()=='1'\nassert not (old/'RUN_STARTED.json').exists() and not (old/'ADMISSION.json').exists()\nimport hashlib\nct=json.loads((old/'CONTRACT.json').read_text());assert all(hashlib.sha256((old/n).read_bytes()).hexdigest()==h for n,h in ct['files'].items())\nprocs=subprocess.check_output(['ps','-eo','args'],text=True);assert not any(str(old)+'/paired_worker.py' in l or str(old)+'/supervisor.py' in l for l in procs.splitlines())\nassert not q.exists()")
ast.parse(t);(R/'deploy.py').write_text(t,encoding='utf8')
(R/'ENGINEERING02.md').write_text('工程01在导入libero时未设置LIBERO_CONFIG_PATH，input遇EOF。worker/supervisor exit1，0模型前向/0优化器/0环境，未创建私有轨迹目录。原源码、合同、日志、失败快照保留。新工程02只补齐已有运行环境LIBERO_CONFIG_PATH/HF_HOME/MPLBACKEND/OPENBLAS_NUM_THREADS，不改模型、输入、噪声、预算或指标。启动前核旧退出/hash/无进程、GPU空闲；新版本必须自己的准入。导入可能创建了空~/.libero目录，没有初始化配置或修改已有配置。',encoding='utf8')

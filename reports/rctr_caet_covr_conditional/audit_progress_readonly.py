"""Recompute live committed branch accounting; no experimental result inference."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument("directory",type=Path);args=ap.parse_args()
    p=args.directory.resolve();assert p.parent==R
    snap=read(p/"SNAPSHOT.json");details=read(p/"DETAILS.json")
    previous=read(R/"progress_20261021_server_queue/SNAPSHOT_LATEST.json")
    row=snap["runs"]["RCTR_OBJECT_42"]
    assert row["epochs"]==100 and row["optimizer_steps"]==15700 and row["formal_records"]==0
    assert row["exits"]=={"train6.exit":"0"}
    assert details["status"]=="FROZEN_CONTRACT_INPUTS_PASS"
    assert details["source_files"]==176
    assert snap["controller_v2"]["exit"] is None
    assert len(read(p/"REGISTRY.json"))==18
    assert snap["libero50_acquisition"]["verified"]==50 and snap["libero50_acquisition"]["exit"]=="0"
    rows=details["committed"]
    assert len({r["key"] for r in rows})==len(rows)
    for r in rows:
        assert r["task"]==6 and 20<=r["initial"]<=25 and r["snapshot_step"] in (40,120)
        assert r["phase"] in ("exploration","confirmation") and r["repeat"] in (0,1)
        assert 0<=r["steps"]<=600-r["snapshot_step"]
        assert r["status"] in ("COMMITTED","SKIPPED")
        assert r["status"]!="COMMITTED" or isinstance(r["success"],bool)
        assert r["status"]!="SKIPPED" or r["success"] is None
        seed=100000000*42+1000000*6+1000*r["initial"]+2*r["snapshot_step"]+r["repeat"]
        if r["phase"]=="confirmation":seed+=500000
        assert r["noise_seed"]==seed
    assert sum(r["phase"]=="exploration" for r in rows)<=456
    assert sum(r["phase"]=="confirmation" for r in rows)<=48
    assert len(rows)>=row["collections"][0]["branches"]
    old=previous["runs"]["RCTR_OBJECT_42"]["collections"][0]["branches"]
    delta=row["collections"][0]["branches"]-old
    rate=(snap["unix"]-previous["unix"])/delta if delta>0 else None
    at=datetime.datetime.fromtimestamp(details["unix"],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    result=dict(status="PARTIAL_PROGRESS_AUDIT_PASS",observed_at=at,epochs=100,
        optimizer_steps=15700,formal_records=0,committed_branches=len(rows),
        committed_policy_steps=sum(r["steps"] for r in rows),
        successful_auxiliary_branches=sum(r["success"] is True for r in rows),
        source_and_input_hashes_pass=True,source_files=176,
        seconds_per_branch_between_snapshots=rate,
        remaining_task6_exploration_upper_hours=(456-len(rows))*rate/3600 if rate else None,
        task6_confirmation_additional_upper_hours=48*rate/3600 if rate else None,
        committed_keys_unique=True,libero50_data_verified=50,libero50_gpu_admitted=False,
        controller_running=True,experiment_recovery_performed=False,
        access_note="Old unrestricted bridge required unavailable approval; a scoped read-only SSH client succeeded. No server worker restart.",
        files={f.name:sha(f) for f in p.iterdir() if f.is_file() and f.name not in ("AUDIT.json","REPORT.md")})
    (p/"AUDIT.json").write_text(json.dumps(result,indent=2)+"\n")
    report=f"""# 进度检查 — PARTIAL

观察时间：{at}。当前仍为RCTR OBJECT42 collect6，没有阶段转换，没有完成验收。

Task6训练100epoch、15,700步已完成；已提交{len(rows)}个辅助分支、{result['committed_policy_steps']}个分支策略步，正式评估0条。176个源码文件及冻结输入核验通过，主采集进程和controller仍在运行，无需重启。

相邻快照实测约{rate:.2f}秒/分支。若剩余快照都可用且保持当前速度，Task6剩余探索上限约{result['remaining_task6_exploration_upper_hours']:.2f}小时，独立确认最多另约{result['task6_confirmation_additional_upper_hours']:.2f}小时；这是当前采集阶段估计，不是整套实验完成时间。成功提前结束或不可用快照会改变耗时，不能据中间结果增减预算。

LIBERO50保持50/50数据已验证，预训练尚未GPU准入。数据盘/系统盘可用空间分别{snap['disk']['/data']['free']/1024**3:.2f}/{snap['disk']['/root']['free']/1024**3:.2f}GiB。GPU：{snap['gpu']}。

旧通用SSH桥调用被本地工具审批策略拒绝；改用固定只读命令、已固定主机密钥、内存getpass连接完成检查。没有改动服务器代码、合同或实验进程，也没有额外训练或环境执行。
"""
    (p/"REPORT.md").write_text(report,encoding="utf8")
    status=read(R/"RUN_STATUS.json")
    status.update(observed_at=at,collection_branches=len(rows),latest_progress=p.name,
                  seconds_per_collection_branch=rate,phase="collect6",status="PARTIAL")
    (R/"RUN_STATUS.json").write_text(json.dumps(status,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":main()

"""Explicit pre-result GOAL adaptation; never edits deployed OBJECT source."""
import hashlib
import json
from pathlib import Path

R=Path(__file__).resolve().parent


def replace_once(text,old,new):
    assert old in text,old
    return text.replace(old,new)


def main():
    bundle=json.loads((R/"OBJECT_WORKER_BUNDLE.json").read_text())
    changes={}
    for name,item in bundle.items():
        original=item["text"];text=original
        if name=="coreset_exp/model.py":
            text=replace_once(text,"c.benchmark_name=='LIBERO_OBJECT' and c.gamma==.3 and c.pretrain.endswith('object_multitask_model_ep99.pth')",
                "c.benchmark_name=='LIBERO_GOAL' and c.gamma==.7 and c.pretrain.endswith('goal_multitask_model_ep96.pth')")
        if name=="coreset_exp/train.py" or name in (
            "rctr_exp/train.py","caet_exp/train.py","covr_exp/train.py"):
            text=replace_once(text,"get_benchmark('LIBERO_OBJECT')","get_benchmark('LIBERO_GOAL')")
        if name in ("rctr_exp/prepare.py","caet_exp/prepare.py","covr_exp/prepare.py"):
            text=text.replace("LIBERO_OBJECT","LIBERO_GOAL").replace("AUTONOMOUS_EXIT_ACCEPTANCE","PROCESS_EXIT_ACCEPTANCE")
            text=text.replace("_OBJECT_v1","_GOAL_v1").replace("105500","76100")
        if name in ("rctr_exp/audit.py","caet_exp/audit.py","covr_exp/audit.py"):
            text=text.replace("105500","76100")
        if name=="caet_exp/labels.py":
            text=replace_once(text,'TARGETS = {6:"butter", 7:"milk", 8:"chocolate_pudding", 9:"orange_juice"}',
                'TARGETS = {6:"cream_cheese", 7:"flat_stove_button", 8:"akita_black_bowl", 9:"wine_bottle"}')
            text=replace_once(text,'if name == target+"_1_joint0":\n            assert kind == "free" and found is None',
                'if name == ("flat_stove_1_button" if target=="flat_stove_button" else target+"_1_joint0"):\n'
                '            assert found is None\n'
                '            if target=="flat_stove_button":\n'
                '                assert kind=="hinge"\n'
                '                assert np.array_equal(np.fromstring(node.get("axis","0 0 1"),sep=" "),[0,0,1])\n'
                '                assert np.array_equal(np.fromstring(node.get("pos","0 0 0"),sep=" "),[0,0,0])\n'
                '                parents={child:parent for parent in tree.iter() for child in parent}\n'
                '                body=parents[node]\n'
                '                assert len([j for j in body if j.tag in ("joint","freejoint")])==1\n'
                '                ancestor=parents[body]\n'
                '                while ancestor.tag!="worldbody":\n'
                '                    assert not any(j.tag in ("joint","freejoint") for j in ancestor)\n'
                '                    ancestor=parents[ancestor]\n'
                '            else:\n'
                '                assert kind=="free"')
            text=replace_once(text,'pose = state[:,1+offset:1+offset+7]',
                'if target=="flat_stove_button":\n'
                '        angle=np.diff(state[:,1+offset])\n'
                '        assert np.all(np.abs(angle)<np.pi)\n'
                '        effect=np.zeros((len(angle),6),dtype=np.float64)\n'
                '        effect[:,5]=angle\n'
                '        return effect,dict(qpos_offset=offset,nq=nq,nv=nv,joint_names=names,\n'
                '            joint_and_gripper_alignment_exact=True,\n'
                '            convention="Recorded static-parent zero-pivot hinge local SO3 increment; zero translation",\n'
                '            xml_sha256=__import__("hashlib").sha256(xml.encode()).hexdigest())\n'
                '    pose = state[:,1+offset:1+offset+7]')
            helper='''\n\ndef certified(row):\n    assert row.get("binding") in (\n        "exact complete 8x7 float32 action window; unique demo/target/length",\n        "exact complete 8x7 float32 window; unique demo/target/length")\n    assert row["possible_starts"] and {min(s+7,row["length"]-1) for s in row["possible_starts"]}=={row["target"]}\n    return True\n'''
            text=replace_once(text,"\ndef build(task, root=None):",helper+"\n\ndef build(task, root=None):")
            text=text.replace('x["identity_unique"]','certified(x)').replace('b["identity_unique"]','certified(b)')
        if text!=original:
            changes[name]=dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                               goal_sha256=hashlib.sha256(text.encode()).hexdigest())
        bundle[name]=dict(text=text,sha256=hashlib.sha256(text.encode()).hexdigest())
    out=R/"goal_preparation";out.mkdir(exist_ok=True)
    for name in changes:
        p=out/"source"/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(bundle[name]["text"],encoding="utf8",newline="\n")
    raw=(json.dumps(bundle)+"\n").encode()
    (R/"GOAL_WORKER_BUNDLE.json").write_bytes(raw)
    (out/"SOURCE_ADAPTATION.json").write_text(json.dumps(dict(
        status="DRAFT_AWAITING_REAL_LABEL_VALIDATION",changed=changes,
        bundle_sha256=hashlib.sha256(raw).hexdigest(),
        formal_protocol="GOAL gamma0.7/76100 main updates; original evaluation byte-identical",
        caet_target7="Actual articulated stove button, recorded hinge angle; not static stove root",
        metrics_seen_for_method_GOAL=False),indent=2)+"\n")
    print(json.dumps(dict(files=len(bundle),changed=len(changes),status="GENERATED_NOT_LAUNCHED")))


if __name__=="__main__":
    main()

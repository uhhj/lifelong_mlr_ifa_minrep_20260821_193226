"""Check the archived on-device UI implementation and SDK source excerpt, without importing either."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import sys


def check(ui, sdk):
    tree = ast.parse(ui.read_text(encoding="utf-8-sig"))
    methods = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    target = methods["update_gripper"]
    emit = methods["gripper_ctrl"]
    assert any(isinstance(n, ast.Call) and
               ast.unparse(n) == "self.gripper_slider.valueChanged.connect(self.update_gripper)" for n in ast.walk(tree))
    assert [ast.unparse(node) for node in target.body][:2] == [
        "self.gripper = self.gripper_slider.value()", "self.gripper_slider_edit.clear()"]
    assert isinstance(target.body[2], ast.Expr) and ast.unparse(target.body[2].value.func) == "self.gripper_slider_edit.append"
    assert len(target.body) == 4 and ast.unparse(target.body[3]) == "self.gripper_ctrl()"
    assert len(emit.body) == 1
    call = emit.body[0].value
    assert ast.unparse(call.func) == "self.piper.GripperCtrl"
    assert ast.unparse(call.args[0]) == "abs(self.gripper * 1000)"
    assert [ast.literal_eval(x) for x in call.args[1:]] == [1000, 1, 0]
    source = sdk.read_text(encoding="utf-8")
    assert "self.GripperCtrl_V3(gripper_angle, gripper_effort, gripper_code, set_zero)" in source
    assert "0x159" in source and "self._arm_can.SendCanMessage(" in source
    return {
        "ui_file_sha256": sha256(ui.read_bytes()).hexdigest(),
        "sdk_inspection_sha256": sha256(sdk.read_bytes()).hexdigest(),
        "installed_sdk_version": source.splitlines()[0],
        "event": "QSlider.valueChanged",
        "update_handler_line": target.lineno,
        "send_handler_line": emit.lineno,
        "source_width_unit_mm": 1,
        "transmitted_width_unit_mm": 0.001,
        "effort_field": 1000,
        "width_enable_code": 1,
        "target_can_id": "0x159",
        "feedback_can_id": "0x2A8",
        "width_changes_permitted_by_handler": "each distinct integer value",
        "actual_manual_event_wire_record": "unavailable",
    }


if __name__ == "__main__":
    print(json.dumps(check(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))

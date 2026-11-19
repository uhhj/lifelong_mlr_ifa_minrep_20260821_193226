"""Offline wire-format checks for the official PiPER V3 encoder and decoder."""
import sys
from itertools import product


def forbid_socket(event, args):
    if event == "socket.__new__":
        raise RuntimeError("This protocol check only uses in-memory messages")


def main():
    sys.addaudithook(forbid_socket)
    from can import Message
    from piper_sdk.piper_msgs.msg_v3 import (
        ArmMsgMasterSlaveModeConfig, ArmMsgMotionCtrl_1, ArmMsgType, PiperMessage,
    )
    from piper_sdk.protocol.protocol_v3 import C_PiperParserV3

    parser = C_PiperParserV3()
    combinations = 0
    for role, feedback, control, linkage in product((0, 0xFA, 0xFC), (0, 0x10, 0x20),
                                                    (0, 0x10, 0x20), (0, 0x10, 0x20)):
        fields = (role, feedback, control, linkage)
        command = PiperMessage(type_=ArmMsgType.PiperMsgMasterSlaveModeConfig,
                               arm_ms_config=ArmMsgMasterSlaveModeConfig(*fields))
        frame = Message()
        assert parser.EncodeMessage(command, frame)
        assert frame.arbitration_id == 0x470
        assert bytes(frame.data) == bytes((*fields, 0, 0, 0, 0))
        combinations += 1
    print(f"Role and offset encodings checked: {combinations}, all on ID 0x470")

    for fields in ((1, 0, 0), (2, 0, 0), (0, 1, 0), (0, 0, 4)):
        command = PiperMessage(type_=ArmMsgType.PiperMsgMotionCtrl_1,
                               arm_motion_ctrl_1=ArmMsgMotionCtrl_1(*fields))
        frame = Message()
        assert parser.EncodeMessage(command, frame)
        assert frame.arbitration_id == 0x150
        assert bytes(frame.data) == bytes((*fields, 0, 0, 0, 0, 0))
    print("Stop, reset, planned-trajectory pause and replay pause occupy distinct fields or values")

    for tail in (bytes(6), bytes(range(1, 7))):
        frame = Message(arbitration_id=0x476, is_extended_id=False, data=bytes([0x70, 0]) + tail)
        decoded = PiperMessage()
        assert parser.DecodeMessage(frame, decoded)
        reply = decoded.arm_feedback_resp_set_instruction
        assert vars(reply) == {"instruction_index": 0x70, "is_set_zero_successfully": 0}
    print("Generic settings reply exposes the instruction index and the zero-setting flag")


if __name__ == "__main__":
    main()

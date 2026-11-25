"""Bounded response to a closing-only tracking fault for the B1 wooden block."""
import math


BUNDLE = "905c4a8df5ace11ff547a8bfbb3a938a98749827097da100b9309a30a6a555fe"
TIMEOUT_NS = 2_000_000_000
DWELL_NS = 150_000_000
MAX_SAMPLE_GAP_NS = 50_000_000
MAX_LEAD_MM = 14.0
WIDTH_BAND = (35.0, 44.0)
CONTACT_NM = (-1.20, -0.80)


def telemetry_at(frames, stamp):
    for row in reversed(frames):
        if row[0] == stamp and len(row) == 4:
            return dict(stamp_ns=stamp, width_mm=row[1], status=row[2], torque_nm=row[3])
        if row[0] <= stamp:
            break
    raise ValueError("缺少与开度同帧的夹爪力矩反馈")


class GripWindow:
    def __init__(self):
        self.phase = "tracking"
        self.deadline = 0
        self.entry_ns = 0
        self.entry_width = 0.0
        self.contact_width = 0.0

    def check_time(self, now):
        if self.deadline and now >= self.deadline:
            raise ValueError("夹爪响应等待超时，已停止；请复核开度、力矩与夹指接触")

    def stable_contact(self, frames, stamp):
        recent = []
        for sample in reversed(frames):
            t, width, code, torque = sample
            if t > stamp:
                continue
            if t < self.entry_ns:
                break
            if (code != 0x40 or not WIDTH_BAND[0] <= width <= WIDTH_BAND[1]
                    or not CONTACT_NM[0] <= torque <= CONTACT_NM[1]):
                return False
            if recent and not 0 < recent[-1][0] - t <= MAX_SAMPLE_GAP_NS:
                return False
            recent.append((t, width))
            if stamp - t >= DWELL_NS:
                widths = [row[1] for row in recent]
                return len(recent) >= 5 and max(widths) - min(widths) <= 0.6
        return False

    def resolve(self, now, stamp, frames, prediction, previous, plan_phase):
        """Return the requested target and whether the positive closing gap is supervised here."""
        self.check_time(now)
        sample = next((row for row in reversed(frames) if row[0] <= stamp), None)
        if sample is None or sample[0] != stamp or len(sample) != 4:
            raise ValueError("缺少与开度同帧的夹爪力矩反馈")
        _, width, code, torque = sample
        if not 0 <= now - stamp <= 60_000_000 or not math.isfinite(torque) or code & 0x7f != 0x40:
            raise ValueError("夹爪反馈过期或驱动状态异常")
        last = previous[6]
        gap = width - last
        if self.phase == "tracking":
            if not (12 < gap <= MAX_LEAD_MM and last <= 55 and prediction[6] < last
                    and plan_phase in ("closing_chunk", "after_close")):
                return prediction, False
            self.phase = "waiting"
            self.entry_ns = now
            self.entry_width = width
            self.deadline = now + TIMEOUT_NS

        if code != 0x40 or abs(torque) > 1.25 or gap > MAX_LEAD_MM:
            raise ValueError("夹爪收爪监督异常：请检查驱动状态、力矩和开度误差")
        if self.phase == "waiting":
            if width > self.entry_width + 1.0 or width < WIDTH_BAND[0] - 1:
                raise ValueError("等待夹持时夹口反向张开或已空合")
            if gap >= 1 and self.stable_contact(frames, stamp):
                self.contact_width = width
                self.phase = "holding"
                self.deadline = 0
            # Freeze the six sent joint references. Further closing follows measured
            # progress, with at most 6 mm lead and the normal slew limit downstream.
            request = min(last, max(prediction[6], width - 6)) if self.phase == "waiting" else last
            return previous[:6] + [request], True

        if self.phase == "holding":
            if abs(width - self.contact_width) > 2 or torque > -0.45:
                raise ValueError("夹持开度或力矩丢失，已停止")
            if prediction[6] >= width + 1:
                self.phase = "releasing"
                self.deadline = now + TIMEOUT_NS
            else:
                return prediction[:6] + [last], True

        if gap <= 12:
            self.phase = "tracking"
            self.deadline = 0
            return prediction, False
        return previous[:6] + [max(last, prediction[6])], True

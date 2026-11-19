"""Compare power, position feedback and reversals using archived CAN frames only."""
import hashlib
import json
import sys

import numpy as np

from compare_onsets import channels, clock, load_window, ns

MS = 1_000_000
CASES = {
    "A": ("2026-11-18T21:14:04", 715641, [2, 5], 11500),
    "B": ("2026-11-19T21:55:21", 902717, [6, 4], 9900),
    "C": ("2026-11-19T15:57:50", 652971, [3], 20200),
}


def reversals(values, reference, cutoff):
    shifts = values - reference
    signs = np.sign(shifts[np.abs(shifts) >= cutoff])
    return int(np.count_nonzero(np.diff(signs)))


def analyze(label, pattern):
    second, micros, axes, duration = CASES[label]
    start = ns(second) + micros * 1000
    raw = load_window(pattern, start - 9000 * MS, start + duration * MS)
    series = channels(raw)
    print("CASE", label, clock(raw["wall"][0]), clock(raw["wall"][-1]),
          "frames", len(raw), "window_sha256", hashlib.sha256(raw.tobytes()).hexdigest())
    for axis in axes:
        z = series[axis]
        qt, q = z["q"]
        t, v = z["v"]
        amps = z["i"][1]
        motor = z["motor_raw"][1]
        base = (t >= start-9000*MS) & (t < start-1000*MS)
        qbase = (qt >= start-9000*MS) & (qt < start-1000*MS)
        ref_q, ref_m = np.median(q[qbase]), np.median(motor[base])
        ref_v, ref_i = np.median(v[base]), np.median(amps[base])
        near = (t >= start) & (t < start+250*MS)
        # Reference ranges include the full eight-second quiet interval.
        print("AXIS", axis, "base_q_motor_ranges",
              [q[qbase].min(), q[qbase].max(), int(motor[base].min()), int(motor[base].max())],
              "base_v_i", [ref_v, ref_i])
        # Look at the first drive reports and earlier joint sample nearest each one.
        # A receiver timestamp pairs the records, without assuming sensor independence.
        for k in np.flatnonzero(near)[:4]:
            j = int(qt.searchsorted(t[k], side="right")-1)
            print("PAIRED", axis, round((t[k]-start)/MS, 3),
                  round((t[k]-qt[j])/MS, 3), round(q[j]-ref_q, 3),
                  float(motor[k]-ref_m), round(v[k], 3), round(amps[k], 3))
        joint_mask = (qt >= start) & (qt < start+250*MS)
        print("SPAN_250", axis, round(float(np.ptp(q[joint_mask])), 3),
              int(np.ptp(motor[near])))
        deviations = (v, ref_v, .2), (amps, ref_i, .5)
        for field, (data, ref, threshold) in zip(("v", "i"), deviations):
            counts = []
            for offset in range(-9000, -1000, 250):
                interval = (t >= start+offset*MS) & (t < start+(offset+250)*MS)
                counts.append(reversals(data[interval], ref, threshold))
            print("REV_250", axis, field, reversals(data[near], ref, threshold),
                  "base_max", max(counts))
    for axis in range(1, 7):
        low = raw[(raw["cid"] == 0x260 + axis) & (raw["dlc"] == 8)]
        t = low["wall"].astype("i8")
        octets = np.frombuffer(low["data"].tobytes(), dtype="u1").reshape(-1, 8)
        volts = np.frombuffer(low["data"].tobytes(), dtype=">u2").reshape(-1, 4)[:, 0] / 10
        prior = t < start
        base = (t >= start-9000*MS) & (t < start-1000*MS)
        close = (t >= start-500*MS) & (t < start+500*MS)
        k = np.searchsorted(t, start)
        below = np.flatnonzero((~prior) & (volts < volts[base].min()))
        print("POWER", axis, "baseline", [volts[base].min(), volts[base].max()],
              "near", [volts[close].min(), volts[close].max()],
              "prev_next_ms_volts", [round((t[n]-start)/MS, 3) for n in (k-1, k)],
              volts[k-1:k+1].tolist(), "max_near_gap_ms", round(np.diff(t[close]).max()/MS, 3),
              "status", np.unique(octets[close, 5]).tolist(),
              "first_below_base", ([round((t[below[0]]-start)/MS, 3), volts[below[0]]]
                                    if len(below) else None))
    near_raw = raw[(raw["wall"] >= start-500*MS) & (raw["wall"] < start+500*MS)]
    status = near_raw[(near_raw["cid"] == 0x2a1) & (near_raw["dlc"] == 8)]
    print("ARM_NEAR", [(x.tobytes().hex(), int(n))
                       for x, n in zip(*np.unique(status["data"], return_counts=True))],
          "CAN_ERR_NEAR", int(np.count_nonzero(near_raw["cid"] & 0x20000000)))


if __name__ == "__main__":
    for key, source in json.loads(sys.argv[1]).items():
        analyze(key, source)

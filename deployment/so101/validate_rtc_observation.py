"""Measure an already running RTC observation session; never access motor buses."""

import argparse
import json
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    initial = json.loads((root / "state_status.json").read_text())
    records = {}
    started = time.time()
    while time.time() - started < args.seconds:
        data = json.loads((root / "received_action.json").read_text())
        message = data["message"]
        if data["received_at"] >= started:
            records[message["rtc"]["chunk_id"]] = dict(
                elapsed=message["inference_seconds"],
                age=data["received_at"] - message["timestamp/ms"] / 1000,
                rtc=message["rtc"],
                finite=bool(np.isfinite(message["left_arm"]["actions"]).all()
                            and np.isfinite(message["right_arm"]["actions"]).all()),
            )
        time.sleep(0.05)
    final = json.loads((root / "state_status.json").read_text())
    values = list(records.values())
    report = dict(
        started_at=started, finished_at=time.time(), chunks=len(values),
        session_count=len({v["rtc"]["session"] for v in values}),
        queue_resets_during_window=final["rtc_resets"] - initial["rtc_resets"],
        all_finite=bool(values) and all(v["finite"] for v in values),
        motor_commands_sent=final["motor_commands_sent"], stop_latch=(root / "STOP_MOTION").exists(),
        final_consumer=final["rtc"], records=values,
    )
    if values:
        report["inference_seconds"] = {
            key: float(np.percentile([v["elapsed"] for v in values], percentile))
            for key, percentile in [("p50", 50), ("p95", 95), ("max", 100)]
        }
        report["max_arrival_age_seconds"] = max(v["age"] for v in values)
    target = root / "local_step100000/rtc_live_validation.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    return 0 if (report["all_finite"] and report["session_count"] == 1
                 and report["queue_resets_during_window"] == 0
                 and report["motor_commands_sent"] == 0 and report["stop_latch"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

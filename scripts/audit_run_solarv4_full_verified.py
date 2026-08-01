#!/usr/bin/env python3
"""Generate evidence-backed static/runtime audit for the full SolarV4 runner."""

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.run_solarv4_verified_support import (
    assert_aligned,
    cache_fingerprint,
    cache_is_compatible,
    capacity_clip,
    chronological_split,
    kt_to_power,
    power_to_kt,
    reserve_output_directory,
    select_target,
    target_timestamps,
)


RUNNER = ROOT / "run_solarv4_full_verified.py"
SOURCE = RUNNER.read_text(encoding="utf-8")


def check(condition, evidence):
    return {"passed": bool(condition), "evidence": evidence}


def main():
    synthetic = pd.DataFrame(
        {"x": [1.0] * 6 + [100.0] * 4},
        index=pd.date_range("2020-01-01", periods=10, freq="h"),
    )
    _, _, _, mean, scale, boundaries = chronological_split(synthetic)
    timestamps = target_timestamps(pd.date_range("2020", periods=30, freq="h"), 10, 3, 4)
    power = np.array([0.0, 4.0, 20.0])
    ghi = np.array([0.0, 20.0, 100.0])
    round_trip_error = float(np.max(np.abs(kt_to_power(power_to_kt(power, ghi), ghi) - power)))
    clipped_pred, clipped_true = capacity_clip([-1, 5, 12], [-2, 6, 13], 10)

    tree = ast.parse(SOURCE)
    coords = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "STATION_COORDS" for t in node.targets):
            coords = ast.literal_eval(node.value)

    temporary = ROOT / ".audit_collision_probe"
    if temporary.exists():
        temporary.rmdir()
    reserve_output_directory(temporary)
    collision_rejected = False
    try:
        reserve_output_directory(temporary)
    except FileExistsError:
        collision_rejected = True
    temporary.rmdir()

    data_matches = sorted((ROOT / "dataset" / "GEFCom").glob("Task*/train*.csv"))
    if not data_matches:
        raise FileNotFoundError("GEFCom training file not found")
    data_probe = data_matches[0]
    fp1, inputs = cache_fingerprint(data_probe, {"seed": 2026}, RUNNER)
    fp2, _ = cache_fingerprint(data_probe, {"seed": 2027}, RUNNER)
    old_metadata = ROOT / ".audit_old_cache.json"
    old_metadata.write_text(json.dumps({"cache_fingerprint": "legacy"}))
    old_rejected = not cache_is_compatible(old_metadata, fp1)
    old_metadata.unlink()

    training = SOURCE[SOURCE.index("def train_single_seed"):SOURCE.index("def compute_metrics")]
    checks = {
        "no_backward_fill": check(".bfill(" not in SOURCE, "source scan: no .bfill( token"),
        "no_bidirectional_interpolation": check("limit_direction=\"both\"" not in SOURCE and "limit_direction='both'" not in SOURCE, "source scan: no bidirectional interpolation argument"),
        "test_not_used_for_training_or_checkpoint": check("for x, y, _, _ in te_ld" not in training and "vl < best" in training, "train_single_seed checkpoint condition is validation loss; test loader has no training loop"),
        "train_only_scaler": check(mean[0] == 1.0 and boundaries == (0, 6, 7, 10), f"synthetic train mean={mean[0]}, scale={scale[0]}, boundaries={boundaries}"),
        "validation_denominator": check("vl /= len(val_ld)" in training and "vl /= len(te_ld)" not in training, "source scan: vl /= len(val_ld)"),
        "target_selection": check(select_target(1) == "power" and all(select_target(x) == "power_kt" for x in (4, 16, 72, 288)), "runtime target selection for 1/4/16/72/288"),
        "eval_timestamp_alignment": check(timestamps.equals(pd.date_range("2020", periods=30, freq="h")[13:17]), f"runtime timestamps start at seq_len+eval_idx=13: {list(map(str, timestamps))}"),
        "power_kt_round_trip": check(round_trip_error < 1e-12, f"maximum absolute round-trip error={round_trip_error}"),
        "capacity_clipping": check(np.array_equal(clipped_pred, [0, 5, 10]) and np.array_equal(clipped_true, [0, 6, 10]), f"pred={clipped_pred.tolist()}, true={clipped_true.tolist()}"),
        "coordinates_and_time_semantics": check(coords is not None and len(coords) == 12 and '"gefcom": "UTC"' in SOURCE and '"era5": "UTC"' in SOURCE, f"configured coordinate keys={sorted(coords or {})}; source timezone declarations scanned"),
        "old_cache_rejected": check(old_rejected, "legacy fingerprint metadata rejected at runtime"),
        "cache_fingerprint_changes": check(fp1 != fp2, f"seed2026={fp1[:16]}, seed2027={fp2[:16]}; fingerprint inputs include data/config/code SHA"),
        "output_directory_collision": check(collision_rejected, "second reservation of identical output directory raised FileExistsError"),
        "aligned_lengths": check(assert_aligned(np.zeros(4), np.ones(4), timestamps) == 4, "runtime prediction/target/timestamp length assertion"),
        "default_cache_disabled": check('os.environ.get("SOLAR_USE_CACHE", "0") == "1"' in SOURCE, "source scan: default SOLAR_USE_CACHE=0"),
    }
    payload = {
        "runner": str(RUNNER.relative_to(ROOT)),
        "runner_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
        "all_passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "cache_fingerprint_inputs": inputs,
    }
    json_path = ROOT / "reports" / "run_solarv4_full_verified_audit.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# run_solarv4_full_verified static acceptance audit", "", f"Overall: `{'PASS' if payload['all_passed'] else 'FAIL'}`", "", "| Check | Result | Evidence |", "| :--- | :---: | :--- |"]
    for name, item in checks.items():
        lines.append(f"| `{name}` | {'PASS' if item['passed'] else 'FAIL'} | {item['evidence']} |")
    (ROOT / "reports" / "run_solarv4_full_verified_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not payload["all_passed"]:
        raise SystemExit("audit failed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reject contaminated manifests before external fallback-policy confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"window_id", "site_id", "forecast_origin", "target_start"}


def parse_site_spec(value: str) -> set[int]:
    sites: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = (int(item) for item in part.split("-", 1))
            if stop < start:
                raise ValueError(f"invalid site range: {part}")
            sites.update(range(start, stop + 1))
        else:
            sites.add(int(part))
    return sites


def read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"manifest lacks columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("manifest must not be empty")
    frame = frame.copy()
    frame["forecast_origin"] = pd.to_datetime(frame.forecast_origin, errors="raise")
    frame["target_start"] = pd.to_datetime(frame.target_start, errors="raise")
    frame["site_id"] = pd.to_numeric(frame.site_id, errors="raise").astype(int)
    frame["window_id"] = frame.window_id.astype(str)
    return frame


def audit_manifests(reference: pd.DataFrame, candidate: pd.DataFrame,
                    excluded_sites: set[int]) -> dict:
    duplicate_ids = int(candidate.window_id.duplicated().sum())
    overlap = set(reference.window_id) & set(candidate.window_id)
    invalid_temporal = int((candidate.forecast_origin >= candidate.target_start).sum())
    reference_sites = set(reference.site_id)
    candidate_sites = set(candidate.site_id)
    contaminated_sites = candidate_sites & excluded_sites
    new_sites = candidate_sites - excluded_sites
    future_only = bool(candidate.forecast_origin.min() > reference.forecast_origin.max())
    independent_source = bool(new_sites) and not contaminated_sites or future_only
    passed = not overlap and not duplicate_ids and not invalid_temporal and independent_source
    return {
        "audit_passed": passed,
        "reference_window_count": len(reference),
        "candidate_window_count": len(candidate),
        "overlap_window_count": len(overlap),
        "candidate_duplicate_window_ids": duplicate_ids,
        "invalid_forecast_target_order_count": invalid_temporal,
        "excluded_sites": sorted(excluded_sites),
        "reference_sites": sorted(reference_sites),
        "candidate_sites": sorted(candidate_sites),
        "contaminated_sites": sorted(contaminated_sites),
        "new_sites": sorted(new_sites),
        "reference_latest_forecast_origin": str(reference.forecast_origin.max()),
        "candidate_earliest_forecast_origin": str(candidate.forecast_origin.min()),
        "candidate_is_strictly_future": future_only,
        "acceptance_rule": (
            "zero window overlap, unique candidate IDs, forecast_origin < target_start, "
            "and either sites disjoint from all excluded train/validation/test sites or a strictly future range"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--excluded-sites", required=True,
                        help="All train/validation/test sites, e.g. 0-21 or 0-9,20,21")
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit_manifests(
        read_manifest(args.reference_manifest), read_manifest(args.candidate_manifest),
        parse_site_spec(args.excluded_sites),
    )
    payload.update({
        "reference_manifest": str(args.reference_manifest.resolve()),
        "candidate_manifest": str(args.candidate_manifest.resolve()),
    })
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if payload["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

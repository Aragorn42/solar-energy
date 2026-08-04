import math

import pandas as pd
import pytest

from stage4_fallback_policy import Stage4FallbackPolicy
from scripts.preflight_stage4c_external_confirmation import audit_manifests, parse_site_spec


def manifest(ids, sites, origins):
    origins = pd.to_datetime(origins)
    return pd.DataFrame({
        "window_id": ids, "site_id": sites, "forecast_origin": origins,
        "target_start": origins + pd.Timedelta(hours=1),
    })


def test_policy_routes_endpoints_with_numeric_tolerance():
    policy = Stage4FallbackPolicy(endpoint_tolerance=1e-6)
    assert policy.select_group(0.0) == "quality_gate"
    assert policy.select_group(1e-7) == "quality_gate"
    assert policy.select_group(0.5) == "cora"
    assert policy.select_group(1 - 1e-7) == "static_mask_gate"
    assert policy.select_group(1.0) == "static_mask_gate"


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf, -math.inf])
def test_policy_rejects_invalid_ratios(value):
    with pytest.raises(ValueError):
        Stage4FallbackPolicy().select_group(value)


def test_site_spec_supports_ranges_and_values():
    assert parse_site_spec("0-2,5,7-8") == {0, 1, 2, 5, 7, 8}


def test_preflight_accepts_disjoint_new_sites():
    reference = manifest(["a"], [0], ["2022-01-01"])
    candidate = manifest(["b"], [22], ["2022-01-01"])
    audit = audit_manifests(reference, candidate, set(range(22)))
    assert audit["audit_passed"]


def test_preflight_rejects_validation_site_even_if_absent_from_reference():
    reference = manifest(["a"], [0], ["2022-01-01"])
    candidate = manifest(["b"], [20], ["2022-01-01"])
    audit = audit_manifests(reference, candidate, set(range(22)))
    assert not audit["audit_passed"]
    assert audit["contaminated_sites"] == [20]


def test_preflight_rejects_overlap_duplicates_and_bad_time_order():
    reference = manifest(["a"], [0], ["2022-01-01"])
    candidate = manifest(["a", "a"], [22, 22], ["2022-01-01", "2022-01-01"])
    candidate["target_start"] = candidate.forecast_origin
    audit = audit_manifests(reference, candidate, set(range(22)))
    assert not audit["audit_passed"]
    assert audit["overlap_window_count"] == 1
    assert audit["candidate_duplicate_window_ids"] == 1
    assert audit["invalid_forecast_target_order_count"] == 2


def test_preflight_allows_strict_future_for_known_sites():
    reference = manifest(["a"], [0], ["2022-01-01"])
    candidate = manifest(["b"], [0], ["2022-02-01"])
    audit = audit_manifests(reference, candidate, set(range(22)))
    assert audit["audit_passed"]
    assert audit["candidate_is_strictly_future"]

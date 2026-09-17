from __future__ import annotations

from linkage.frequencies import compute, field_frequencies
from linkage.generate.corrupt import ABSENT, MISSING


def _case(crime_type, **mo_core):
    return {"crime_type": crime_type, "mo_core": mo_core, "mo_ext": {}}


def test_field_frequencies_excludes_sentinels():
    cases = [
        _case("VEHICLE_THEFT", time_band="night"),
        _case("VEHICLE_THEFT", time_band="night"),
        _case("VEHICLE_THEFT", time_band=MISSING),
        _case("VEHICLE_THEFT", time_band="day"),
    ]
    freqs = field_frequencies(cases, "time_band", "mo_core")
    # MISSING excluded from both numerator and denominator: 2/3 night, 1/3 day
    assert freqs["night"] == 2 / 3
    assert freqs["day"] == 1 / 3
    assert MISSING not in freqs


def test_field_frequencies_empty_pool_is_empty_dict():
    assert field_frequencies([], "time_band", "mo_core") == {}


def test_tag_field_frequencies_are_per_tag_not_categorical():
    cases = [
        _case("BURGLARY_RESIDENTIAL", tools=["crowbar", "cutter"]),
        _case("BURGLARY_RESIDENTIAL", tools=["crowbar"]),
        _case("BURGLARY_RESIDENTIAL", tools=["crowbar", "cutter", "gas_cutter"]),
        _case("BURGLARY_RESIDENTIAL", tools=[]),
    ]
    freqs = field_frequencies(cases, "tools", "mo_core")
    assert freqs["crowbar"] == 3 / 4
    assert freqs["cutter"] == 2 / 4
    assert freqs["gas_cutter"] == 1 / 4
    # tag rates are independent per-tag Bernoulli rates, not a categorical
    # distribution — they need not (and here don't) sum to 1
    assert sum(freqs.values()) != 1


def test_compute_pools_cross_type_across_all_crime_types():
    cases = [
        _case("VEHICLE_THEFT", time_band="night"),
        _case("SNATCHING", time_band="night"),
        _case("SNATCHING", time_band="day"),
    ]
    freqs = compute(cases)
    assert freqs["cross_type"]["time_band"]["night"] == 2 / 3
    # same_type pools stay separate per crime_type
    assert freqs["same_type"]["VEHICLE_THEFT"]["time_band"] == {"night": 1.0}
    assert freqs["same_type"]["SNATCHING"]["time_band"] == {"night": 0.5, "day": 0.5}

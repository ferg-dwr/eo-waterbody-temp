"""CDEC water-temperature parser, grounded on a verbatim live LIS response."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from eo_waterbody_temp.cdec import (
    LIS_LATLON,
    WATER_TEMP_SENSOR,
    CdecTemperatureSource,
    TemperatureReading,
    parse_cdec_csv,
)

FIXTURE = Path(__file__).parent / "fixtures" / "cdec" / "lis_sensor25_20260130.csv"

_HDR = (
    "STATION_ID,DURATION,SENSOR_NUMBER,SENSOR_TYPE,DATE TIME,"
    "OBS DATE,VALUE,DATA_FLAG,UNITS\n"
)


def test_parses_only_numeric_values():
    readings = parse_cdec_csv(FIXTURE.read_text())
    # 4 real values (52.4), 3 '---' rows skipped
    assert len(readings) == 4
    assert all(isinstance(r, TemperatureReading) for r in readings)


def test_fahrenheit_converted_to_celsius():
    readings = parse_cdec_csv(FIXTURE.read_text())
    # 52.4 F -> 11.333... C
    assert readings[0].temp_c == pytest.approx((52.4 - 32) * 5 / 9, abs=1e-6)
    assert readings[0].source_units == "DEG F"


def test_pst_timestamps_become_aware_utc():
    readings = parse_cdec_csv(FIXTURE.read_text())
    first = readings[0]
    assert first.time_utc.tzinfo is not None
    # 2026-01-30 00:00 PST == 2026-01-30 08:00 UTC
    assert first.time_utc == datetime(2026, 1, 30, 8, 0, tzinfo=UTC)


def test_readings_sorted_by_time():
    readings = parse_cdec_csv(FIXTURE.read_text())
    times = [r.time_utc for r in readings]
    assert times == sorted(times)


def test_degc_units_pass_through():
    text = _HDR + "XYZ,E,146,TEMP W,20260201 1200,20260201 1200,11.3, ,DEG C\n"
    r = parse_cdec_csv(text)[0]
    assert r.temp_c == pytest.approx(11.3)
    assert r.source_units == "DEG C"


def test_unexpected_units_raises():
    text = _HDR + "XYZ,E,25,TEMP W,20260201 1200,20260201 1200,300, ,KELVIN\n"
    with pytest.raises(ValueError, match="unexpected temperature units"):
        parse_cdec_csv(text)


def test_nearest_within_gap():
    src = CdecTemperatureSource()
    readings = parse_cdec_csv(FIXTURE.read_text())
    # target 00:20 PST = 08:20 UTC; nearest real reading is 00:15 (08:15 UTC)
    target = datetime(2026, 1, 30, 8, 20, tzinfo=UTC)
    r = src.nearest(target, readings, max_gap=timedelta(minutes=30))
    assert r is not None
    assert r.time_utc == datetime(2026, 1, 30, 8, 15, tzinfo=UTC)


def test_nearest_returns_none_beyond_gap():
    src = CdecTemperatureSource()
    readings = parse_cdec_csv(FIXTURE.read_text())
    # readings end 00:45 PST (08:45 UTC); a target 6h later has no match in-gap
    target = datetime(2026, 1, 30, 14, 45, tzinfo=UTC)
    assert src.nearest(target, readings, max_gap=timedelta(hours=1)) is None


def test_nearest_requires_aware_time():
    src = CdecTemperatureSource()
    with pytest.raises(ValueError, match="timezone-aware"):
        src.nearest(datetime(2026, 1, 30, 8, 0), [])


def test_verified_constants():
    assert WATER_TEMP_SENSOR == 25
    assert LIS_LATLON == (38.474781, -121.588226)


def test_numeric_bad_read_sentinel_dropped():
    # real LIS feed emits 99999.0 as a bad-read sentinel (NOT '---'); it must not
    # survive as a ~55537 degC reading. Grounded on the 2026-01-31 sample.
    fx = Path(__file__).parent / "fixtures" / "cdec" / "lis_sensor25_20260131.csv"
    readings = parse_cdec_csv(fx.read_text())
    assert len(readings) == 3  # 4 numeric rows, the 99999 dropped
    assert all(-1.0 <= r.temp_c <= 45.0 for r in readings)
    assert max(r.temp_c for r in readings) < 20  # ~11.3 C, nowhere near 55000


def test_non_physical_values_dropped():
    text = _HDR + "XYZ,E,25,TEMP W,20260201 1200,20260201 1200,99999.0, ,DEG F\n"
    assert parse_cdec_csv(text) == []

"""In-situ water temperature from CDEC (the skin->bulk validation anchor).

A ``TemperatureGaugeSource`` answers one question: what was the bulk water
temperature, in degrees Celsius, at a station near a given UTC time? Used to
validate ECOSTRESS water-*surface* (skin) temperature against an in-water gauge.

Conventions mirror (do not import) the sibling repo's gauges.py, grounded on a
live LIS response (2026-07):
  - CSVDataServlet header: STATION_ID, DURATION, SENSOR_NUMBER, SENSOR_TYPE,
    DATE TIME, OBS DATE, VALUE, DATA_FLAG, UNITS.
  - CDEC times are PST year-round (UTC-8, no DST) -> converted to aware UTC.
  - Water temperature is sensor 25, reported in DEG F at LIS -> converted to degC
    driven by the UNITS column (never assumed; a DEG C station passes through).
  - Missing values are '---' (skipped) AND numeric bad-read sentinels (e.g.
    99999.0) -> dropped via a physical-plausibility range, never zeroed.
  - Pure stdlib: no new dependencies (this module needs neither [io] nor [eo]).

Per-station verification is non-negotiable (the sibling repo found dead station
codes and a datum change): the water-temp sensor/units here were verified against
a live LIS sample, not recalled.
"""

from __future__ import annotations

import csv
import io
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone

PST = timezone(timedelta(hours=-8))  # CDEC reports PST year-round (no DST)
CDEC_CSV_URL = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet"
WATER_TEMP_SENSOR = 25  # verified live at LIS (SENSOR_TYPE "TEMP W", DEG F)

# Physically plausible liquid-water range (degC). CDEC also flags bad reads with
# numeric sentinels (e.g. 99999.0) that are NOT "---", so a non-numeric skip is
# not enough; anything outside this range is dropped as non-physical.
PLAUSIBLE_C = (-1.0, 45.0)

# LIS = Yolo Bypass at Lisbon (Toe Drain), verified 2026-07 via CDEC staMeta.
LIS_STATION = "LIS"
LIS_LATLON = (38.474781, -121.588226)


def _f_to_c(value: float) -> float:
    return (value - 32.0) * 5.0 / 9.0


@dataclass(frozen=True)
class TemperatureReading:
    """One in-situ water-temperature observation, normalized to degrees Celsius."""

    station: str
    time_utc: datetime  # timezone-aware, UTC
    temp_c: float
    source_units: str  # units as reported (e.g. "DEG F")


def parse_cdec_csv(text: str) -> list[TemperatureReading]:
    """Parse a CDEC CSVDataServlet water-temperature response into readings (degC).

    Non-numeric VALUEs (e.g. '---') are skipped. PST timestamps become aware UTC.
    Units are converted per the UNITS column (DEG F -> degC; DEG C passthrough).
    """
    out: list[TemperatureReading] = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        raw = (row.get("VALUE") or "").strip()
        try:
            value = float(raw)
        except ValueError:
            continue  # '---' and other non-numerics: skip, never zero
        units = (row.get("UNITS") or "").strip().upper()
        if units == "DEG F":
            temp_c = _f_to_c(value)
        elif units == "DEG C":
            temp_c = value
        else:
            raise ValueError(f"unexpected temperature units: {units!r}")
        lo, hi = PLAUSIBLE_C
        if not (lo <= temp_c <= hi):
            continue  # numeric bad-read sentinel (e.g. 99999) or non-physical
        t = datetime.strptime(row["DATE TIME"].strip(), "%Y%m%d %H%M")
        time_utc = t.replace(tzinfo=PST).astimezone(UTC)
        out.append(
            TemperatureReading(
                station=row["STATION_ID"].strip(),
                time_utc=time_utc,
                temp_c=temp_c,
                source_units=units,
            )
        )
    out.sort(key=lambda r: r.time_utc)
    return out


class TemperatureGaugeSource(ABC):
    """Interface for in-situ water temperature used to validate skin retrievals."""

    @abstractmethod
    def readings(
        self, start_utc: datetime, end_utc: datetime
    ) -> list[TemperatureReading]:
        """All readings in [start_utc, end_utc], sorted by time."""

    def nearest(
        self,
        when_utc: datetime,
        readings: list[TemperatureReading],
        max_gap: timedelta = timedelta(hours=1),
    ) -> TemperatureReading | None:
        """Reading closest to ``when_utc`` within ``max_gap``, else None.

        Returns None rather than a far-away value: a scene with no in-situ reading
        inside the gap is UNVALIDATED, not silently matched to a stale observation.
        """
        if when_utc.tzinfo is None:
            raise ValueError("when_utc must be timezone-aware (pass UTC).")
        best = None
        best_gap = max_gap
        for r in readings:
            gap = abs(r.time_utc - when_utc)
            if gap <= best_gap:
                best, best_gap = r, gap
        return best


class CdecTemperatureSource(TemperatureGaugeSource):
    """Water temperature from a CDEC station via the public CSVDataServlet (no auth)."""

    def __init__(
        self,
        station: str = LIS_STATION,
        sensor: int = WATER_TEMP_SENSOR,
        dur_code: str = "E",
    ):
        self.station = station
        self.sensor = sensor
        self.dur_code = dur_code

    def _fetch(self, start_utc: datetime, end_utc: datetime) -> str:
        # CDEC Start/End are PST dates; widen a day each side so the PST/UTC
        # boundary can't clip readings near the window edges.
        start_d = (start_utc.astimezone(PST) - timedelta(days=1)).date()
        end_d = (end_utc.astimezone(PST) + timedelta(days=1)).date()
        params = urllib.parse.urlencode(
            {
                "Stations": self.station,
                "SensorNums": self.sensor,
                "dur_code": self.dur_code,
                "Start": start_d.isoformat(),
                "End": end_d.isoformat(),
            }
        )
        with urllib.request.urlopen(f"{CDEC_CSV_URL}?{params}") as resp:
            return resp.read().decode("utf-8", errors="replace")

    def readings(
        self, start_utc: datetime, end_utc: datetime
    ) -> list[TemperatureReading]:
        all_readings = parse_cdec_csv(self._fetch(start_utc, end_utc))
        return [r for r in all_readings if start_utc <= r.time_utc <= end_utc]

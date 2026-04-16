"""
Macro market data — loads India VIX, USD/INR, Brent Crude, sector indices,
and provides text-formatted context for LLM observations.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Historical RBI repo rate changes (date -> rate %).
# Forward-filled: use the most recent rate on or before the query date.
RBI_REPO_RATE_HISTORY: list[tuple[str, float]] = [
    ("2019-10-04", 5.15),
    ("2020-03-27", 4.40),
    ("2020-05-22", 4.00),
    ("2022-05-04", 4.40),
    ("2022-06-08", 4.90),
    ("2022-08-05", 5.40),
    ("2022-09-30", 5.90),
    ("2022-12-07", 6.25),
    ("2023-02-08", 6.50),
    ("2025-02-07", 6.25),
    ("2025-04-09", 6.00),
]

# Notable market calendar events (month-day -> description).
# Recurring annual events + known one-off dates.
MARKET_CALENDAR: list[tuple[str, str]] = [
    # Union Budget (annual, Feb 1)
    ("2020-02-01", "Union Budget"),
    ("2021-02-01", "Union Budget"),
    ("2022-02-01", "Union Budget"),
    ("2023-02-01", "Union Budget"),
    ("2024-02-01", "Interim Budget"),
    ("2024-07-23", "Union Budget"),
    ("2025-02-01", "Union Budget"),
    ("2026-02-01", "Union Budget"),
    # General Elections
    ("2024-04-19", "General Election Phase 1"),
    ("2024-06-04", "Election Results"),
    # RBI MPC meetings (key dates — announce days)
    ("2024-02-08", "RBI MPC Decision"),
    ("2024-04-05", "RBI MPC Decision"),
    ("2024-06-07", "RBI MPC Decision"),
    ("2024-08-08", "RBI MPC Decision"),
    ("2024-10-09", "RBI MPC Decision"),
    ("2024-12-06", "RBI MPC Decision"),
    ("2025-02-07", "RBI MPC Decision"),
    ("2025-04-09", "RBI MPC Decision"),
    ("2025-06-06", "RBI MPC Decision"),
    ("2025-08-06", "RBI MPC Decision"),
    ("2025-10-08", "RBI MPC Decision"),
    ("2025-12-05", "RBI MPC Decision"),
]

# VIX classification thresholds
VIX_LOW = 13.0
VIX_NORMAL = 18.0
VIX_ELEVATED = 24.0


def _parse_calendar() -> dict[date, str]:
    """Convert calendar list to date-keyed dict."""
    return {
        datetime.strptime(d, "%Y-%m-%d").date(): desc
        for d, desc in MARKET_CALENDAR
    }


def _build_rbi_rate_series() -> list[tuple[date, float]]:
    """Convert RBI rate history to sorted (date, rate) pairs."""
    return [
        (datetime.strptime(d, "%Y-%m-%d").date(), rate)
        for d, rate in RBI_REPO_RATE_HISTORY
    ]


def get_rbi_rate(query_date: date) -> tuple[float, date]:
    """Get the RBI repo rate effective on a given date.

    Returns (rate, effective_date) — the most recent rate change
    on or before query_date.
    """
    rates = _build_rbi_rate_series()
    effective_rate = rates[0][1]
    effective_date = rates[0][0]

    for change_date, rate in rates:
        if change_date <= query_date:
            effective_rate = rate
            effective_date = change_date
        else:
            break

    return effective_rate, effective_date


def get_upcoming_events(query_date: date, lookahead_days: int = 5) -> list[tuple[date, str]]:
    """Get calendar events within lookahead_days of query_date."""
    calendar = _parse_calendar()
    upcoming = []
    for i in range(1, lookahead_days + 1):
        check_date = query_date + timedelta(days=i)
        if check_date in calendar:
            upcoming.append((check_date, calendar[check_date]))
    return upcoming


def load_macro_data(macro_dir: Path) -> dict[str, pd.DataFrame] | None:
    """Load all macro CSVs from a directory.

    Returns dict keyed by macro name (e.g., "INDIA_VIX"), or None
    if the directory doesn't exist or is empty.
    """
    if not macro_dir.exists():
        logger.info("Macro data directory not found: %s", macro_dir)
        return None

    macro_data: dict[str, pd.DataFrame] = {}

    for csv_path in sorted(macro_dir.glob("*_daily.csv")):
        name = csv_path.stem.replace("_daily", "")
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        macro_data[name] = df

    if not macro_data:
        logger.info("No macro CSVs found in %s", macro_dir)
        return None

    logger.info("Loaded %d macro instruments", len(macro_data))
    return macro_data


def _get_value_at_date(
    df: pd.DataFrame,
    query_date: date,
    column: str = "close",
) -> float | None:
    """Get the value at or before a given date (forward-fill logic)."""
    ts = pd.Timestamp(query_date)
    mask = df["timestamp"] <= ts
    if not mask.any():
        return None
    row = df.loc[mask].iloc[-1]
    return float(row[column])


def _get_daily_change(
    df: pd.DataFrame,
    query_date: date,
    column: str = "close",
) -> float | None:
    """Get the daily % change at a given date."""
    ts = pd.Timestamp(query_date)
    mask = df["timestamp"] <= ts
    matching = df.loc[mask]
    if len(matching) < 2:
        return None
    today = float(matching.iloc[-1][column])
    yesterday = float(matching.iloc[-2][column])
    if yesterday == 0:
        return None
    return (today - yesterday) / yesterday * 100


def _get_rolling_avg(
    df: pd.DataFrame,
    query_date: date,
    column: str = "close",
    window: int = 15,
) -> float | None:
    """Get the rolling average ending at or before query_date."""
    ts = pd.Timestamp(query_date)
    mask = df["timestamp"] <= ts
    matching = df.loc[mask]
    if len(matching) < window:
        return None
    return float(matching[column].iloc[-window:].mean())


def _classify_vix(vix: float) -> str:
    """Classify India VIX level."""
    if vix < VIX_LOW:
        return "low"
    elif vix < VIX_NORMAL:
        return "normal"
    elif vix < VIX_ELEVATED:
        return "elevated"
    return "high"


def get_macro_snapshot(
    macro_data: dict[str, pd.DataFrame],
    query_date: date,
) -> dict:
    """Build a macro snapshot for a given date.

    Returns a dict with all macro signals, or empty dict if
    critical data is missing.
    """
    snapshot: dict = {}

    # India VIX
    if "INDIA_VIX" in macro_data:
        vix = _get_value_at_date(macro_data["INDIA_VIX"], query_date)
        vix_avg = _get_rolling_avg(macro_data["INDIA_VIX"], query_date, window=15)
        if vix is not None:
            snapshot["vix"] = round(vix, 1)
            snapshot["vix_avg_15d"] = round(vix_avg, 1) if vix_avg else None
            snapshot["vix_label"] = _classify_vix(vix)

    # USD/INR
    if "USDINR" in macro_data:
        usdinr = _get_value_at_date(macro_data["USDINR"], query_date)
        usdinr_change = _get_daily_change(macro_data["USDINR"], query_date)
        if usdinr is not None:
            snapshot["usdinr"] = round(usdinr, 2)
            snapshot["usdinr_change"] = round(usdinr_change, 2) if usdinr_change else 0.0

    # Brent Crude
    if "BRENT_CRUDE" in macro_data:
        brent = _get_value_at_date(macro_data["BRENT_CRUDE"], query_date)
        brent_change = _get_daily_change(macro_data["BRENT_CRUDE"], query_date)
        if brent is not None:
            snapshot["brent"] = round(brent, 1)
            snapshot["brent_change"] = round(brent_change, 2) if brent_change else 0.0

    # Sector indices
    sectors = {}
    for name, label in [("NIFTY_BANK", "Bank"), ("NIFTY_IT", "IT"), ("NIFTY_PHARMA", "Pharma")]:
        if name in macro_data:
            change = _get_daily_change(macro_data[name], query_date)
            if change is not None:
                sectors[label] = round(change, 1)
    if sectors:
        snapshot["sectors"] = sectors
        # Determine which sector is leading
        leading = max(sectors, key=sectors.get)
        if sectors[leading] > 0 and leading in ("Bank",):
            snapshot["rotation_signal"] = "cyclicals leading"
        elif sectors[leading] > 0 and leading in ("IT", "Pharma"):
            snapshot["rotation_signal"] = "defensives leading"
        else:
            snapshot["rotation_signal"] = "mixed"

    # RBI repo rate
    rate, effective_date = get_rbi_rate(query_date)
    snapshot["rbi_rate"] = rate
    snapshot["rbi_last_change"] = effective_date.strftime("%b %Y")

    # Upcoming events
    events = get_upcoming_events(query_date)
    if events:
        snapshot["upcoming_events"] = [
            (d.strftime("%b %d"), desc) for d, desc in events
        ]

    return snapshot


def macro_to_text(snapshot: dict) -> str:
    """Format a macro snapshot as text for LLM observations.

    Returns empty string if snapshot is empty.
    """
    if not snapshot:
        return ""

    lines = ["Market Context:"]

    # VIX
    if "vix" in snapshot:
        vix_str = f"  India VIX: {snapshot['vix']:.1f} ({snapshot['vix_label']}"
        if snapshot.get("vix_avg_15d"):
            comparison = "above" if snapshot["vix"] > snapshot["vix_avg_15d"] else "below"
            vix_str += f", {comparison} 15-day avg of {snapshot['vix_avg_15d']:.1f}"
        vix_str += ")"
        lines.append(vix_str)

    # USD/INR
    if "usdinr" in snapshot:
        change = snapshot.get("usdinr_change", 0.0)
        direction = "weakening" if change > 0 else "strengthening" if change < 0 else "flat"
        lines.append(
            f"  USD/INR: {snapshot['usdinr']:.2f} ({change:+.1f}% today, {direction})"
        )

    # Brent Crude
    if "brent" in snapshot:
        change = snapshot.get("brent_change", 0.0)
        lines.append(f"  Brent Crude: ${snapshot['brent']:.1f} ({change:+.1f}% today)")

    # Sectors
    if "sectors" in snapshot:
        sector_parts = [
            f"{name} {change:+.1f}%"
            for name, change in snapshot["sectors"].items()
        ]
        rotation = snapshot.get("rotation_signal", "mixed")
        lines.append(f"  Sectors: {' | '.join(sector_parts)} ({rotation})")

    # RBI rate
    if "rbi_rate" in snapshot:
        lines.append(
            f"  RBI Repo Rate: {snapshot['rbi_rate']:.2f}% (last change: {snapshot['rbi_last_change']})"
        )

    # Upcoming events
    if "upcoming_events" in snapshot:
        for event_date, desc in snapshot["upcoming_events"]:
            lines.append(f"  Upcoming: {desc} on {event_date}")

    return "\n".join(lines)

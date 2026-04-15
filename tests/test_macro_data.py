"""Tests for macro data loading, snapshot extraction, and text formatting."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from server.macro_data import (
    get_macro_snapshot,
    get_rbi_rate,
    get_upcoming_events,
    load_macro_data,
    macro_to_text,
)


@pytest.fixture
def macro_dir(tmp_path: Path) -> Path:
    """Create a temporary macro data directory with sample CSVs."""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")  # business days

    # India VIX: starts at 14, rises to 20
    vix_closes = np.linspace(14, 20, len(dates))
    _write_macro_csv(tmp_path, "INDIA_VIX", dates, vix_closes)

    # USD/INR: starts at 83.0, drifts up
    usdinr_closes = np.linspace(83.0, 84.0, len(dates))
    _write_macro_csv(tmp_path, "USDINR", dates, usdinr_closes)

    # Brent Crude: starts at 78, fluctuates
    np.random.seed(42)
    brent_closes = 78 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    _write_macro_csv(tmp_path, "BRENT_CRUDE", dates, brent_closes)

    # Sector indices
    bank_closes = 45000 + np.cumsum(np.random.randn(len(dates)) * 100)
    _write_macro_csv(tmp_path, "NIFTY_BANK", dates, bank_closes)

    it_closes = 35000 + np.cumsum(np.random.randn(len(dates)) * 80)
    _write_macro_csv(tmp_path, "NIFTY_IT", dates, it_closes)

    pharma_closes = 18000 + np.cumsum(np.random.randn(len(dates)) * 50)
    _write_macro_csv(tmp_path, "NIFTY_PHARMA", dates, pharma_closes)

    return tmp_path


def _write_macro_csv(
    directory: Path,
    name: str,
    dates: pd.DatetimeIndex,
    closes: np.ndarray,
) -> None:
    """Write a macro CSV file matching the download format."""
    n = len(dates)
    np.random.seed(hash(name) % 2**31)
    df = pd.DataFrame({
        "timestamp": dates,
        "open": closes - np.random.rand(n) * 0.5,
        "high": closes + np.random.rand(n) * 1.0,
        "low": closes - np.random.rand(n) * 1.0,
        "close": closes,
        "volume": np.zeros(n, dtype=int),
    })
    df.to_csv(directory / f"{name}_daily.csv", index=False)


# --- load_macro_data ---


def test_load_macro_data_returns_all_instruments(macro_dir: Path) -> None:
    data = load_macro_data(macro_dir)
    assert data is not None
    assert len(data) == 6
    assert "INDIA_VIX" in data
    assert "USDINR" in data
    assert "BRENT_CRUDE" in data


def test_load_macro_data_returns_none_for_missing_dir(tmp_path: Path) -> None:
    result = load_macro_data(tmp_path / "nonexistent")
    assert result is None


def test_load_macro_data_returns_none_for_empty_dir(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = load_macro_data(empty)
    assert result is None


def test_loaded_dataframes_have_expected_columns(macro_dir: Path) -> None:
    data = load_macro_data(macro_dir)
    assert data is not None
    for name, df in data.items():
        assert "timestamp" in df.columns, f"{name} missing timestamp"
        assert "close" in df.columns, f"{name} missing close"
        assert len(df) == 30, f"{name} has {len(df)} rows, expected 30"


# --- get_rbi_rate ---


def test_rbi_rate_before_first_change() -> None:
    rate, _ = get_rbi_rate(date(2019, 1, 1))
    assert rate == 5.15  # First entry is the fallback


def test_rbi_rate_on_change_date() -> None:
    rate, effective = get_rbi_rate(date(2020, 5, 22))
    assert rate == 4.00
    assert effective == date(2020, 5, 22)


def test_rbi_rate_between_changes() -> None:
    # Between 2023-02-08 (6.50) and 2025-02-07 (6.25)
    rate, effective = get_rbi_rate(date(2024, 6, 15))
    assert rate == 6.50
    assert effective == date(2023, 2, 8)


def test_rbi_rate_after_latest_change() -> None:
    rate, effective = get_rbi_rate(date(2025, 12, 1))
    assert rate == 6.00
    assert effective == date(2025, 4, 9)


# --- get_upcoming_events ---


def test_upcoming_events_finds_budget() -> None:
    # 3 days before Union Budget 2025
    events = get_upcoming_events(date(2025, 1, 29), lookahead_days=5)
    assert len(events) >= 1
    descriptions = [desc for _, desc in events]
    assert "Union Budget" in descriptions


def test_upcoming_events_empty_when_no_events() -> None:
    events = get_upcoming_events(date(2024, 7, 1), lookahead_days=5)
    assert events == []


# --- get_macro_snapshot ---


def test_snapshot_contains_all_fields(macro_dir: Path) -> None:
    data = load_macro_data(macro_dir)
    assert data is not None
    snapshot = get_macro_snapshot(data, date(2024, 1, 15))
    assert "vix" in snapshot
    assert "vix_label" in snapshot
    assert "usdinr" in snapshot
    assert "brent" in snapshot
    assert "sectors" in snapshot
    assert "rbi_rate" in snapshot


def test_snapshot_vix_classification(macro_dir: Path) -> None:
    data = load_macro_data(macro_dir)
    assert data is not None
    # Early dates: VIX ~14 (normal range)
    snapshot_early = get_macro_snapshot(data, date(2024, 1, 3))
    assert snapshot_early["vix_label"] in ("low", "normal")

    # Late dates: VIX ~20 (elevated)
    snapshot_late = get_macro_snapshot(data, date(2024, 2, 8))
    assert snapshot_late["vix_label"] in ("normal", "elevated")


def test_snapshot_sector_rotation(macro_dir: Path) -> None:
    data = load_macro_data(macro_dir)
    assert data is not None
    snapshot = get_macro_snapshot(data, date(2024, 1, 15))
    assert "sectors" in snapshot
    assert "Bank" in snapshot["sectors"]
    assert "IT" in snapshot["sectors"]
    assert "Pharma" in snapshot["sectors"]
    assert "rotation_signal" in snapshot


def test_snapshot_forward_fills_on_missing_date(macro_dir: Path) -> None:
    """Weekend/holiday dates should use the most recent available data."""
    data = load_macro_data(macro_dir)
    assert data is not None
    # Jan 6 2024 is a Saturday
    snapshot = get_macro_snapshot(data, date(2024, 1, 6))
    # Should still get VIX data (forward-filled from Jan 5)
    assert "vix" in snapshot


# --- macro_to_text ---


def test_macro_to_text_basic_format(macro_dir: Path) -> None:
    data = load_macro_data(macro_dir)
    assert data is not None
    snapshot = get_macro_snapshot(data, date(2024, 1, 15))
    text = macro_to_text(snapshot)

    assert text.startswith("Market Context:")
    assert "India VIX:" in text
    assert "USD/INR:" in text
    assert "Brent Crude:" in text
    assert "Sectors:" in text
    assert "RBI Repo Rate:" in text


def test_macro_to_text_empty_snapshot() -> None:
    text = macro_to_text({})
    assert text == ""


def test_macro_to_text_contains_direction_labels(macro_dir: Path) -> None:
    data = load_macro_data(macro_dir)
    assert data is not None
    snapshot = get_macro_snapshot(data, date(2024, 1, 15))
    text = macro_to_text(snapshot)

    # USD/INR should have a direction label
    assert any(d in text for d in ("weakening", "strengthening", "flat"))


def test_macro_to_text_shows_upcoming_events() -> None:
    snapshot = {
        "vix": 15.0,
        "vix_label": "normal",
        "rbi_rate": 6.50,
        "rbi_last_change": "Feb 2023",
        "upcoming_events": [("Feb 01", "Union Budget")],
    }
    text = macro_to_text(snapshot)
    assert "Union Budget" in text
    assert "Feb 01" in text

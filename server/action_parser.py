"""Action parsing for trading environment.

Parses action strings like 'BUY RELIANCE 0.3', 'SELL TCS', 'HOLD'
into structured dicts. Handles single-stock symbol defaults and
fraction clamping.
"""

from __future__ import annotations


def parse_action(action_str: str, task_config: dict) -> dict:
    """Parse an action string into a structured dict.

    Args:
        action_str: Raw action like 'BUY RELIANCE 0.3' or 'HOLD'
        task_config: Task config with 'symbols' list for validation

    Returns:
        Dict with 'type' (BUY/SELL/HOLD), optional 'symbol', optional 'fraction'
    """
    parts = action_str.strip().upper().split()
    if not parts:
        return {"type": "HOLD"}

    action_type = parts[0]
    if action_type not in ("BUY", "SELL", "HOLD"):
        return {"type": "HOLD"}

    if action_type == "HOLD":
        return {"type": "HOLD"}

    symbols = task_config["symbols"]

    # For single_stock task, default symbol
    if len(parts) == 1 and len(symbols) == 1:
        symbol = symbols[0]
    elif len(parts) >= 2:
        symbol = parts[1]
    else:
        return {"type": "HOLD"}

    # Validate symbol
    if symbol not in symbols:
        return {"type": "HOLD"}

    fraction = 1.0
    if len(parts) >= 3:
        try:
            fraction = float(parts[2])
            fraction = max(0.0, min(1.0, fraction))
        except ValueError:
            fraction = 1.0

    return {"type": action_type, "symbol": symbol, "fraction": fraction}

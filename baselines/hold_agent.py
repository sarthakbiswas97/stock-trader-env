"""Hold agent — always returns HOLD. The absolute floor baseline."""


def hold_agent(observation: str) -> str:
    """Do nothing, every single day."""
    return "HOLD"

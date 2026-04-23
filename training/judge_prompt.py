"""LLM-as-Judge rubric for trading decision quality.

Defines the scoring rubric, builds chat prompts for the judge LLM,
and parses structured JSON responses. Used offline to pre-score
GRPO training data before training.

Rubric criteria (5 binary, weighted):
  signal_alignment (25%) — action matches technical indicators
  risk_discipline  (20%) — position sizing, loss management
  timing_quality   (20%) — entry/exit relative to momentum
  regime_awareness (15%) — respects market regime context
  reasoning_coherence (20%) — <think> block justifies action
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

CRITERIA_WEIGHTS = {
    "signal": 0.25,
    "risk": 0.20,
    "timing": 0.20,
    "regime": 0.15,
    "reasoning": 0.20,
}

JUDGE_SYSTEM_PROMPT = """\
You are an expert trading evaluator. You assess the quality of stock trading \
decisions made by an AI agent operating in the Indian equity market (NIFTY stocks).

You will be given:
1. The market observation the agent saw (technical indicators, prices, trends)
2. The action the agent took (BUY or SELL)
3. The agent's reasoning (inside <think> tags)

Score the decision on these 5 criteria (each 0 or 1):

1. **signal** — Does the action align with technical signals? \
BUY when RSI < 30 or bullish MACD crossover = good. \
BUY when RSI > 70 or bearish trend = bad.

2. **risk** — Is the trade disciplined? \
Appropriate position sizing, cutting losses on losing positions, \
not overexposing to a single stock = good.

3. **timing** — Is the entry/exit well-timed? \
Buying near support/oversold, selling near resistance/overbought = good. \
Chasing momentum after large moves = bad.

4. **regime** — Does the decision respect the market regime? \
Buying in a bull/sideways market = acceptable. \
Aggressive buying in a strong bear market without reversal signal = bad.

5. **reasoning** — Does the <think> block logically justify the action? \
Clear causal chain from observation to action = good. \
Generic/vague reasoning or reasoning that contradicts the action = bad.

Respond ONLY with valid JSON (no markdown, no extra text):
{"signal": 0 or 1, "risk": 0 or 1, "timing": 0 or 1, "regime": 0 or 1, "reasoning": 0 or 1}\
"""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "Observation: RELIANCE at Rs2450. RSI: 26.3 (oversold). "
            "MACD: bullish crossover. Trend: bearish but weakening. "
            "Bollinger: lower band. Volume: 1.8x spike. Regime: sideways.\n"
            "Action: BUY\n"
            "Reasoning: <think>RSI is 26 (oversold) and MACD just crossed "
            "bullish. Volume spike confirms buying interest. Price at lower "
            "Bollinger band suggests mean reversion likely.</think>"
        ),
    },
    {
        "role": "assistant",
        "content": '{"signal": 1, "risk": 1, "timing": 1, "regime": 1, "reasoning": 1}',
    },
    {
        "role": "user",
        "content": (
            "Observation: TCS at Rs3800. RSI: 42.1 (neutral). "
            "MACD: bearish, below signal. Trend: bearish. "
            "Bollinger: below middle. Volume: 0.7x (low). Regime: mild_bear.\n"
            "Action: SELL\n"
            "Reasoning: <think>The stock looks weak so I will sell.</think>"
        ),
    },
    {
        "role": "assistant",
        "content": '{"signal": 0, "risk": 1, "timing": 0, "regime": 1, "reasoning": 0}',
    },
]


@dataclass(frozen=True)
class JudgeContext:
    """Context for a single trading decision to be judged."""

    observation: str
    action: str
    reasoning: str
    has_position: bool = False
    position_pnl: float | None = None

    def to_user_message(self) -> str:
        parts = [f"Observation: {self.observation[:800]}"]
        parts.append(f"Action: {self.action}")
        if self.has_position and self.position_pnl is not None:
            parts.append(f"Current position P&L: {self.position_pnl:+.1f}%")
        parts.append(f"Reasoning: {self.reasoning[:500]}")
        return "\n".join(parts)


def build_judge_prompt(ctx: JudgeContext) -> list[dict]:
    """Build chat messages for the judge LLM."""
    messages = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}]
    messages.extend(FEW_SHOT_EXAMPLES)
    messages.append({"role": "user", "content": ctx.to_user_message()})
    return messages


@dataclass(frozen=True)
class JudgeScore:
    """Parsed judge response."""

    signal: int
    risk: int
    timing: int
    regime: int
    reasoning: int
    total: float
    raw_response: str = ""

    @property
    def criteria_dict(self) -> dict[str, int]:
        return {
            "signal": self.signal,
            "risk": self.risk,
            "timing": self.timing,
            "regime": self.regime,
            "reasoning": self.reasoning,
        }


def compute_weighted_score(criteria: dict[str, int]) -> float:
    """Compute weighted total from binary criteria."""
    total = 0.0
    for key, weight in CRITERIA_WEIGHTS.items():
        total += criteria.get(key, 0) * weight
    return round(total, 3)


def parse_judge_response(text: str) -> JudgeScore:
    """Parse judge LLM response into a JudgeScore.

    Tries JSON parse first, falls back to regex extraction.
    Returns a default neutral score (0.5) on total failure.
    """
    text = text.strip()

    # Try direct JSON parse
    try:
        data = json.loads(text)
        criteria = {
            k: int(bool(data.get(k, 0)))
            for k in CRITERIA_WEIGHTS
        }
        return JudgeScore(
            **criteria,
            total=compute_weighted_score(criteria),
            raw_response=text,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    # Fallback: extract JSON from markdown or surrounding text
    json_match = re.search(r'\{[^}]+\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            criteria = {
                k: int(bool(data.get(k, 0)))
                for k in CRITERIA_WEIGHTS
            }
            return JudgeScore(
                **criteria,
                total=compute_weighted_score(criteria),
                raw_response=text,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    # Total failure — return neutral
    logger.warning("Failed to parse judge response: %s", text[:200])
    return JudgeScore(
        signal=0, risk=0, timing=0, regime=0, reasoning=0,
        total=0.0, raw_response=text,
    )

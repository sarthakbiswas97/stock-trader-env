"""Tests for LLM-as-Judge infrastructure (no real API calls)."""

from training.judge_prompt import (
    JudgeContext,
    JudgeScore,
    build_judge_prompt,
    compute_weighted_score,
    parse_judge_response,
    CRITERIA_WEIGHTS,
)


class TestJudgeContext:
    def test_to_user_message_basic(self):
        ctx = JudgeContext(
            observation="RELIANCE at Rs2450. RSI: 28.",
            action="BUY",
            reasoning="<think>RSI oversold.</think>",
        )
        msg = ctx.to_user_message()
        assert "RELIANCE" in msg
        assert "BUY" in msg
        assert "<think>" in msg

    def test_to_user_message_with_position(self):
        ctx = JudgeContext(
            observation="TCS at Rs3800.",
            action="SELL",
            reasoning="<think>Taking profit.</think>",
            has_position=True,
            position_pnl=5.2,
        )
        msg = ctx.to_user_message()
        assert "+5.2%" in msg

    def test_truncates_long_observation(self):
        ctx = JudgeContext(
            observation="x" * 2000,
            action="BUY",
            reasoning="<think>test</think>",
        )
        msg = ctx.to_user_message()
        assert len(msg) < 2000


class TestBuildJudgePrompt:
    def test_returns_list_of_dicts(self):
        ctx = JudgeContext(
            observation="test", action="BUY", reasoning="<think>test</think>",
        )
        messages = build_judge_prompt(ctx)
        assert isinstance(messages, list)
        assert all(isinstance(m, dict) for m in messages)

    def test_has_system_and_user(self):
        ctx = JudgeContext(
            observation="test", action="BUY", reasoning="<think>test</think>",
        )
        messages = build_judge_prompt(ctx)
        roles = [m["role"] for m in messages]
        assert roles[0] == "system"
        assert roles[-1] == "user"

    def test_includes_few_shot(self):
        ctx = JudgeContext(
            observation="test", action="SELL", reasoning="<think>test</think>",
        )
        messages = build_judge_prompt(ctx)
        # System + 2 few-shot pairs (user+assistant) + final user = 6
        assert len(messages) == 6


class TestParseJudgeResponse:
    def test_valid_json(self):
        response = '{"signal": 1, "risk": 1, "timing": 0, "regime": 1, "reasoning": 0}'
        score = parse_judge_response(response)
        assert score.signal == 1
        assert score.risk == 1
        assert score.timing == 0
        assert score.regime == 1
        assert score.reasoning == 0
        assert 0 < score.total < 1

    def test_all_ones(self):
        response = '{"signal": 1, "risk": 1, "timing": 1, "regime": 1, "reasoning": 1}'
        score = parse_judge_response(response)
        assert score.total == 1.0

    def test_all_zeros(self):
        response = '{"signal": 0, "risk": 0, "timing": 0, "regime": 0, "reasoning": 0}'
        score = parse_judge_response(response)
        assert score.total == 0.0

    def test_json_in_markdown(self):
        response = '```json\n{"signal": 1, "risk": 0, "timing": 1, "regime": 0, "reasoning": 1}\n```'
        score = parse_judge_response(response)
        assert score.signal == 1
        assert score.timing == 1

    def test_malformed_json_returns_zero(self):
        response = "I think this is a good trade because..."
        score = parse_judge_response(response)
        assert score.total == 0.0

    def test_empty_string(self):
        score = parse_judge_response("")
        assert score.total == 0.0

    def test_partial_json(self):
        response = '{"signal": 1, "risk": 1}'
        score = parse_judge_response(response)
        assert score.signal == 1
        assert score.risk == 1
        assert score.timing == 0  # Missing defaults to 0

    def test_preserves_raw_response(self):
        response = '{"signal": 1, "risk": 0, "timing": 0, "regime": 0, "reasoning": 0}'
        score = parse_judge_response(response)
        assert score.raw_response == response


class TestComputeWeightedScore:
    def test_weights_sum_to_one(self):
        assert abs(sum(CRITERIA_WEIGHTS.values()) - 1.0) < 1e-6

    def test_all_ones_gives_one(self):
        criteria = {k: 1 for k in CRITERIA_WEIGHTS}
        assert compute_weighted_score(criteria) == 1.0

    def test_signal_only(self):
        criteria = {"signal": 1, "risk": 0, "timing": 0, "regime": 0, "reasoning": 0}
        assert compute_weighted_score(criteria) == 0.25


class TestJudgeScore:
    def test_criteria_dict(self):
        score = JudgeScore(
            signal=1, risk=0, timing=1, regime=0, reasoning=1, total=0.65,
        )
        d = score.criteria_dict
        assert d == {"signal": 1, "risk": 0, "timing": 1, "regime": 0, "reasoning": 1}

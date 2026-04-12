"""Tests for grading functions — the score that determines agent quality."""

from server.tasks import (
    grade_single_stock,
    grade_portfolio,
    grade_full_autonomous,
    _clamp_score,
    SCORE_MIN,
    SCORE_MAX,
)


class TestClampScore:
    """Scores must always be in open interval (0, 1) for validator compliance."""

    def test_clamp_below_zero(self):
        assert _clamp_score(-0.5) == SCORE_MIN

    def test_clamp_above_one(self):
        assert _clamp_score(1.5) == SCORE_MAX

    def test_clamp_exactly_zero(self):
        assert _clamp_score(0.0) == SCORE_MIN

    def test_clamp_exactly_one(self):
        assert _clamp_score(1.0) == SCORE_MAX

    def test_normal_value_passes_through(self):
        assert _clamp_score(0.5) == 0.5


class TestGradeSingleStock:
    """Single stock grader: agent return vs buy-and-hold benchmark."""

    def test_lost_money_badly(self):
        # Lost >5% — should score around 0.1
        score = grade_single_stock(94_000, 100_000, 0.05, 5)
        assert score < 0.2

    def test_lost_money_slightly(self):
        # Lost 2% — should score 0.2-0.3
        score = grade_single_stock(98_000, 100_000, 0.05, 5)
        assert 0.15 < score < 0.35

    def test_beat_benchmark(self):
        # Agent made 10%, benchmark made 5% — ratio 2.0, should score high
        score = grade_single_stock(110_000, 100_000, 0.05, 10)
        assert score > 0.7

    def test_matched_benchmark(self):
        # Agent made 5%, benchmark made 5% — ratio 1.0
        score = grade_single_stock(105_000, 100_000, 0.05, 10)
        assert 0.5 < score < 0.8

    def test_benchmark_negative(self):
        # Market went down but agent made money
        score = grade_single_stock(103_000, 100_000, -0.02, 5)
        assert score > 0.5

    def test_score_always_in_bounds(self):
        """No matter the inputs, score must be in (0, 1)."""
        extreme_cases = [
            (200_000, 100_000, 0.01, 100),   # Doubled money
            (1_000, 100_000, -0.5, 0),        # Lost almost everything
            (100_000, 100_000, 0.0, 0),        # Did nothing, market flat
        ]
        for final, initial, bh, trades in extreme_cases:
            score = grade_single_stock(final, initial, bh, trades)
            assert SCORE_MIN <= score <= SCORE_MAX


class TestGradePortfolio:
    """Portfolio grader: 60% Sharpe + 25% discipline + 15% activity."""

    def test_good_risk_adjusted_return(self):
        # Consistent positive daily returns — high Sharpe
        daily_returns = [0.005] * 30  # 0.5% per day, no variance
        score = grade_portfolio(115_000, 100_000, daily_returns, 0, 15)
        assert score > 0.6

    def test_high_risk_violations_penalized(self):
        daily_returns = [0.003] * 30
        clean = grade_portfolio(109_000, 100_000, daily_returns, 0, 15)
        dirty = grade_portfolio(109_000, 100_000, daily_returns, 5, 15)
        assert clean > dirty

    def test_too_passive_penalized(self):
        """1 trade in 30 days should score lower than balanced trading."""
        daily_returns = [0.001] * 30
        passive = grade_portfolio(103_000, 100_000, daily_returns, 0, 1)
        balanced = grade_portfolio(103_000, 100_000, daily_returns, 0, 15)
        assert balanced > passive

    def test_overtrading_penalized(self):
        """200 trades in 30 days should score lower than balanced trading."""
        daily_returns = [0.001] * 30
        overtrade = grade_portfolio(103_000, 100_000, daily_returns, 0, 200)
        balanced = grade_portfolio(103_000, 100_000, daily_returns, 0, 15)
        assert balanced > overtrade

    def test_score_always_in_bounds(self):
        score = grade_portfolio(50_000, 100_000, [-0.02] * 30, 10, 0)
        assert SCORE_MIN <= score <= SCORE_MAX


class TestGradeFullAutonomous:
    """Hard task grader: return + risk-adjusted + regime discipline + risk mgmt."""

    def test_perfect_regime_discipline(self):
        # Respected all gated days
        score = grade_full_autonomous(
            510_000, 500_000, [0.002] * 40,
            risk_violations=0, regime_gated_days=10,
            regime_respected=10, total_trades=20, max_drawdown=0.02,
        )
        assert score > 0.5

    def test_regime_violation_penalized(self):
        # Same as above but violated every regime gate
        good = grade_full_autonomous(
            510_000, 500_000, [0.002] * 40,
            risk_violations=0, regime_gated_days=10,
            regime_respected=10, total_trades=20, max_drawdown=0.02,
        )
        bad = grade_full_autonomous(
            510_000, 500_000, [0.002] * 40,
            risk_violations=0, regime_gated_days=10,
            regime_respected=0, total_trades=20, max_drawdown=0.02,
        )
        assert good > bad

    def test_high_drawdown_penalized(self):
        low_dd = grade_full_autonomous(
            520_000, 500_000, [0.002] * 40,
            risk_violations=0, regime_gated_days=0,
            regime_respected=0, total_trades=20, max_drawdown=0.01,
        )
        high_dd = grade_full_autonomous(
            520_000, 500_000, [0.002] * 40,
            risk_violations=0, regime_gated_days=0,
            regime_respected=0, total_trades=20, max_drawdown=0.15,
        )
        assert low_dd > high_dd

    def test_no_regime_gated_days(self):
        # No regime gate triggered — should get full regime score (1.0)
        score = grade_full_autonomous(
            520_000, 500_000, [0.002] * 40,
            risk_violations=0, regime_gated_days=0,
            regime_respected=0, total_trades=20, max_drawdown=0.02,
        )
        assert score > 0.5

    def test_score_always_in_bounds(self):
        score = grade_full_autonomous(
            300_000, 500_000, [-0.01] * 40,
            risk_violations=20, regime_gated_days=20,
            regime_respected=0, total_trades=0, max_drawdown=0.4,
        )
        assert SCORE_MIN <= score <= SCORE_MAX

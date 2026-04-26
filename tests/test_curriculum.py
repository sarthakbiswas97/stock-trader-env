"""Tests for the adaptive curriculum manager."""

from __future__ import annotations

from server.curriculum import CURRICULUM_ORDER, CurriculumManager


class TestCurriculumPromotion:
    def test_starts_at_first_tier(self):
        cm = CurriculumManager()
        assert cm.current_tier == "single_stock"
        assert cm.tier_index == 0

    def test_promotes_after_consistent_high_scores(self):
        cm = CurriculumManager()
        for _ in range(5):
            cm.record_score(0.65)
        assert cm.current_tier == "single_stock_costs"

    def test_no_promotion_below_threshold(self):
        cm = CurriculumManager()
        for _ in range(10):
            cm.record_score(0.50)
        assert cm.current_tier == "single_stock"

    def test_no_promotion_insufficient_episodes(self):
        cm = CurriculumManager()
        for _ in range(4):
            cm.record_score(0.70)
        assert cm.current_tier == "single_stock"

    def test_full_progression(self):
        cm = CurriculumManager()
        tiers_visited = [cm.current_tier]

        # single_stock → single_stock_costs (need 0.60+)
        for _ in range(5):
            cm.record_score(0.65)
        tiers_visited.append(cm.current_tier)

        # single_stock_costs → multi_stock_3 (need 0.55+)
        for _ in range(5):
            cm.record_score(0.60)
        tiers_visited.append(cm.current_tier)

        # multi_stock_3 → portfolio (need 0.50+)
        for _ in range(5):
            cm.record_score(0.55)
        tiers_visited.append(cm.current_tier)

        # portfolio → full_autonomous (need 0.50+)
        for _ in range(5):
            cm.record_score(0.55)
        tiers_visited.append(cm.current_tier)

        assert tiers_visited == CURRICULUM_ORDER

    def test_no_promotion_from_final_tier(self):
        cm = CurriculumManager()
        cm.current_tier = "full_autonomous"
        for _ in range(10):
            cm.record_score(0.90)
        assert cm.current_tier == "full_autonomous"


class TestCurriculumDemotion:
    def test_demotes_on_low_scores(self):
        cm = CurriculumManager()
        cm.current_tier = "single_stock_costs"
        for _ in range(3):
            cm.record_score(0.20)
        assert cm.current_tier == "single_stock"

    def test_no_demotion_from_first_tier(self):
        cm = CurriculumManager()
        for _ in range(10):
            cm.record_score(0.10)
        assert cm.current_tier == "single_stock"

    def test_no_demotion_above_threshold(self):
        cm = CurriculumManager()
        cm.current_tier = "portfolio"
        for _ in range(10):
            cm.record_score(0.35)
        assert cm.current_tier == "portfolio"


class TestCurriculumTracking:
    def test_episode_count(self):
        cm = CurriculumManager()
        for _ in range(7):
            cm.record_score(0.50)
        assert cm.episode_count == 7

    def test_transitions_logged(self):
        cm = CurriculumManager()
        for _ in range(5):
            cm.record_score(0.65)
        assert len(cm.transitions) == 1
        assert cm.transitions[0].direction == "promotion"
        assert cm.transitions[0].from_tier == "single_stock"
        assert cm.transitions[0].to_tier == "single_stock_costs"

    def test_summary(self):
        cm = CurriculumManager()
        for _ in range(5):
            cm.record_score(0.65)
        summary = cm.summary()
        assert summary["current_tier"] == "single_stock_costs"
        assert summary["promotions"] == 1
        assert summary["demotions"] == 0

    def test_record_score_returns_new_tier_on_transition(self):
        cm = CurriculumManager()
        for _ in range(4):
            result = cm.record_score(0.65)
            assert result is None
        result = cm.record_score(0.65)
        assert result == "single_stock_costs"

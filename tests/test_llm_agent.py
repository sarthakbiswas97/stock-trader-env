"""Tests for LLM agent — parsing and prompt construction."""

from baselines.llm_agent import parse_action, SYSTEM_PROMPT, USER_TEMPLATE


class TestParseAction:
    """Extract trading actions from model responses."""

    def test_simple_hold(self):
        assert parse_action("HOLD") == "HOLD"

    def test_simple_buy(self):
        assert parse_action("BUY RELIANCE") == "BUY RELIANCE"

    def test_buy_with_fraction(self):
        assert parse_action("BUY RELIANCE 0.5") == "BUY RELIANCE 0.5"

    def test_action_after_reasoning(self):
        response = """<think>RSI is low, good entry.</think>
BUY RELIANCE 0.3"""
        assert parse_action(response) == "BUY RELIANCE 0.3"

    def test_action_buried_in_text(self):
        response = """Let me analyze the market...
The RSI shows oversold conditions.
Based on this analysis:
SELL TCS"""
        assert parse_action(response) == "SELL TCS"

    def test_no_valid_action_defaults_hold(self):
        assert parse_action("I think the market is uncertain") == "HOLD"

    def test_empty_response_defaults_hold(self):
        assert parse_action("") == "HOLD"

    def test_lowercase_action(self):
        # parse_action uppercases, so lowercase input should work
        assert parse_action("buy reliance") == "BUY RELIANCE"

    def test_action_with_trailing_text(self):
        response = "BUY RELIANCE 0.5 because RSI is low"
        assert parse_action(response) == "BUY RELIANCE 0.5"

    def test_bare_buy_single_stock(self):
        assert parse_action("BUY") == "BUY"

    def test_bare_sell(self):
        assert parse_action("SELL") == "SELL"


class TestPromptConstruction:
    """Verify prompt templates are well-formed."""

    def test_system_prompt_has_rules(self):
        assert "HOLD" in SYSTEM_PROMPT
        assert "BUY" in SYSTEM_PROMPT
        assert "<think>" in SYSTEM_PROMPT

    def test_user_template_has_placeholder(self):
        assert "{observation}" in USER_TEMPLATE
        filled = USER_TEMPLATE.format(observation="Day 1 of 20...")
        assert "Day 1 of 20..." in filled

"""LLM-based trading agent — uses a language model to reason about market state.

Supports two backends:
    - "local": Loads model via transformers (for small models or GPU machines)
    - "api": Uses HuggingFace Inference API (for large models, free tier)

The agent receives the market observation text, wraps it in a trading prompt,
and parses the model's response into a valid action.
"""

from __future__ import annotations

import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert stock trader operating in the Indian equity market.
You receive daily market observations with technical indicators and must decide on a trading action.

Rules:
- Respond with EXACTLY one action on the last line
- Valid actions: HOLD, BUY, SELL, BUY <SYMBOL>, SELL <SYMBOL>, BUY <SYMBOL> <FRACTION>
- Before your action, briefly explain your reasoning in 1-2 sentences inside <think> tags
- Use the FRACTION parameter (0.0-1.0) to size positions based on conviction and risk

Risk Management:
- Check your portfolio drawdown and trading capacity before entering trades
- If on a losing streak (3+ consecutive losses), reduce exposure or wait
- Positions held longer than 5 days incur holding costs — plan your exits
- When drawdown exceeds 3%, your trading capacity is reduced — size down accordingly

Example response:
<think>RSI is 25 (oversold) and MACD just turned bullish. Volume is spiking. Good entry point.</think>
BUY RELIANCE 0.5"""

USER_TEMPLATE = """Here is today's market data:

{observation}

What is your trading action?"""


def parse_action(response: str) -> str:
    """Extract the trading action from model response.

    Looks for the last line that starts with BUY/SELL/HOLD.
    Falls back to HOLD if no valid action found.
    """
    lines = response.strip().split("\n")
    # Search from bottom up for a valid action line
    for line in reversed(lines):
        line = line.strip().upper()
        if line.startswith(("BUY", "SELL", "HOLD")):
            # Clean up any trailing punctuation or extra text
            match = re.match(r"((?:BUY|SELL|HOLD)(?:\s+[A-Z]{2,20})?(?:\s+\d*\.?\d+)?)", line)
            if match:
                return match.group(1)
    return "HOLD"


def create_local_agent(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str = "cpu",
    max_new_tokens: int = 200,
) -> Callable[[str], str]:
    """Create an agent that runs a local HuggingFace model.

    Args:
        model_name: HuggingFace model ID.
        device: "cpu", "mps", or "cuda".
        max_new_tokens: Max tokens to generate per action.

    Returns:
        Agent function: observation -> action string.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model %s on %s...", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device,
    )
    model.eval()
    logger.info("Model loaded.")

    def agent_fn(observation: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(observation=observation)},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        action = parse_action(response)
        logger.debug("LLM response: %s -> action: %s", response[:100], action)
        return action

    return agent_fn


def create_api_agent(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    api_token: str | None = None,
    max_new_tokens: int = 1024,
) -> Callable[[str], str]:
    """Create an agent that uses HuggingFace Inference API.

    Args:
        model_name: HuggingFace model ID (must be available on Inference API).
        api_token: HF token. If None, uses HF_TOKEN env var.
        max_new_tokens: Max tokens to generate.

    Returns:
        Agent function: observation -> action string.
    """
    import os
    from huggingface_hub import InferenceClient

    token = api_token or os.environ.get("HF_TOKEN")
    client = InferenceClient(model=model_name, token=token)

    def agent_fn(observation: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(observation=observation)},
        ]
        try:
            response = client.chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.7,
            )
            msg = response.choices[0].message
            # DeepSeek-R1 puts reasoning in reasoning_content, answer in content
            text = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None) or ""
            if not text and reasoning:
                # Model only produced reasoning, no final answer — parse action from reasoning
                text = reasoning
            action = parse_action(text)
            logger.debug("API response: %s -> action: %s", text[:100], action)
            return action
        except Exception as e:
            logger.warning("API call failed: %s. Defaulting to HOLD.", e)
            return "HOLD"

    return agent_fn

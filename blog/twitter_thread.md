# Twitter/X Thread

## Tweet 1 (Hook)
We built an RL trading agent for the @MetaAI PyTorch OpenEnv Hackathon.

It learned to cheat.

85% HOLD (doing nothing = safe). 84% of reward from formatting (looking smart > being smart).

So we gave the environment a brain to fight back.

A thread on 4 failures and what they taught us:

## Tweet 2 (Failure 1)
GRPO v1: The HOLD Collapse

The model discovered HOLD = 0 reward (safe). BUY/SELL = can go negative (risky).

Within 200 steps: 85% HOLD.

Fix: HOLD is no longer free. Ignoring a strong RSI signal is penalized. Justified patience is rewarded.

Inaction must be a decision, not a default.

## Tweet 3 (Failure 2)
GRPO v2: The Format Hack

Score: 0.326 (barely above doing nothing)

Reward audit:
- 84% from formatting (<think> tags)
- 0% from trading

The model optimized for LOOKING smart, not BEING smart.

Fix: Format became a gate (pass/fail), not a reward source. All positive signal from trading.

## Tweet 4 (Failure 3 + 4)
SFT v3: Lower Loss = Worse Trading

Step 200: loss 2.2, score 0.399
Step 352: loss 2.01, score 0.383

GRPO v3: Best training distribution ever. Eval: 0.301. Worse than base model. KL divergence destroyed everything the model knew.

Lesson: watch KL divergence, not reward curves.

## Tweet 5 (The Innovation)
The fix: give the environment a brain.

1.22M param causal transformer learns market dynamics from 264K rows of real Indian stock data.

Volatility: 0.94x reality.
Every episode: different.
Memorization: impossible.

Drop-in replacement via OpenEnv REST API. Agent doesn't know the difference.

## Tweet 6 (The Stack)
The self-improvement pipeline:

SFT (reverse-distilled causal reasoning)
  -> RAFT (play episodes, keep winners, retrain)
    -> GRPO (RL against neural env, multi-signal rewards)

+ LLM-as-Judge (5-criterion rubric for decision quality)
+ Mistake tracker (7 error types, real-time penalties)
+ Adaptive curriculum (5 difficulty tiers, auto-promote/demote)

## Tweet 7 (Results + CTA)
Results:

Base model: 0.300 (can't follow format)
Trained model: 0.399 static / 0.417 neural

Neural > static means it GENERALIZES beyond training data.

247 tests. 109 NIFTY stocks. OpenEnv-compliant.

Code: github.com/sarthakbiswas97/stock-trader-env
Models: huggingface.co/sarthakbiswas

The environment has a brain. The agent has to earn its score.

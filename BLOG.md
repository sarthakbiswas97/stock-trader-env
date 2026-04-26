# 11 Models, Multiple Crashes, and a Neural Environment That Fights Back

In trading, past patterns don't guarantee future results. I learned this the hard way not from the market, but from training RL agents that kept finding ways to cheat.

This project started as a stock trading environment for the [Meta PyTorch OpenEnv Hackathon](https://pytorch.org/blog/openenv/). But along the way it became something bigger: a case study in how the environment you train against shapes what the model actually learns. The same idea applies to code generation, scientific reasoning, planning anywhere a model needs to make decisions under uncertainty. The environment is the teacher. If the teacher is predictable, the student learns shortcuts.

You can try the live environment here: [stock-trader-env on HF Spaces](https://huggingface.co/spaces/sarthakbiswas/stock-trader-env). The full code is on [GitHub](https://github.com/sarthakbiswas97/stock-trader-env).

---

## Before the hackathon

I had already built a trading agent before this hackathon started. It was a rule-based system using the Kite Connect API, the kind of thing a retail trader would build. If RSI drops below 30, buy. If it crosses 70, sell. Simple rules, and they worked okay on backtests. But backtests are just replaying history. The agent wasn't learning anything, it was just following a script I wrote.

When the hackathon announcement came, the problem statement was about building RL environments. I had never trained a model using GRPO. Never built an RL environment. Never even used TRL. Everything about this was new to me, and honestly that's what made it interesting.

---

## Building the environment

The environment replays real daily price data from Indian equity markets. I collected OHLCV data for [109 NIFTY stocks spanning 2015-2026](https://huggingface.co/datasets/sarthakbiswas/stock-trader-market-data) about 264,000 rows covering bull, bear and sideways market. The environment is OpenEnv-compliant (REST + WebSocket API, typed Pydantic models, seed-reproducible episodes).

What the agent sees each day is a text observation not raw numbers. The feature engine converts OHLCV data into human-readable technical analysis: RSI, MACD, Bollinger Band position, trend direction, momentum, volatility, volume spikes etc. This makes it natural for LLMs to reason about. The agent reads text like "RSI: 28.3 (oversold), MACD: bullish crossover, Volume: 1.4x average" and decides BUY, SELL, or HOLD.

Three difficulty levels with increasing constraints:
- Single stock, no costs (learn basic timing)
- Portfolio with transaction costs and position limits (learn risk management)
- Full autonomous with regime gates (learn when NOT to trade)

---

## SFT v1: The data imbalance disaster (Score: 0.300)

My first attempt at fine-tuning was straightforward. I collected 7,000 training examples from a rule-based agent and ran SFT on [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) using Unsloth with QLoRA.

The model scored 0.300. That's the HOLD floor, the exact score you get by doing absolutely nothing.

The problem was obvious: 91% of my training data was HOLD actions. The model learned the data distribution perfectly. 91% of its outputs were HOLD. Garbage in, garbage out.

## SFT v2: Template reasoning (Score: 0.398)

I rebalanced the dataset to 35% BUY / 25% SELL / 40% HOLD and scaled to [40,000 examples across 97 stocks](https://huggingface.co/datasets/sarthakbiswas/stock-trader-sft-v2). Added template reasoning each example had a `<think>` block explaining the decision based on indicators.

Score jumped to 0.398. The [SFT v2 model](https://huggingface.co/sarthakbiswas/stock-trader-sft-v2-model) could follow the format, real indicators, and actually trade. Two out of five test episodes scored 0.6 (beating buy-and-hold). First real sign of good trading agent.

But the reasoning was bit not that good. Every BUY sounded the same. The model learned a template, not actual market related thinking.

---

## GRPO v1: My first RL run ever (Score: ~0.300)

GRPO generates multiple candidate responses, scores them, and reinforces the better ones. I figured: the environment gives rewards, GRPO uses rewards, this should work.

BUT it did not work.

Within 200 training steps, 85% of all actions were HOLD. The model discovered that HOLD gets zero reward (safe), while BUY and SELL can go negative (risky). A perfectly rational agent that learned the worst possible lesson: doing nothing is always safe.

When every candidate outputs the same HOLD action, GRPO gets zero learning signal. No variance in the group means no gradient. Training was technically running but nothing was being learned.

## GRPO v2: The format hack (Score: 0.326)

I added four reward functions: format compliance, reasoning quality, trading P&L, and prediction accuracy. Surely more signal would help.

The [GRPO v2 model](https://huggingface.co/sarthakbiswas/stock-trader-grpo-v2-model) scored 0.326. Barely above the HOLD floor.

When I dug into the reward breakdown: 84% of total reward came from formatting. The model had figured out that producing clean `<think>...</think>` tags and a well-formatted action line was worth way more than making a good trading decision. It optimized for looking smart instead of being smart.

The model will optimize exactly what you measure. If your reward function has a loophole, the model finds it before you do.

## GRPO v2.1: Reward alignment (Score: 0.395)

I stripped it down to two reward functions. Format became a gate: 0 if valid, -1 if invalid. Not a reward source format compliance is expected, not rewarded. All positive reward came from actual trading performance: 30% step-level P&L + 70% episode-level return.

The [GRPO v2.1 model](https://huggingface.co/sarthakbiswas/stock-trader-grpo-v2.1-model) scored 0.395. 10 out of 20 test episodes scored 0.6 (the model could trade profitably). But 2 episodes scored 0.1 (catastrophic losses). it learned a strategy but not when to apply it.

Also, KL divergence climbed to 3.9 during training. The model was drifting far from its SFT base forgetting base knowledge while chasing RL reward.

## GRPO v3: The knowledge catastrophe (Score: 0.301)

This was the one that hurt. I applied ideas from Pikus et al.'s "Hard Examples Are All You Need" trained on the model's own trajectories, filtered by difficulty, used G=8 generations for better variance. The training metrics looked fantastic: 59% HOLD, 25% BUY, 16% SELL. Best action distribution I'd ever seen. Reward curves were climbing.

Eval score: 0.301. Worse than the untrained base model.

KL divergence had climbed to 4.2. The model forgot how to trade. Strong training metrics, destroyed knowledge. The dashboard looked great while the patient was dying.

This was the lowest point. Multiple GRPO attempts, and none of them beat basic SFT. I was starting to wonder if RL was even the right approach for this problem.

---

## Starting over: Distilled reasoning

I went back to SFT but changed everything about the data. Instead of template reasoning ("RSI below 30, therefore BUY"), I used GPT-4o-mini to generate real causal explanations for each trading decision. Reverse distillation, start with the correct action, generate the reasoning backwards.

[10,000 examples](https://huggingface.co/datasets/sarthakbiswas/stock-trader-sft-v3) instead of 40,000. Quality over quantity. Each example explained conflicting signals, risk factors, regime context. Not "RSI is low so buy" but "RSI at 28 suggests oversold conditions, but MACD hasn't confirmed the reversal and volume is below average waiting for confirmation before entering."

The training config was conservative on purpose: lr=5e-6 (4x lower than v2), LoRA r=16 (4x smaller), targeting only attention layers (not FFN). The goal was to add reasoning capability without destroying what the base model already knew.

The [SFT v3 model](https://huggingface.co/sarthakbiswas/stock-trader-sft-v3-model) at step 200 scored 0.399 on static replay and 0.417 on the neural environment. But here's the thing: step 352 had lower training loss and scored 0.383. Lower loss, worse trading. I almost shipped the wrong checkpoint. After this, I always picked checkpoints by task score, never by training loss.

---

## The neural environment: Why static replay is broken

Every experiment so far used static replay. The environment plays back the same historical CSV data. Same price sequences, same indicators, same patterns. The agent can memorize "buy on day 7 of this RELIANCE sequence" and score well without learning anything about trading.

Real markets are stochastic. Tomorrow won't look like any day in the training data. An agent that memorized price sequences has learned nothing transferable.

So I built a neural world model. It's a causal transformer, 1.22M parameters, 4 layers, mixture density output head, trained on the same [264,000 rows of real market data](https://huggingface.co/datasets/sarthakbiswas/stock-trader-market-data). It generates synthetic but realistic OHLCV price data. Volatility calibration: 0.94x real markets. Trained in 7 minutes on a single GPU.

The neural simulator is a drop-in replacement for CSV replay(the static one had earlier). The agent sees the exact same observation format RSI, MACD, Bollinger Bands, all computed from the generated prices. It can't tell if the data is real or synthetic. But every episode is different. Every seed generates a unique price trajectory.This time agent can't memorize to score high.

What the environment provides as learning signals:
- **Signal-based HOLD evaluation**: HOLD is not free. If RSI is below 25 and the agent HOLDs, that's a missed opportunity and it gets penalized. If indicators are neutral and the agent HOLDs, that's justified patience and it gets a small positive signal. The agent has to justify not trading.
- **Mistake tracking**: Seven specific error types detected in real-time: regime violations, overbought buys, oversold sells, position limit breaches, loss holds, missed opportunities. Each feeds directly into the step reward.
- **Drawdown-based capacity scaling**: As the portfolio draws down, trading capacity shrinks from 100% to 25%. The agent learns to trade smaller when it's losing.
- **Position holding costs**: Holding a position beyond 5 days incurs a daily cost. Forces the agent to think about exit timing, not just entry.

The static vs neural comparison tells the story. SFT v3 scored 0.399 on static but 0.417 on neural. The neural environment surfaces patterns that static replay hides, because every episode is a new scenario the model must reason from scratch.

---

## The breakthrough: GRPO against neural env (Score: 0.537)

Everything before this was setup. SFT v3 gave me a model that could trade. The neural environment gave me an environment that couldn't be gamed. Now I connected them.

I collected 1,000 prompts from 50 episodes against the neural environment (mean score 0.395, 36% of episodes above 0.5). Then ran GRPO with settings designed to avoid every previous failure:
- Started from SFT v3 (best model, not from a failed GRPO checkpoint)
- 300 steps (not 1000, shorter means less KL drift)
- beta=0.05 (not 0.01, stronger constraint against forgetting)
- Format as gate only (not reward source)
- Neural env prompts (can't memorize)

The [GRPO neural model](https://huggingface.co/sarthakbiswas/stock-trader-grpo-neural-model) scored 0.537 on neural environment and 0.470 on static replay ([eval results](https://github.com/sarthakbiswas97/stock-trader-env/blob/main/results/grpo_neural_eval.json) in repo). KL stayed under 0.35 the entire run. No catastrophe. No collapse.

That's **79%** improvement over the base model on neural env. For the first time, RL actually helped instead of hurting.

Why it worked when previous GRPO attempts failed:
1. Better starting point (SFT v3, not SFT v2)
2. Better reward (aligned with what eval actually measures)
3. Better environment (stochastic, agent can't memorize now)
4. Better KL management (beta=0.05, 300 steps)

---

## Pushing further: Improved environment

After the breakthrough I improved the environment itself. Added drawdown pressure, position holding costs, losing streak penalties. The idea: make the env harder, force the model to develop more sophisticated strategies.

[GRPO v3.1](https://huggingface.co/sarthakbiswas/stock-trader-grpo-v3.1-model) scored 0.418 on static replay. [GRPO v3.2](https://huggingface.co/sarthakbiswas/stock-trader-grpo-v3.2-model) trained for 500 steps with checkpoints every 100, checkpoint-400 scored 0.485 on neural env, but step-500 collapsed to 0.282. Same pattern as SFT v3: peak performance isn't at the end of training.

HOLD percentage dropped from 95% to 85% over v3.2 training. The model was learning to trade more actively instead of defaulting to inaction.

GRPO v3.3 ran with beta=0.02 (allowing more exploration), lr=2e-7, 150 steps. The [final checkpoint](https://huggingface.co/sarthakbiswas/stock-trader-grpo-v3.3-model) scored 0.416 on neural env. Didn't beat v3.0's 0.537, the lower beta and shorter training wasn't enough to converge. But every run adds to what I know about this system.

---

## The full scoreboard

Every model I trained, in order. The failures matter as much as the successes.

| # | Model | Static | Neural | What happened |
|---|-------|--------|--------|---------------|
| 1 | [DeepSeek 7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) (base) | 0.300 | 0.298 | Can't follow format, defaults to HOLD |
| 2 | SFT v1 | 0.300 | - | 91% HOLD, data imbalance |
| 3 | [SFT v2](https://huggingface.co/sarthakbiswas/stock-trader-sft-v2-model) | 0.398 | - | Template reasoning, first active trading |
| 4 | GRPO v1 | ~0.300 | - | Collapsed to HOLD (zero-cost inaction) |
| 5 | [GRPO v2](https://huggingface.co/sarthakbiswas/stock-trader-grpo-v2-model) | 0.326 | - | 84% reward from formatting |
| 6 | [GRPO v2.1](https://huggingface.co/sarthakbiswas/stock-trader-grpo-v2.1-model) | 0.395 | - | Reward-aligned, bimodal (10/20 at 0.6) |
| 7 | GRPO v3 | 0.301 | - | KL catastrophe (4.2), forgot everything |
| 8 | [SFT v3](https://huggingface.co/sarthakbiswas/stock-trader-sft-v3-model) step-352 | 0.383 | - | Overtrained, lower loss worse score |
| 9 | [SFT v3](https://huggingface.co/sarthakbiswas/stock-trader-sft-v3-model) step-200 | 0.399 | 0.417 | Distilled reasoning, best SFT |
| 10 | RAFT (from base) | 0.300 | 0.300 | Wrong starting point, pure HOLD |
| 11 | RAFT v2 (from SFT v3) | 0.360 | 0.399 | 640 winners, slight degradation |
| 12 | [GRPO neural](https://huggingface.co/sarthakbiswas/stock-trader-grpo-neural-model) | 0.470 | **0.537** | **Best model, RL against neural env** |
| 13 | [GRPO v3.1](https://huggingface.co/sarthakbiswas/stock-trader-grpo-v3.1-model) | 0.418 | 0.310 | Improved env (harder rewards) |
| 14 | [GRPO v3.2](https://huggingface.co/sarthakbiswas/stock-trader-grpo-v3.2-model) ckpt-400 | - | 0.485 | Improved env, HOLD% 95->85% |
| 15 | [GRPO v3.3](https://huggingface.co/sarthakbiswas/stock-trader-grpo-v3.3-model) | - | 0.416 | Lower beta, shorter run, didn't converge |

### Training curves from real runs

![Training curves: learning curve, SFT loss, GRPO KL divergence, trading reward](https://raw.githubusercontent.com/sarthakbiswas97/stock-trader-env/v3/world-model/results/training_curves_final.png)

*Top-left: Score progression across training stages. Top-right: SFT loss (best checkpoint at step 200). Bottom-left: GRPO KL stayed under 0.35 (previous v3 hit 4.2). Bottom-right: Trading reward over 300 steps.*

Training logs and eval JSONs are in the [results/](https://github.com/sarthakbiswas97/stock-trader-env/tree/main/results) directory.

---

## The bigger picture

This project taught me that the environment matters as much as the model. Maybe more.

A static environment teaches memorization. A neural environment teaches generalization. The same 7B model went from 0.300 (useless) to 0.537 (actually trading) not because the model architecture changed, but because the environment and rewards evolved alongside it.

The same approach works beyond trading. Any domain where you can define a verifiable outcome, code that passes tests, proofs that are valid, plans that succeed, you can build a neural environment that generates novel scenarios, wire it to GRPO, and iterate. The environment is the verifier. GRPO is the trainer. Connect them and keep improving both.

For me personally, this hackathon was a crash course in RL. Few weeks ago I had never trained a model with GRPO. Now I've done it 11 times, crashed multiple of them, and learned more from the crashes than from the successes. Every failure, the HOLD collapse, the format hack, the KL catastrophe taught me something that reading papers alone never made concrete.

The reward function is not just a technical detail. It's the most important design decision in the entire pipeline. Get it wrong and your model will find a way to exploit it. Get it right and the model will surprise you with what it learns.

---

## What I'd do differently

- Start with reward alignment from day 1. I wasted three GRPO runs before I realized format compliance shouldn't be a reward source.
- Monitor KL divergence from the first run. Would have caught the v3 catastrophe 200 steps earlier.
- Quality over quantity in training data. 10K distilled examples beat 40K templates. The LIMA paper was right.

---

## Resources

- **Live environment**: [HF Space](https://huggingface.co/spaces/sarthakbiswas/stock-trader-env)
- **Code**: [GitHub](https://github.com/sarthakbiswas97/stock-trader-env)
- **Best model (GRPO neural, 0.537)**: [HF Hub](https://huggingface.co/sarthakbiswas/stock-trader-grpo-neural-model)
- **Market data (264K rows, 109 stocks)**: [HF Dataset](https://huggingface.co/datasets/sarthakbiswas/stock-trader-market-data)
- **SFT v3 training data (10K distilled)**: [HF Dataset](https://huggingface.co/datasets/sarthakbiswas/stock-trader-sft-v3)
- **Training logs and curves**: [results/](https://github.com/sarthakbiswas97/stock-trader-env/tree/main/results)
- **Training notebook**: [Colab](https://colab.research.google.com/)

Built by Sarthak Biswas for the Meta PyTorch OpenEnv Hackathon, April 2026.

The environment has a brain. The agent has to earn its score.

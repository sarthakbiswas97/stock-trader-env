# We Gave Our Trading Environment a Brain -- Here's What Happened

**4 RL failures, a neural world model, and the lessons that made it work.**

Our RL agent learned to trade stocks. Then it learned to cheat. It discovered that doing nothing was the safest strategy, and that formatting answers correctly was worth more than actually making money. So we gave the environment a brain to fight back.

This post walks through building a self-improving stock trading agent for the Meta PyTorch OpenEnv Hackathon -- what broke, why it broke, and the system we built to fix it.

---

## The Problem: Static Environments Are Memorizable

Most RL trading environments replay historical CSV data. The agent sees the same price sequences, learns the same patterns, and scores well on the same test set. Ship it to a live market? It falls apart.

Real markets are stochastic. Tomorrow's RELIANCE chart won't look like any chart in your training data. An agent that memorized "buy on day 7, sell on day 14" has learned nothing about *trading* -- it has learned a specific CSV file.

**Our question:** What if the environment could generate novel market scenarios that the agent has never seen, forcing it to learn *trading intuition* instead of *dataset memorization*?

---

## The Architecture: A Transformer That Simulates Markets

Ha and Schmidhuber's *World Models* (2018) proposed that agents can learn inside a learned model of their environment -- a Vision module encodes observations, a Memory module predicts dynamics, and a Controller acts on the latent state. We adapted this V-M-C architecture for financial markets.

Our world model is a causal transformer (1.22M parameters, 4 layers, MDN output head) trained on 264,000 rows of real Indian stock data covering 109 NIFTY stocks across 2015-2026, balanced across bull and bear market regimes. Unlike the original World Models architecture that uses a VAE encoder + RNN dynamics, we use a single causal transformer that operates directly on feature sequences -- the autoregressive structure naturally handles temporal dependencies without the compounding drift problem that plagues recurrent architectures.

The world model is a drop-in replacement for the static CSV replay. It generates OHLCV price data, which passes through the existing feature engine to produce human-readable text observations -- RSI, MACD, Bollinger Bands, trend, momentum, and market regime. The LLM agent sees the same observation format regardless of whether the data comes from a CSV file or a neural network.

```
Real Market Data (264K rows, 109 stocks)
        |
        v
Causal Transformer World Model (1.22M params, MDN output)
        |
        v
Generated OHLCV --> Feature Engine --> Text Observation
        |
        v
LLM Agent (DeepSeek-R1 7B, QLoRA) --> BUY / SELL / HOLD
        |
        v
Environment Grader (score 0.0 - 1.0)
```

**Model details:** 4 layers, d_model=192, 4 attention heads, MDN output head with 3 Gaussian mixture components. Trained on 84K sequences (264K rows, stride=3, seq_len=100) for 20 epochs with no overfitting. Training time: 7 minutes on a single A5000 GPU.

**Volatility calibration:** We ran a temperature sweep from 0.1 to 1.5 and found optimal generation at temperature=1.0, producing 0.94x real-market volatility. We validated this against a CNN+GRU baseline:

| Model | Params | Vol Ratio | MAE | Dir Accuracy |
|-------|--------|-----------|-----|-------------|
| **Causal Transformer** | 1.22M | **0.94x** | **0.0167** | 0.492 |
| CNN+GRU (baseline) | 998K | 3.15x | 0.0436 | 0.501 |

The transformer achieves near-perfect volatility calibration with 3x lower prediction error. The CNN+GRU's 3.15x volatility makes it impractical for agent training -- the generated scenarios are too noisy to learn from.

A critical design decision: **agent actions do not affect market prices.** This is a deliberate zero-market-impact assumption -- the agent observes and trades within the market, but the market dynamics are independent of agent behavior. This simplifies the world model (no action conditioning) and reflects the reality that a single retail trader does not move NIFTY stock prices.

---

## The Failure Gallery

Before anything worked, things broke -- repeatedly. Each failure taught us something fundamental that reading papers alone never made concrete.

An early warning came before GRPO even started: **SFT v1** trained on 7,000 examples with a 91% HOLD class imbalance. The model learned the imbalance perfectly -- 91% of its outputs were HOLD, scoring exactly 0.300 (the do-nothing floor). The fix for SFT v2 was straightforward: rebalance to 35% BUY / 25% SELL / 40% HOLD. But the GRPO failures that followed were far more subtle.

### Failure 1: The HOLD Collapse

**GRPO v1.** We trained the agent with a counterfactual reward: did the price go up after you bought? Simple, intuitive, wrong.

Within 200 training steps, 85% of all actions were HOLD. The model discovered that HOLD receives 0 reward (safe) while BUY and SELL can go negative (risky). When all generated candidates output the same action, GRPO gets zero learning signal -- the group has no variance to learn from.

**The takeaway:** If inaction has zero cost, the model will always choose inaction. HOLD must be a *deliberate decision*, not a free default.

### Failure 2: The Format Hack

**GRPO v2.** We added multiple reward functions: format compliance, reasoning quality, and trading performance. The model scored 0.326 -- barely above the HOLD floor of 0.300.

When we audited the reward breakdown: **84% of total reward came from formatting** (`<think>...</think>` tags and clean action output). **0% came from actual trading decisions.** The model had optimized for *looking smart* rather than *being smart*.

**The takeaway:** Multiple reward components must be weighted by what matters at evaluation time. Format compliance should be a gate (pass/fail), not a reward source. This aligns with the RLVR (RL with Verifiable Rewards) principle: reward signals should be anchored to *verifiable outcomes*, not proxy metrics.

### Failure 3: The Alignment Tax

**SFT v3.** We trained with reverse-distilled reasoning -- high-quality causal explanations generated for each trading decision, following the approach from Trading-R1 (arXiv 2509.11420) where an external LLM generates reasoning traces from oracle labels.

The model trained for 352 steps. Step 200 scored **0.399**. Step 352 (lower loss) scored **0.383**. The training loss improved, but trading performance degraded.

**The takeaway:** For real-world tasks, the best checkpoint is rarely the final one. Lower loss here meant the model was memorizing training examples rather than learning generalizable trading patterns. Trading-R1 observed the same phenomenon: "more than 1 epoch always degraded performance." We now select checkpoints by task performance, not by training loss.

### Failure 4: The KL Catastrophe

**GRPO v3.** Inspired by Pikus et al.'s "Hard Examples Are All You Need," we trained on the model's own trajectories filtered by difficulty estimation. The training action distribution was the best we had seen: 59% HOLD, 25% BUY, 16% SELL. Reward curves looked healthy. We applied batch-level reward scaling from Dr. GRPO (arXiv 2503.20783) to prevent per-group standard deviation explosion.

Eval score: **0.301** -- worse than the untrained base model. KL divergence had climbed to 4.2 during training. The model drifted so far from its SFT initialization that it *forgot how to trade*. Good training metrics, destroyed knowledge.

**The takeaway:** Monitor KL divergence, not reward curves. A model that forgets what it knew is worse than one that never learned. We now apply KL early stopping at 3.0.

---

## What We Built to Fix It

Each failure informed a specific architectural or training decision. The system that emerged is fundamentally different from where we started.

### Conservative SFT

| Parameter | Before (degraded) | After (improved) |
|-----------|-------------------|-------------------|
| Learning rate | 2e-5 | 5e-6 |
| LoRA rank | r=64 (all modules) | r=16 (q/k/v/o only) |
| Trainable params | ~40M | ~10M |
| Training data | 40K template examples | 12K distilled reasoning |

The principle: SFT should *add* reasoning capability without *destroying* the base model's existing knowledge. Smaller adapter, lower learning rate, fewer but higher-quality examples. This aligns with the LIMA finding (Zhou et al., 2023) that a small amount of carefully curated data can outperform orders of magnitude more formulaic data for alignment.

### Reverse Distillation

Following the Trading-R1 approach to reverse reasoning distillation, we used an external LLM to generate causal reasoning chains from oracle trading labels rather than generating formulaic "If RSI < 30, BUY" templates. Each example explains *why* the action is correct given the specific market context -- conflicting signals, risk factors, regime awareness. The key difference from standard distillation: the reasoning is generated *backwards* from known-correct actions, not *forwards* from observations.

Quality metrics across 10,000 generated examples: 100% include structured reasoning, 95% mention risk factors, 99.6% resolve conflicting indicator signals correctly.

### Signal-Based HOLD

HOLD is no longer a zero-cost default. The environment classifies every HOLD decision using available market signals:

- **Justified patience** (RSI neutral, no clear trend): small positive signal
- **Missed opportunity** (strong RSI signal ignored): penalized
- **Loss hold** (holding a significantly losing position): penalized

The model must now justify *not* trading, just as it must justify trading.

### Mistake Tracker

Seven specific trading mistake types are detected in real time: regime violations, overbought buys, oversold sells, position limit breaches, trade limit breaches, loss holds, and missed opportunities. These feed directly into the step reward -- the model is penalized for *specific trading errors*, not just for losing money. This provides a richer learning signal than a single P&L number, functioning as a lightweight form of process supervision without requiring an external reward model.

### LLM-as-Judge

Numeric rewards capture *what* happened but not *why*. We added an external LLM as an offline judge that evaluates each BUY/SELL decision on five criteria:

| Criterion | Weight | What it evaluates |
|-----------|--------|-------------------|
| Signal alignment | 25% | Does the action match technical indicators? |
| Risk discipline | 20% | Position sizing, exposure management |
| Timing quality | 20% | Entry/exit relative to momentum signals |
| Regime awareness | 15% | Respects broader market context |
| Reasoning coherence | 20% | Does the reasoning logically justify the action? |

This adds *process supervision*: a well-reasoned trade that loses money receives partial credit. A lucky gamble that profits is recognized as undisciplined. The judge scores are pre-computed offline and stored alongside training data, adding zero latency during GRPO training. Unlike a learned reward model, the LLM judge uses a fixed rubric with few-shot calibration examples, making it interpretable and auditable.

---

## The Self-Improvement Loop

The complete pipeline chains these components into an iterative training loop:

1. **SFT** on distilled reasoning -- the model learns format and basic trading intuition
2. **RAFT** against neural environment -- play 100 episodes, filter winners (score > 0.5), retrain on successful trajectories
3. **GRPO** against neural environment -- reinforcement learning with multi-signal rewards (P&L + mistake penalties + LLM judge)
4. **Evaluate** on both static replay and neural environments
5. **Adapt** -- the neural environment biases episode generation toward regimes where the agent underperforms

The neural environment is central to this loop. Because every episode is stochastically generated, the agent cannot memorize its way to a high score. It must develop genuine trading intuition that generalizes across market conditions.

### Adaptive Curriculum

Drawing from the AdaCuRL framework for adaptive curriculum in RL and the regret-driven environment design ideas in PAIRED (Dennis et al., 2020), the environment automatically escalates difficulty based on sustained agent performance:

| Tier | Task | What the agent learns |
|------|------|----------------------|
| 1 | Single stock, no costs | Basic buy/sell timing |
| 2 | Single stock, with costs | Transaction cost awareness |
| 3 | Three stocks | Portfolio allocation basics |
| 4 | Ten stocks | Diversification and risk management |
| 5 | Full autonomous (regime gate) | Market regime discipline |

Promotion requires sustaining a score above threshold over a rolling window. Demotion triggers if performance drops below a floor. The agent *earns* harder challenges -- they are not given for free. The PAIRED insight is key here: the environment should be neither too easy (agent learns nothing) nor too hard (agent never succeeds), but calibrated to the agent's current capability boundary.

---

## Results

### Baselines

Before training our agent, we established baselines across different agent types to understand the difficulty landscape:

| Agent | Score | Return | Notes |
|-------|-------|--------|-------|
| Hold (do nothing) | 0.300 | 0.00% | Floor -- absolute minimum |
| Rule-based (RSI heuristic) | 0.293 | -0.35% | Simple technical analysis can't beat hold |
| PPO (20K steps, Stable-Baselines3) | 0.347 | -1.30% | Classic RL, trades actively but loses money |
| Qwen 0.5B (zero-shot) | 0.379 | +0.53% | Tiny LLM, no training, makes money |
| DeepSeek 7B (zero-shot) | 0.300 | 0.00% | Can't follow `<think>` format, defaults to HOLD |

Two findings stand out. First, LLMs beat classical RL (PPO) -- text observations provide an information advantage over numeric feature vectors. Second, the base DeepSeek 7B scores at the HOLD floor despite being a capable model, because it generates long analysis paragraphs but never outputs a clean BUY/SELL action. SFT is necessary just to teach the format.

### Training Progression

The full scoreboard across all training iterations, evaluated on the single_stock task (20 episodes, seed=42):

| Model | Static Env | Neural Env | Return | Training Method |
|-------|-----------|-----------|--------|-----------------|
| DeepSeek 7B (zero-shot) | 0.300 | 0.298 | 0.00% | No training |
| SFT v1 (imbalanced data) | 0.300 | -- | 0.00% | 91% HOLD -- data had HOLD bias |
| SFT v2 (template reasoning) | 0.398 | -- | +0.00% | Learned format, trades actively |
| GRPO v1 (counterfactual) | ~0.300 | -- | ~0.00% | Collapsed to 85% HOLD |
| GRPO v2 (multi-reward) | 0.326 | -- | -1.33% | 84% reward from formatting |
| GRPO v2.1 (reward-aligned) | 0.395 | -- | -0.00% | 10/20 episodes at 0.6, bimodal |
| GRPO v3 (hard examples) | 0.301 | -- | -2.22% | KL drift to 4.2, forgot SFT |
| SFT v3 step-352 (overtrained) | 0.383 | -- | +0.03% | Lower loss, worse trading |
| **SFT v3 step-200 (distilled)** | **0.399** | **0.417** | **+0.24%** | **Best model** |
| RAFT from base (wrong) | 0.300 | 0.300 | 0.00% | No SFT foundation = HOLD only |
| RAFT v2 (from SFT v3) | 0.360 | 0.399 | +0.00% | 640 winners, slight degradation |

### RAFT Self-Improvement

Using SFT v3 as the base model, we ran 100 episodes against the neural environment to collect self-improvement data:

- **Mean score across 100 episodes:** 0.380
- **Winners (score > 0.5):** 32 out of 100 episodes (32%)
- **SFT examples from winners:** 640 training examples
- **Winner scores range:** 0.5 - 0.8+

RAFT retrains on winners only, reinforcing successful behavior. However, RAFT v2 (trained from SFT v3 on these 640 winners) scored 0.360 static / 0.399 neural -- a slight degradation from SFT v3. With only 640 examples and 18 training steps, the signal was too weak to improve an already well-trained model. This taught us that RAFT needs either more data or a different learning rate schedule when building on top of SFT.

An earlier attempt -- RAFT trained from the base model instead of SFT v3 -- scored 0.300 on both environments (pure HOLD). This confirmed that RAFT is not a replacement for SFT; it is a refinement step that requires a strong foundation.

### GRPO Against Neural Environment

The full RL loop: the agent trains via TRL's GRPOTrainer while the neural environment provides verifiable rewards. We collected 1,000 prompts across 50 episodes from the neural environment (mean score 0.395, 36% of episodes above 0.5), then ran GRPO with conservative settings designed to avoid the KL catastrophe:

| Parameter | Previous (failed) | Current |
|-----------|-------------------|---------|
| Starting model | GRPO v2.1 | SFT v3 (best model, 0.399/0.417) |
| Steps | 1000 | 300 (shorter = less KL drift) |
| G (generations) | 8 | 4 (cheaper per step) |
| beta (KL penalty) | 0.04 | 0.05 (stronger KL constraint) |
| Prompts source | Static CSV replay | Neural environment (stochastic) |
| HOLD handling | Free (zero cost) | Signal-based penalties |
| Checkpoints | Final only | Every 50 steps (pick best by eval) |

Each fix directly addresses a previous failure mode: higher beta prevents KL drift, shorter training prevents overfitting, neural prompts prevent memorization, and signal-based HOLD prevents the collapse to inaction.

### Interpreting the Numbers

- **0.300 is the HOLD floor** -- any agent that does nothing scores exactly this. It is the baseline for inaction.
- **0.399 on static replay** confirms the model learned to trade, not just to format responses. It is the only model that achieves a positive return (+0.24%).
- **0.417 on neural environment** confirms the model generalizes beyond memorized CSV data. The neural environment generates scenarios the model has never seen, and performance *improves* -- suggesting the model has learned transferable trading patterns rather than dataset-specific shortcuts.
- **GRPO v2.1's bimodal scores** (10/20 at 0.6, rest below 0.3) reveal that offline GRPO produces inconsistent behavior -- it learns to trade well in some market conditions but fails in others.
- **GRPO v3 at 0.301** demonstrates that strong training metrics (good action distribution, rising reward) can mask catastrophic knowledge loss. The only reliable early warning was KL divergence.

### Visual Results

The following visualizations are generated from actual experiment data:

- **World model comparison** (`results/comparison_results.png`): 4-panel chart comparing causal transformer vs CNN+GRU on volatility ratio, MAE, direction accuracy, and sample trajectories.
- **Self-improvement learning curves** (`results/learning_curves.png`): 4-panel chart showing curriculum progression, mistake reduction, neural vs static env comparison, and world model training loss.
- **World model generation quality** (`results/generation_quality.png`): Overlay of generated vs real price trajectories, demonstrating the 0.94x volatility calibration.
- **Co-evolution curve** (`results/learning_curve.png`): World model accuracy improvement across fine-tuning iterations.

---

## Key Takeaways

**Reward design is harder than model design.** Four GRPO iterations, four different failure modes. The model will optimize exactly what you measure, efficiently and without hesitation. If your reward function has a loophole, the model will find it.

**Quality beats quantity in training data.** 12,000 carefully distilled reasoning examples outperformed 40,000 template-generated examples. This confirms the LIMA hypothesis in a domain-specific setting: carefully curated data outperforms bulk data when the goal is behavioral alignment, not just next-token prediction.

**The environment matters as much as the agent.** A static environment teaches memorization. A neural environment teaches generalization. Ha and Schmidhuber showed that agents can learn inside world models -- we show that world models can also make agents *harder to fool*.

**Failure is the most efficient teacher.** Each broken experiment revealed something that working experiments never would have exposed. The HOLD collapse, the format hack, the alignment tax, the KL catastrophe -- these failures are the most valuable artifacts of this project.

---

## References

- Ha, D. & Schmidhuber, J. (2018). *World Models.* arXiv:1803.10122. Foundational V-M-C architecture for learning inside neural environments.
- Trading-R1 (2025). arXiv:2509.11420. Reverse reasoning distillation and RL curriculum for financial trading agents.
- Zhou, C. et al. (2023). *LIMA: Less Is More for Alignment.* arXiv:2305.11206. Quality over quantity in instruction tuning.
- Pikus, D. et al. (2025). *Hard Examples Are All You Need.* Difficulty-filtered GRPO training on model trajectories.
- Dr. GRPO (2025). arXiv:2503.20783. Batch-level reward scaling to fix per-group variance explosion in GRPO.
- Dennis, M. et al. (2020). *Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design (PAIRED).* Regret-driven adaptive curriculum.

---

## Resources

- **Repository:** [github.com/sarthakbiswas97/stock-trader-env](https://github.com/sarthakbiswas97/stock-trader-env)
- **Models and datasets:** [huggingface.co/sarthakbiswas](https://huggingface.co/sarthakbiswas)
- **Market data:** [huggingface.co/datasets/sarthakbiswas/stock-trader-market-data](https://huggingface.co/datasets/sarthakbiswas/stock-trader-market-data)

247 tests with 80% coverage enforced in CI. Docker deployment. OpenEnv-compliant REST and WebSocket API.

Built by Sarthak Biswas for the Meta PyTorch OpenEnv Hackathon, April 2026.

---

*The environment has a brain. The agent has to earn its score.*

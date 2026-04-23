# Pitch Notes (3 min pitch + 2 min Q&A)

## One-Line Identity
"We gave the environment a brain -- a transformer that learns market dynamics, so the agent can't just memorize."

## Structure (3 minutes)

### 0:00-0:15 -- Hook
"Our RL agent learned to trade Indian stocks. Then it learned that doing nothing was the safest strategy. 85% of its actions were HOLD. So we gave the environment a brain to fight back."

### 0:15-0:45 -- The Problem (30s)
- Static environments replay CSV data -- agent memorizes, doesn't learn
- Real markets are stochastic -- yesterday's chart never repeats
- "What if the environment could generate scenarios the agent has never seen?"

### 0:45-1:15 -- The Architecture (30s)
- Causal transformer world model (1.22M params)
- Trained on 264K rows of real Indian market data
- Volatility 0.94x reality -- realistic enough to train on
- Drop-in replacement via OpenEnv REST API
- Agent doesn't know if data is real or generated

### 1:15-2:00 -- The Failure-to-Fix Journey (45s)
Pick TWO failures max (don't rush all four):

**Failure 1 (HOLD collapse):** "The model learned that HOLD is free. Zero cost, zero risk. 85% HOLD within 200 steps. So we made HOLD signal-based -- ignoring a strong buy signal now costs the agent."

**Failure 2 (Format hack):** "84% of reward came from formatting, 0% from trading. The model optimized for looking smart. We switched format to a gate, not a reward."

"Each failure taught us something that working experiments never would have."

### 2:00-2:30 -- Self-Improvement Pipeline (30s)
- SFT with reverse-distilled reasoning (GPT-4o-mini, $2.50 for 10K examples)
- RAFT: play episodes against neural env, keep winners, retrain
- LLM-as-Judge: GPT-4o-mini scores reasoning quality on 5 criteria
- Adaptive curriculum: 5 tiers, auto-promote based on score
- "The agent earns harder challenges -- they aren't given for free"

### 2:30-3:00 -- Results + Close (30s)
- Base model: 0.300 (can't follow format)
- Trained model: 0.399 static / 0.417 neural
- Neural > static confirms the model GENERALIZES
- "247 tests, 109 stocks, OpenEnv-compliant"
- "The environment has a brain. The agent has to earn its score."

## Q&A Prep

**Q: "Why not just use more training data?"**
A: "We tried. 40K template examples scored 0.398. 12K distilled examples scored 0.399. Quality beats quantity -- the LIMA paper was right, and we confirmed it empirically."

**Q: "How do you prevent the neural env from generating unrealistic data?"**
A: "The world model's volatility is 0.94x reality -- validated against ground truth. We clip extreme moves and use temperature 1.0 from a sweep of 0.1-1.5. The feature engine normalizes everything to indicators the agent already understands."

**Q: "Why did GRPO fail and SFT work?"**
A: "GRPO failed because of KL drift -- the model forgot its SFT knowledge while optimizing for RL reward. The signal to watch is KL divergence, not reward curves. We now use KL early stopping."

**Q: "What's the co-evolution part?"**
A: "The agent plays against the neural env, and the env adapts to surface the agent's weaknesses -- more episodes in regimes where the agent struggles. Both improve together."

**Q: "How is this different from other trading envs?"**
A: "Most trading envs are static CSV replay. Ours generates novel scenarios via a learned world model. The agent can't memorize. Plus we have signal-based HOLD penalties, LLM-as-judge for process supervision, and a 5-tier adaptive curriculum."

**Q: "What would you do with more time/compute?"**
A: "Online GRPO with TRL environment_factory -- the agent trains against the live neural env in real time, not on frozen prompts. That's Flow B -- true co-evolution."

## Key Phrases to Reinforce
- "The environment has a brain"
- "The agent has to earn its score"
- "Failure taught us more than success"
- "Constraint-driven -- single GPU, QLoRA"
- "Quality beats quantity"

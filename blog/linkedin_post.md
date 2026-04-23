# LinkedIn Post

## What 4 Failed RL Experiments Taught Me About Reward Design

I spent the last two weeks building an RL trading agent for the Meta PyTorch OpenEnv Hackathon -- a 7B LLM that reads market observations and decides BUY/SELL/HOLD on Indian equities.

It broke four times. Each time differently. Each time more instructively than any successful run.

**Failure 1:** The model learned that HOLD (doing nothing) carries zero downside. Within 200 training steps, 85% of actions were HOLD. If inaction is free, the agent will always choose it.

**Failure 2:** We added multiple reward functions. Post-mortem: 84% of reward came from formatting responses correctly. 0% from trading. The model optimized for looking smart, not being smart.

**Failure 3:** The checkpoint with the lowest training loss scored worse on actual trading than a checkpoint 150 steps earlier. Lower loss was memorization, not learning.

**Failure 4:** The best-looking training run -- ideal action distribution, rising rewards -- produced a model that scored below the untrained baseline. KL divergence had silently destroyed everything the model knew.

The fix: we gave the environment a brain.

A 1.22M parameter causal transformer learns market dynamics from 264,000 rows of real Indian stock data. Every episode is different. The agent cannot memorize its way to a good score. It has to actually learn to trade.

Combined with signal-based HOLD penalties (inaction has a cost), an LLM-as-Judge for reasoning quality evaluation, a 7-type mistake tracker wired directly into rewards, and a 5-tier adaptive curriculum that promotes the agent based on sustained performance.

Result: Base model 0.300 -> Trained model 0.417 on the neural environment.

The biggest lesson: reward design is harder than model design. The model will optimize exactly what you measure. If your metric has a loophole, the model will find it -- efficiently and without hesitation.

Built with PyTorch, OpenEnv, Unsloth, TRL, and a lot of failed experiments.

#MachineLearning #ReinforcementLearning #FinTech #PyTorch #AI #OpenEnv

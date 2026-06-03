# Neural World Model

I built this because agents were memorizing price sequences. Feed them the same seed, they see the same path every time, and they learn to exploit patterns that won't repeat. A simulator fixes that by generating novel episodes from the same starting point.

Most world model work is in 3D: robots, game engines, physics sims. Markets are just another environment with state that evolves. The "physics" here is statistical: how returns, volatility, and volume move together. This model learns that joint distribution, then samples new trajectories autoregressively. Agents train on scenarios they've never seen.

---

## What this is

People use "world model" to mean three very different things:

1. **Renderers** spit out pixels. Beautiful, maybe physically incoherent.
2. **Planners** spit out actions. Look at state, decide what to do.
3. **Simulators** spit out state. Model the underlying system so you can ask "what happens next?"

This is a **simulator**. It outputs raw market state: returns, volatility, volume. Not screenshots and not buy/sell decisions. That state can be rendered into charts or fed to a planner. The simulator is the piece everything else builds on.

---

## Why simulation, not replay

Historical replay is deterministic. Same seed, same path. Agents overfit hard. I watched this happen. GRPO runs would score well on training seeds and collapse on held-out ones.

A simulator breaks the determinism. Seed it with 100 days of real data, autoregressively roll out the next 20–40 days. Every seed gives a different path, but the statistical properties match real markets. The agent generalizes because it has to. It can't memorize what doesn't repeat.

Two architectures, same core idea: predict the next state as a distribution, not a point.

---

## Architectures

| | CausalTransformerWorldModel | MarketWorldModel |
|---|---|---|
| **Size** | ~1.22M params | ~140K (default) – 500K (large) |
| **Context** | Full sequence via causal attention | Compressed via CNN + GRU |
| **Predicts** | Next state at every position | Next state from final hidden state |
| **Best for** | Long episodes, richer context | Fast training, simpler dynamics |

### Causal transformer (the one I actually use)

Causal mask means each position only sees past data. Training matches inference. Feed a sequence, get next-step predictions at every token.

- 4 layers, d_model=192, 4 heads, ff=384
- 1D conv projects features into embedding space
- Learned positional embeddings
- MDN head at every position

### CNN + GRU baseline

Lighter. 1D convs for local pattern extraction, GRU for memory, MDN for prediction. Includes a decoder for reconstruction regularization. Extra signal to keep the encoder honest.

### Why MDN?

Predicting a single number for tomorrow's return is wrong. Markets are stochastic. An MDN outputs a mixture of Gaussians. A full distribution. Sample from it during generation, get different but plausible futures every time.

---

## State representation

Raw OHLCV is converted to returns-space features. Stationary, scale-invariant:

| Feature | What it is |
|---|---|
| `open_ret` | Open vs yesterday's close |
| `high_ret` | High vs yesterday's close |
| `low_ret` | Low vs yesterday's close |
| `close_ret` | Full day return |
| `log_vol_change` | Volume shift |
| `intraday_range` | (High − Low) / prev close |
| `body_ratio` | \|Close − Open\| / (High − Low) |

The model predicts the 5 price/volume features. Derived features are reconstructed analytically during rollouts.

Inputs clipped to `[-0.5, 0.5]`. Generated returns clipped to `[-0.05, 0.05]` so prices don't explode.

---

## Training

```bash
# Causal transformer, what I train and use
PYTHONPATH=. python scripts/train_world_model.py --model transformer --epochs 10

# CNN+GRU, lighter baseline
PYTHONPATH=. python scripts/train_world_model.py --model cnn-gru --epochs 10
```

Both use AdamW + cosine LR decay. Gradient clip at 1.0. Early stopping patience=3.

| | Transformer | CNN+GRU |
|---|---|---|
| LR | 3e-4 | 1e-3 |
| Weight decay | 1e-4 | 1e-5 |
| Batch | 128 | 128 |
| Loss | MDN NLL | MDN NLL + 0.1× recon MSE |
| Seq len | 100 | 50–100 |

Transformer uses multi-position causal loss: every token predicts the next step. More sample-efficient than single-step.

---

## Episode generation

```python
from world_model.model import CausalTransformerWorldModel, TransformerConfig

model = CausalTransformerWorldModel(TransformerConfig(seq_len=100))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

history = seed_features[-100:]  # (100, 7)

episode = []
for day in range(30):
    tensor = torch.from_numpy(history).unsqueeze(0)
    next_state, _ = model.predict_next(tensor, temperature=1.0)
    episode.append(next_state)
    history = np.concatenate([history[1:], next_state], axis=0)
```

**Temperature** controls randomness. Low (0.5) = conservative, stays near mean. High (1.5) = explores tails, good for stress-testing agent robustness.

---

## Validation

Generated episodes are scored against real held-out data:

| Metric | Checks |
|---|---|
| `volatility_ratio` | Does noise match real markets? |
| `direction_accuracy` | Up/down days correct? |
| `mae_returns` | How far off are returns? |

**What I observed (causal transformer):**
- Volatility calibration: 0.94× real markets
- Prediction error: ~3× lower than CNN+GRU baseline
- Every seed gives a unique path. No memorization possible

---

## File map

```
world_model/
├── model.py              # Both architectures, MDN loss, recon loss
├── data.py               # Feature extraction, datasets, normalization
├── trainer.py            # CNN+GRU trainer (legacy, still works)
└── transformer_building_blocks.ipynb
                          # Notebook walking through the transformer pieces

scripts/train_world_model.py
                          # Unified CLI for both architectures

server/neural_simulator.py
                          # Episode generator used by the RL environment
```

---

## Dependencies

- PyTorch ≥ 2.0
- NumPy, Pandas

No custom kernels. Standard PyTorch only.

---

## License

MIT (same as parent repo)

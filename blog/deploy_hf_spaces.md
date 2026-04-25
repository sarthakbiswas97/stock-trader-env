# Deploy v3/world-model to HF Spaces

## What needs to happen

Push the v3/world-model branch to the HF Space repo. The existing sync workflow pushes main, but we want v3 instead.

## Pre-checks

1. Dockerfile already handles world model checkpoint download from HF Hub (line 12-21)
2. requirements.txt has all runtime deps (openenv-core, fastapi, uvicorn, torch, pandas, numpy)
3. server/app.py uses openenv create_app -- no changes needed
4. openenv.yaml is correct
5. Neural env auto-detects world model checkpoint at `checkpoints/world_model/`

## What the Space needs that v3 has but main doesn't

- `server/neural_simulator.py` -- the neural env
- `server/curriculum.py` -- adaptive difficulty
- `server/mistake_tracker.py` -- 7 error types
- `world_model/` -- model code for loading checkpoint
- `training/` -- gym wrapper, evaluate (needed for inference.py)
- Updated `server/environment.py` -- simulator_mode support, HOLD fix

## Data files

Market data (data/ohlcv/, data/macro/) is gitignored. The environment auto-downloads from HF Hub on first reset. This already works in the Dockerfile build step AND at runtime via ensure_market_data() in eval_sft.py.

BUT: the environment itself (server/environment.py reset()) does NOT auto-download. It expects data to exist. The Dockerfile needs to download data at build time.

## Step-by-step deployment

### Step 1: Add market data download to Dockerfile

The current Dockerfile downloads the world model checkpoint but NOT market data. Add:

```dockerfile
# Download market data
RUN python -c "\
from datasets import load_dataset; \
from pathlib import Path; \
ds = load_dataset('sarthakbiswas/stock-trader-market-data'); \
ohlcv_dir = Path('data/ohlcv'); ohlcv_dir.mkdir(parents=True, exist_ok=True); \
macro_dir = Path('data/macro'); macro_dir.mkdir(parents=True, exist_ok=True); \
[g.drop(columns=['symbol','data_type']).to_csv(ohlcv_dir/f'{s}_daily.csv', index=False) for s,g in ds['ohlcv'].to_pandas().groupby('symbol')]; \
[g.drop(columns=['symbol','data_type']).to_csv(macro_dir/f'{s}_daily.csv', index=False) for s,g in ds['macro'].to_pandas().groupby('symbol')]; \
print(f'Downloaded {len(ds[\"ohlcv\"].to_pandas().symbol.unique())} stocks') \
"
```

Also add `datasets` and `huggingface_hub` to requirements.txt if missing.

### Step 2: Push v3 branch to HF Space

Option A -- manual push:
```bash
git remote add hf-space https://sarthakbiswas:$HF_TOKEN@huggingface.co/spaces/sarthakbiswas/stock-trader-env

git push hf-space v3/world-model:main --force
```

This force-pushes v3/world-model as main on the HF Space, triggering a Docker rebuild.

Option B -- update the sync workflow to push v3 branch:
Edit `.github/workflows/sync-hf.yml` to trigger on v3/world-model instead of main.

### Step 3: Verify

After push, HF Spaces will:
1. Build Docker image (2-3 min)
2. Download market data + world model checkpoint (1-2 min)
3. Start uvicorn server
4. Health check passes

Test endpoints:
- `GET /health` -- should return 200
- `POST /reset` -- should return MarketObservation
- `POST /step` with `{"action": "HOLD"}` -- should return next observation

### Step 4: Test neural env via Space

```bash
curl -X POST https://sarthakbiswas-stock-trader-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 42, "task_id": "single_stock", "simulator_mode": "neural"}'
```

## Requirements.txt changes needed

Add `datasets` (for market data download in Dockerfile):
```
openenv-core
fastapi
uvicorn[standard]
pandas
numpy
pydantic>=2.0
openai
httpx
yfinance
torch
datasets
huggingface_hub
```

## Risks

1. **Docker build fails** -- torch CPU install can be finicky. Current Dockerfile already handles this.
2. **World model checkpoint too large** -- best_transformer.pt is ~5MB. Not an issue.
3. **Market data download timeout** -- 264K rows. Should be fine but could timeout on slow HF Spaces build.
4. **Neural env requires torch** -- already in requirements.txt.
5. **HF Space free tier memory** -- 7B model inference WON'T work on Space. The Space runs the ENVIRONMENT only (not the agent). Inference happens on Colab/RunPod.

## Key point for judges

The HF Space runs the **environment**, not the agent. Any agent can connect via the REST API and trade. The Space demonstrates:
- OpenEnv compliance (reset/step/state endpoints)
- Neural env mode (simulator_mode="neural")
- Docker deployment
- Real market data

The **agent** (model inference) runs separately on Colab or GPU pod.

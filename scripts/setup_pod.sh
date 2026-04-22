#!/bin/bash
# One-shot pod setup for SFT/GRPO training.
# Run: bash scripts/setup_pod.sh

set -e

echo "=== Pod Setup ==="

# Clone and checkout
if [ ! -d "stock-trader-env" ]; then
    git clone https://github.com/sarthakbiswas97/stock-trader-env.git
fi
cd stock-trader-env
git checkout v3/world-model
git pull origin v3/world-model

# Install dependencies
pip install -q openenv-core gymnasium mlflow
pip install -q --ignore-installed blinker

# Download market data
python3 -c "
from datasets import load_dataset
from pathlib import Path
ds = load_dataset('sarthakbiswas/stock-trader-market-data')
ohlcv_dir = Path('data/ohlcv')
ohlcv_dir.mkdir(parents=True, exist_ok=True)
macro_dir = Path('data/macro')
macro_dir.mkdir(parents=True, exist_ok=True)
for symbol, group in ds['ohlcv'].to_pandas().groupby('symbol'):
    group.drop(columns=['symbol','data_type']).to_csv(ohlcv_dir / f'{symbol}_daily.csv', index=False)
for name, group in ds['macro'].to_pandas().groupby('symbol'):
    group.drop(columns=['symbol','data_type']).to_csv(macro_dir / f'{name}_daily.csv', index=False)
print(f'Downloaded {len(ds[\"ohlcv\"].to_pandas().symbol.unique())} stocks + macro data')
"

# Download world model checkpoint
python3 -c "
from huggingface_hub import hf_hub_download
from pathlib import Path
Path('checkpoints/world_model').mkdir(parents=True, exist_ok=True)
hf_hub_download(repo_id='sarthakbiswas/stock-trader-market-data', filename='best_transformer.pt', repo_type='dataset', local_dir='checkpoints/world_model')
print('World model checkpoint downloaded')
"

echo ""
echo "=== Setup Complete ==="
echo "Ready to train. Run:"
echo "  cd stock-trader-env"
echo "  PYTHONPATH=. python3 scripts/train_sft.py"
echo "  PYTHONPATH=. python3 scripts/train_sft.py --push-to-hub"

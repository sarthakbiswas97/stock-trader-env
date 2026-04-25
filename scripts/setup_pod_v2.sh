#!/bin/bash
# One-shot pod setup for training/eval on RunPod.
# Handles all dependency conflicts (pydantic, mergekit, mcp, blinker).
#
# Usage: bash scripts/setup_pod_v2.sh
# Then: export HF_TOKEN="your_token"

set -e

echo "=== Pod Setup v2 ==="

# Clone and checkout
if [ ! -d "stock-trader-env" ]; then
    git clone https://github.com/sarthakbiswas97/stock-trader-env.git
fi
cd stock-trader-env
git checkout v3/world-model
git pull origin v3/world-model

# Core deps
pip install -q --ignore-installed blinker
pip install -q openenv-core gymnasium mlflow immutables

# TRL dependency stubs (mergekit, llm_blender, weave)
pip install -q --no-deps llm_blender mergekit weave 2>/dev/null || true

# Fix pydantic conflict (mcp requires <2.11, mergekit needs compatible version)
pip install -q "pydantic>=2.0,<2.11" "mcp<1.9" 2>/dev/null || \
pip install -q pydantic==2.10.6 2>/dev/null || true

# Patch TRANSFORMERS_CACHE (llm_blender needs it, transformers 5.x removed it)
python3 -c "
import transformers.utils.hub as _hub
from huggingface_hub.constants import HF_HUB_CACHE
_hub.TRANSFORMERS_CACHE = HF_HUB_CACHE
import llm_blender
print('llm_blender OK')
" 2>/dev/null || echo "llm_blender patch skipped"

# Verify unsloth loads
python3 -c "from unsloth import FastLanguageModel; print('Unsloth OK')"

# Download market data
python3 -c "
from datasets import load_dataset
from pathlib import Path
ohlcv_dir = Path('data/ohlcv')
if not ohlcv_dir.exists() or not any(ohlcv_dir.glob('*.csv')):
    ds = load_dataset('sarthakbiswas/stock-trader-market-data')
    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    macro_dir = Path('data/macro')
    macro_dir.mkdir(parents=True, exist_ok=True)
    for symbol, group in ds['ohlcv'].to_pandas().groupby('symbol'):
        group.drop(columns=['symbol','data_type']).to_csv(ohlcv_dir / f'{symbol}_daily.csv', index=False)
    for name, group in ds['macro'].to_pandas().groupby('symbol'):
        group.drop(columns=['symbol','data_type']).to_csv(macro_dir / f'{name}_daily.csv', index=False)
    print(f'Downloaded {len(ds[\"ohlcv\"].to_pandas().symbol.unique())} stocks + macro data')
else:
    print(f'Market data exists: {len(list(ohlcv_dir.glob(\"*.csv\")))} stocks')
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
echo "Next steps:"
echo "  export HF_TOKEN=your_token"
echo "  cd stock-trader-env"
echo "  PYTHONPATH=. python3 scripts/eval_sft.py --checkpoint sarthakbiswas/stock-trader-grpo-neural-model --episodes 5"

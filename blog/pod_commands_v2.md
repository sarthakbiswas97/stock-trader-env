# Pod Commands: Improved Env Training + Multi-Tier Eval

## Pre-requisites
1. Push all code to GitHub (refactored env + system prompt)
2. Spin up A5000 pod
3. Set HF_TOKEN

## Step 1: Setup (5 min)

```bash
cd /root && \
git clone https://github.com/sarthakbiswas97/stock-trader-env.git && \
cd stock-trader-env && \
git checkout v3/world-model && \
pip install -q --ignore-installed blinker && \
pip install -q openenv-core gymnasium mlflow immutables && \
pip install -q --no-deps llm_blender mergekit weave
```

## Step 2: Download data (1 min)

```bash
cd /root/stock-trader-env && \
python3 -c "
from huggingface_hub import hf_hub_download
from pathlib import Path
Path('checkpoints/world_model').mkdir(parents=True, exist_ok=True)
hf_hub_download(repo_id='sarthakbiswas/stock-trader-market-data', filename='best_transformer.pt', repo_type='dataset', local_dir='checkpoints/world_model')
print('Done')
"
```

## Step 3: Eval existing GRPO on 3 tiers — BEFORE baseline (15 min)

```bash
nohup bash -c '
cd /root/stock-trader-env && \
echo "=== BEFORE EVAL START $(date) ===" && \

PYTHONPATH=. python3 scripts/eval_sft.py \
  --checkpoint sarthakbiswas/stock-trader-grpo-neural-model \
  --simulator-mode replay \
  --task single_stock \
  --episodes 5 \
  --output /root/before_single_stock_replay.json && \

PYTHONPATH=. python3 scripts/eval_sft.py \
  --checkpoint sarthakbiswas/stock-trader-grpo-neural-model \
  --simulator-mode neural \
  --task single_stock \
  --episodes 5 \
  --output /root/before_single_stock_neural.json && \

PYTHONPATH=. python3 scripts/eval_sft.py \
  --checkpoint sarthakbiswas/stock-trader-grpo-neural-model \
  --simulator-mode replay \
  --task single_stock_costs \
  --episodes 5 \
  --output /root/before_costs_replay.json && \

PYTHONPATH=. python3 scripts/eval_sft.py \
  --checkpoint sarthakbiswas/stock-trader-grpo-neural-model \
  --simulator-mode replay \
  --task multi_stock_3 \
  --episodes 5 \
  --output /root/before_multi3_replay.json && \

echo "=== BEFORE EVAL DONE $(date) ==="
' > /root/before_eval.log 2>&1 &
```

Check: `grep "RESULTS\|===" /root/before_eval.log`

## Step 4: Collect prompts from improved neural env (20 min)

```bash
nohup bash -c '
cd /root/stock-trader-env && \
echo "=== COLLECT START $(date) ===" && \
PYTHONPATH=. python3 scripts/collect_model_episodes.py \
  --checkpoint sarthakbiswas/stock-trader-sft-v3-model \
  --simulator-mode neural \
  --episodes 50 \
  --output /workspace/grpo_improved_prompts.jsonl && \
echo "=== COLLECT DONE $(date) ==="
' > /root/collect_improved.log 2>&1 &
```

Check: `grep "Episode\|DONE" /root/collect_improved.log | tail -5`

## Step 5: GRPO training on improved env (1.5 hr)

```bash
export HF_TOKEN="YOUR_TOKEN"
nohup bash -c '
cd /root/stock-trader-env && \
echo "=== GRPO v2 START $(date) ===" && \
PYTHONPATH=. python3 scripts/train_grpo.py \
  --sft-checkpoint sarthakbiswas/stock-trader-sft-v3-model \
  --prompts-dataset /workspace/grpo_improved_prompts.jsonl \
  --max-steps 300 \
  --num-generations 4 \
  --beta 0.05 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 5e-7 \
  --output-dir /workspace/grpo-improved-checkpoint && \
echo "=== GRPO v2 DONE $(date) ==="
' > /root/grpo_improved.log 2>&1 &
```

Check: `grep "Actions\|===" /root/grpo_improved.log | tail -5`

## Step 6: Eval new model on 3 tiers — AFTER (15 min)

```bash
nohup bash -c '
cd /root/stock-trader-env && \
echo "=== AFTER EVAL START $(date) ===" && \

PYTHONPATH=. python3 scripts/eval_sft.py \
  --checkpoint /workspace/grpo-improved-checkpoint \
  --simulator-mode replay \
  --task single_stock \
  --episodes 5 \
  --output /root/after_single_stock_replay.json && \

PYTHONPATH=. python3 scripts/eval_sft.py \
  --checkpoint /workspace/grpo-improved-checkpoint \
  --simulator-mode neural \
  --task single_stock \
  --episodes 5 \
  --output /root/after_single_stock_neural.json && \

PYTHONPATH=. python3 scripts/eval_sft.py \
  --checkpoint /workspace/grpo-improved-checkpoint \
  --simulator-mode replay \
  --task single_stock_costs \
  --episodes 5 \
  --output /root/after_costs_replay.json && \

PYTHONPATH=. python3 scripts/eval_sft.py \
  --checkpoint /workspace/grpo-improved-checkpoint \
  --simulator-mode replay \
  --task multi_stock_3 \
  --episodes 5 \
  --output /root/after_multi3_replay.json && \

echo "=== AFTER EVAL DONE $(date) ==="
' > /root/after_eval.log 2>&1 &
```

## Step 7: Upload + save (5 min)

```bash
python3 -c "
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get('HF_TOKEN'))
api.create_repo('sarthakbiswas/stock-trader-grpo-improved-model', repo_type='model', exist_ok=True)
api.upload_folder(folder_path='/workspace/grpo-improved-checkpoint', repo_id='sarthakbiswas/stock-trader-grpo-improved-model', repo_type='model', token=os.environ.get('HF_TOKEN'))
print('Uploaded')
"
```

## Check results

```bash
for f in /root/before_*.json /root/after_*.json; do
  echo "=== $(basename $f) ==="
  python3 -c "import json; d=json.load(open('$f')); print(f'Score: {d[\"mean_score\"]:.3f} | Return: {d[\"mean_return\"]*100:+.2f}%')"
done
```

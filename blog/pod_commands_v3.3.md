# Pod Commands: GRPO v3.3 — Tuned Hyperparams + KL Early Stop

## What changed from v3.2
- beta: 0.05 -> 0.02 (allow more exploration, v3.2 KL was too low at 0.07-0.19)
- num_generations: 4 -> 6 (better advantage estimates)
- lr: 5e-7 -> 2e-7 (gentler since starting from SFT v3)
- 300 steps (not 500 — v3.2 collapsed at step 500)
- save every 50 steps (finer granularity than v3.2's 100-step gaps)
- KL early stop at 2.0 (catches instability before catastrophe)
- Eval: 10 episodes per checkpoint (not 5 — reduces variance)
- Eval on single_stock AND single_stock_costs (shows generalization)

## Setup (copy-paste one block)

```bash
cd /root && \
git clone https://github.com/sarthakbiswas97/stock-trader-env.git 2>/dev/null; \
cd /root/stock-trader-env && \
git checkout v3/world-model && \
git pull origin v3/world-model && \
pip install -q --ignore-installed blinker && \
pip install -q openenv-core gymnasium mlflow immutables weave && \
pip install -q gql[requests] dataclasses_json && \
pip install -q --no-deps llm_blender mergekit==0.0.5 && \
pip install -q "fastmcp>=3.0.0" && \
python3 -c "
from huggingface_hub import hf_hub_download
from pathlib import Path
Path('checkpoints/world_model').mkdir(parents=True, exist_ok=True)
hf_hub_download(repo_id='sarthakbiswas/stock-trader-market-data', filename='best_transformer.pt', repo_type='dataset', local_dir='checkpoints/world_model')
Path('/workspace/grpo-data').mkdir(parents=True, exist_ok=True)
hf_hub_download(repo_id='sarthakbiswas/stock-trader-market-data', filename='grpo_improved_prompts.jsonl', repo_type='dataset', local_dir='/workspace/grpo-data')
print('Data done')
" && \
python3 -c "from unsloth import FastLanguageModel; print('ALL OK')"
```

## Pipeline (single nohup)

```bash
export HF_TOKEN="YOUR_TOKEN"

cat > /root/pipeline_v3.3.sh << 'SCRIPT_EOF'
#!/bin/bash
set -e
cd /root/stock-trader-env
export HF_TOKEN="YOUR_TOKEN"

echo "=== GRPO v3.3 START $(date) ==="

# Train: beta=0.02, G=6, lr=2e-7, 300 steps, save every 50
PYTHONPATH=. python3 scripts/train_grpo.py \
  --sft-checkpoint sarthakbiswas/stock-trader-sft-v3-model \
  --prompts-dataset /workspace/grpo-data/grpo_improved_prompts.jsonl \
  --max-steps 300 \
  --save-steps 50 \
  --num-generations 6 \
  --beta 0.02 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 2e-7 \
  --kl-early-stop 2.0 \
  --output-dir /workspace/grpo-v3.3-checkpoint

echo "=== TRAINING DONE $(date) ==="

# Eval each checkpoint on neural env (10 episodes, two tasks)
echo "=== EVAL CHECKPOINTS START $(date) ==="
for step in 50 100 150 200 250 300; do
  ckpt="/workspace/grpo-v3.3-checkpoint/checkpoint-${step}"
  if [ -d "$ckpt" ]; then
    echo "--- Evaluating checkpoint-${step} ---"

    # single_stock (neural)
    PYTHONPATH=. python3 scripts/eval_sft.py \
      --checkpoint "$ckpt" \
      --simulator-mode neural \
      --task single_stock \
      --episodes 10 \
      --output "/root/ckpt_${step}_neural_single.json"

    # single_stock_costs (neural)
    PYTHONPATH=. python3 scripts/eval_sft.py \
      --checkpoint "$ckpt" \
      --simulator-mode neural \
      --task single_stock_costs \
      --episodes 10 \
      --output "/root/ckpt_${step}_neural_costs.json"
  fi
done

# Also eval final model dir
echo "--- Evaluating final ---"
PYTHONPATH=. python3 scripts/eval_sft.py \
  --checkpoint /workspace/grpo-v3.3-checkpoint \
  --simulator-mode neural \
  --task single_stock \
  --episodes 10 \
  --output /root/ckpt_final_neural_single.json

PYTHONPATH=. python3 scripts/eval_sft.py \
  --checkpoint /workspace/grpo-v3.3-checkpoint \
  --simulator-mode neural \
  --task single_stock_costs \
  --episodes 10 \
  --output /root/ckpt_final_neural_costs.json

# Print checkpoint comparison table
echo ""
echo "=========================================="
echo "CHECKPOINT COMPARISON (neural env)"
echo "=========================================="
echo ""
echo "--- single_stock ---"
for f in /root/ckpt_*_neural_single.json; do
  name=$(basename "$f" _neural_single.json)
  python3 -c "
import json
d = json.load(open('${f}'))
scores = [round(s,3) for s in d['scores']]
print(f'  ${name}: {d[\"mean_score\"]:.3f} +/- {d[\"std_score\"]:.3f} | {scores}')"
done

echo ""
echo "--- single_stock_costs ---"
for f in /root/ckpt_*_neural_costs.json; do
  name=$(basename "$f" _neural_costs.json)
  python3 -c "
import json
d = json.load(open('${f}'))
scores = [round(s,3) for s in d['scores']]
print(f'  ${name}: {d[\"mean_score\"]:.3f} +/- {d[\"std_score\"]:.3f} | {scores}')"
done
echo "=========================================="

# Find best checkpoint
python3 -c "
import json, glob
best_score, best_name = 0, 'none'
for f in sorted(glob.glob('/root/ckpt_*_neural_single.json')):
    d = json.load(open(f))
    if d['mean_score'] > best_score:
        best_score = d['mean_score']
        best_name = f
print(f'BEST: {best_name} -> {best_score:.3f}')
"

# Upload all checkpoints to HF
echo "=== UPLOAD $(date) ==="
python3 -c "
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get('HF_TOKEN'))
api.create_repo('sarthakbiswas/stock-trader-grpo-v3.3-model', repo_type='model', exist_ok=True)
api.upload_folder(folder_path='/workspace/grpo-v3.3-checkpoint', repo_id='sarthakbiswas/stock-trader-grpo-v3.3-model', repo_type='model', token=os.environ.get('HF_TOKEN'))
print('Model v3.3 uploaded')
"

# Upload eval results
python3 -c "
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get('HF_TOKEN'))
api.create_repo('sarthakbiswas/stock-trader-market-data', repo_type='dataset', exist_ok=True)
import glob, json
results = {}
for f in sorted(glob.glob('/root/ckpt_*.json')):
    name = os.path.basename(f).replace('.json', '')
    results[name] = json.load(open(f))
with open('/tmp/grpo_v3.3_eval_results.json', 'w') as fout:
    json.dump(results, fout, indent=2)
api.upload_file(path_or_fileobj='/tmp/grpo_v3.3_eval_results.json', path_in_repo='grpo_v3.3_eval_results.json', repo_id='sarthakbiswas/stock-trader-market-data', repo_type='dataset', token=os.environ.get('HF_TOKEN'))
print('Eval results uploaded')
"

echo "=== ALL DONE $(date) ==="
echo "Stopping pod in 60s..."
sleep 60
runpodctl stop pod $RUNPOD_POD_ID 2>/dev/null || echo "Manual pod stop needed"
SCRIPT_EOF

# Replace YOUR_TOKEN in the script
sed -i "s/YOUR_TOKEN/${HF_TOKEN}/g" /root/pipeline_v3.3.sh

nohup bash /root/pipeline_v3.3.sh > /root/pipeline_v3.3.log 2>&1 &
echo "Pipeline started. Check: grep '===' /root/pipeline_v3.3.log"
```

## Monitor

```bash
# Quick status
grep "===\|BEST\|KL.*threshold" /root/pipeline_v3.3.log | tail -15

# Watch KL (should be 0.2-0.8, not <0.1 like v3.2)
grep "kl" /root/pipeline_v3.3.log | tail -10

# Action distribution
grep "Actions\|HOLD%" /root/pipeline_v3.3.log | tail -10

# Full checkpoint table (after eval phase)
grep -A 20 "CHECKPOINT COMPARISON" /root/pipeline_v3.3.log
```

# Pod Commands: GRPO v3.2 — Extended 500 Steps

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

cat > /root/pipeline_v3.sh << 'SCRIPT_EOF'
#!/bin/bash
set -e
cd /root/stock-trader-env
export HF_TOKEN="YOUR_TOKEN"

echo "=== GRPO v3.2 START $(date) ==="
PYTHONPATH=. python3 scripts/train_grpo.py \
  --sft-checkpoint sarthakbiswas/stock-trader-sft-v3-model \
  --prompts-dataset /workspace/grpo-data/grpo_improved_prompts.jsonl \
  --max-steps 500 \
  --save-steps 100 \
  --num-generations 4 \
  --beta 0.05 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 5e-7 \
  --output-dir /workspace/grpo-v3.2-checkpoint
echo "=== TRAINING DONE $(date) ==="

# Eval each checkpoint on neural env (quick - 5 episodes each)
echo "=== EVAL CHECKPOINTS START $(date) ==="
for step in 100 200 300 400 500; do
  ckpt="/workspace/grpo-v3.2-checkpoint/checkpoint-${step}"
  if [ -d "$ckpt" ]; then
    PYTHONPATH=. python3 scripts/eval_sft.py \
      --checkpoint "$ckpt" \
      --simulator-mode neural \
      --task single_stock \
      --episodes 5 \
      --output "/root/ckpt_${step}_neural.json"
  fi
done

# Also eval final checkpoint
PYTHONPATH=. python3 scripts/eval_sft.py \
  --checkpoint /workspace/grpo-v3.2-checkpoint \
  --simulator-mode neural \
  --task single_stock \
  --episodes 5 \
  --output /root/ckpt_final_neural.json

# Print all checkpoint scores
echo ""
echo "=========================================="
echo "CHECKPOINT COMPARISON (neural env)"
echo "=========================================="
for f in /root/ckpt_*_neural.json; do
  name=$(basename "$f" .json)
  python3 -c "
import json
d = json.load(open('${f}'))
print(f'  ${name}: {d[\"mean_score\"]:.3f} | {[round(s,3) for s in d[\"scores\"]]}')"
done
echo "=========================================="

# Upload best checkpoint (the final one for now - pick manually if needed)
echo "=== UPLOAD $(date) ==="
python3 -c "
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get('HF_TOKEN'))
api.create_repo('sarthakbiswas/stock-trader-grpo-v3.2-model', repo_type='model', exist_ok=True)
api.upload_folder(folder_path='/workspace/grpo-v3.2-checkpoint', repo_id='sarthakbiswas/stock-trader-grpo-v3.2-model', repo_type='model', token=os.environ.get('HF_TOKEN'))
print('Model v3.2 uploaded')
"

echo "=== ALL DONE $(date) ==="
echo "Stopping pod in 60s..."
sleep 60
runpodctl stop pod $RUNPOD_POD_ID 2>/dev/null || echo "Manual pod stop needed"
SCRIPT_EOF

# Replace YOUR_TOKEN in the script
sed -i "s/YOUR_TOKEN/${HF_TOKEN}/g" /root/pipeline_v3.sh

nohup bash /root/pipeline_v3.sh > /root/pipeline_v3.log 2>&1 &
echo "Pipeline started. Check: grep '===' /root/pipeline_v3.log"
```

## Monitor

```bash
grep "Actions\|===\|RESULTS" /root/pipeline_v3.log | tail -10
```

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Download world model checkpoint from HF Hub if not present
RUN python -c "\
from pathlib import Path; \
p = Path('checkpoints/world_model/best_transformer.pt'); \
p.parent.mkdir(parents=True, exist_ok=True); \
if not p.exists(): \
    from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='sarthakbiswas/stock-trader-market-data', filename='best_transformer.pt', repo_type='dataset', local_dir='checkpoints/world_model'); \
    print('Downloaded checkpoint') \
else: print('Checkpoint exists') \
" || echo "Checkpoint download skipped"

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import os, httpx; httpx.get(f'http://localhost:{os.environ.get(\"PORT\", \"8000\")}/health').raise_for_status()" || exit 1

CMD uvicorn server.app:app --host 0.0.0.0 --port ${PORT}

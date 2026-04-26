FROM python:3.11-slim

WORKDIR /app

# Install deps (torch CPU for world model inference only)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Download market data from HF Hub (109 stocks + macro)
RUN python -c "\
from datasets import load_dataset; \
from pathlib import Path; \
ds = load_dataset('sarthakbiswas/stock-trader-market-data'); \
ohlcv = Path('data/ohlcv'); ohlcv.mkdir(parents=True, exist_ok=True); \
macro = Path('data/macro'); macro.mkdir(parents=True, exist_ok=True); \
[g.drop(columns=['symbol','data_type']).to_csv(ohlcv/f'{s}_daily.csv', index=False) for s,g in ds['ohlcv'].to_pandas().groupby('symbol')]; \
[g.drop(columns=['symbol','data_type']).to_csv(macro/f'{s}_daily.csv', index=False) for s,g in ds['macro'].to_pandas().groupby('symbol')]; \
print(f'Downloaded {len(ds[\"ohlcv\"].to_pandas().symbol.unique())} stocks + macro') \
"

# Download world model checkpoint
RUN python -c "\
from pathlib import Path; \
p = Path('checkpoints/world_model/best_transformer.pt'); \
p.parent.mkdir(parents=True, exist_ok=True); \
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='sarthakbiswas/stock-trader-market-data', filename='best_transformer.pt', repo_type='dataset', local_dir='checkpoints/world_model'); \
print('World model downloaded') \
"

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()" || exit 1

CMD uvicorn server.app:app --host 0.0.0.0 --port ${PORT}

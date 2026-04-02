"""FastAPI application for the Stock Trading Environment."""

from openenv.core.env_server import create_app

from server.environment import StockTradingEnvironment
from models import TradeAction, MarketObservation


app = create_app(
    env=StockTradingEnvironment,
    action_cls=TradeAction,
    observation_cls=MarketObservation,
)


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

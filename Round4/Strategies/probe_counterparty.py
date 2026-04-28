"""Probe trader: prints the first non-empty market_trades payload it sees, to
confirm the Rust backtester populates Trade.buyer / Trade.seller from the R4
CSV columns. Does no actual trading.
"""
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List


class Trader:
    def __init__(self):
        self.printed = False

    def run(self, state: TradingState):
        out: Dict[str, List[Order]] = {}
        if not self.printed and state.market_trades:
            for symbol, trades in state.market_trades.items():
                if trades:
                    t = trades[0]
                    print(
                        f"PROBE ts={state.timestamp} symbol={symbol} "
                        f"price={t.price} qty={t.quantity} "
                        f"buyer={t.buyer!r} seller={t.seller!r}"
                    )
                    self.printed = True
                    break
        return out, 0, ""

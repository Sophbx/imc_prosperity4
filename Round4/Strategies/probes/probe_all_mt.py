"""Probe: count market_trades per product."""
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
from collections import Counter


class Trader:
    def __init__(self):
        self.product_counts: Counter = Counter()

    def run(self, state: TradingState):
        for sym, trades in state.market_trades.items():
            if trades:
                self.product_counts[sym] += len(trades)
        if state.timestamp >= 999000:
            for sym, n in sorted(self.product_counts.items()):
                print(f"FINAL {sym}: {n} market_trades")
        return {}, 0, ""

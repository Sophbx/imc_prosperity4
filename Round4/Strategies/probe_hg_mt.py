"""Probe: count HG market_trades and Mark 22 events. Does NO trading."""
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List


class Trader:
    def __init__(self):
        self.total_hg_mt = 0
        self.m22_hg_events = 0
        self.last_summary_ts = 0

    def run(self, state: TradingState):
        out: Dict[str, List[Order]] = {}
        hg_mt = state.market_trades.get("HYDROGEL_PACK", []) or []
        if hg_mt:
            self.total_hg_mt += len(hg_mt)
            for t in hg_mt:
                if t.buyer == "Mark 22" or t.seller == "Mark 22":
                    self.m22_hg_events += 1
                    print(f"M22_HG ts={state.timestamp} buyer={t.buyer} seller={t.seller}")
        # Periodic summary
        if state.timestamp - self.last_summary_ts >= 100000:
            self.last_summary_ts = state.timestamp
            print(f"SUMMARY ts={state.timestamp} total_hg_mt={self.total_hg_mt} m22_hg={self.m22_hg_events}")
        return out, 0, ""

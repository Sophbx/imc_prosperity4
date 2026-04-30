"""
PEBBLES basket snap arb — v4: tuned v3c.

Path so far (v1/v2/v3a/v3b/v3c folders pruned earlier; numbers preserved here)

    v1   take (cross spread, all 5 legs)              ->  -58
    v2   make, single-sided, bid+1 / ask-1            ->  +1,871
    v3a  make, single-sided, bid+2 / ask-2            ->  +1,531
    v3b  make, single-sided, bid+3 / ask-3            ->  +1,194
    v3c  make, two-sided, bid+1 / ask-1, K=2 b=1 m=10 ->  +25,434
    v4   tuned v3c (this file)                        ->  +53,988  (+112% vs v3c)

What v4 changes
    Structurally identical to v3c (two-sided make at bid+1 / ask-1, same
    target/inventory rule). Only the parameters change:

        BASELINE_SIZE  1  ->  5     (the big one - 5 lots per side baseline,
                                    not 1, so spread capture is ~5x larger)
        K              2  ->  1     (irrelevant at high baseline; PnL is
                                    identical for K in {1, 2, 4, 8, 16, 32})
        MAX_TRADE_SIZE 10 ->  10    (unchanged; >=5 saturates)
        OFFSET         1  ->  1     (unchanged; ofs=1 still optimal)

What the sweep showed
    1. BASELINE_SIZE is by far the most sensitive parameter:
            0 -> 2,704     (only quote on basket signal - almost no spread capture)
            1 -> 23,536
            2 -> 40,308
            3 -> 48,720
            5+ -> 53,988   (saturated; counterparty flow caps fill rate)
    2. K (signal scaling) only helps when baseline is small. At baseline=5,
       K in {1, 2, 4, 8, 16, 32} all yield 53,988 because BASELINE_SIZE
       already saturates two-sided quoting and the signal-driven extra
       lots can't grow further inside the position cap.
    3. MAX_TRADE_SIZE is a non-issue once it's >= BASELINE_SIZE.
    4. OFFSET=1 is optimal (sweep done in v3a/v3b earlier; ofs=2,3 hurt).
"""

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple


class Trader:
    SUFFIXES = ["XS", "S", "M", "L", "XL"]
    SYMS = [f"PEBBLES_{s}" for s in SUFFIXES]
    POS_LIMIT = 10
    BASKET_FAIR = 50_000.0

    K = 1.0                # any value ≥ 1 gives identical PnL at this baseline
    MAX_TRADE_SIZE = 10
    BASELINE_SIZE = 5      # ★ the change that matters
    OFFSET = 1

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}

        books: Dict[str, Tuple[int, int]] = {}
        mids: Dict[str, float] = {}
        for sym in self.SYMS:
            d = state.order_depths.get(sym)
            if d is None or not d.buy_orders or not d.sell_orders:
                return {}, 0, ""
            bb = max(d.buy_orders); ba = min(d.sell_orders)
            if bb >= ba:
                return {}, 0, ""
            books[sym] = (bb, ba)
            mids[sym] = (bb + ba) / 2.0

        basket_dev = sum(mids.values()) - self.BASKET_FAIR

        for sym in self.SYMS:
            bb, ba = books[sym]
            position = int(state.position.get(sym, 0))

            target = int(round(-self.K * basket_dev))
            target = max(-self.POS_LIMIT, min(self.POS_LIMIT, target))

            buy_room = self.POS_LIMIT - position
            sell_room = self.POS_LIMIT + position
            buy_signal_qty  = max(0, target - position)
            sell_signal_qty = max(0, position - target)
            buy_qty  = max(self.BASELINE_SIZE if buy_room  > 0 else 0, buy_signal_qty)
            sell_qty = max(self.BASELINE_SIZE if sell_room > 0 else 0, sell_signal_qty)
            buy_qty  = min(buy_qty,  self.MAX_TRADE_SIZE, buy_room)
            sell_qty = min(sell_qty, self.MAX_TRADE_SIZE, sell_room)

            orders: List[Order] = []
            if buy_qty > 0:
                bid_price = bb + self.OFFSET
                if bid_price >= ba:
                    bid_price = ba - 1
                if bid_price > bb:
                    orders.append(Order(sym, bid_price, +buy_qty))

            if sell_qty > 0:
                ask_price = ba - self.OFFSET
                if ask_price <= bb:
                    ask_price = bb + 1
                if ask_price < ba:
                    orders.append(Order(sym, ask_price, -sell_qty))

            if orders:
                result[sym] = orders

        return result, 0, ""

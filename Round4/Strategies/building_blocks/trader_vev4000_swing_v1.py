"""
VEV_4000-only swing trader, R4 v1.

The same intraday-swing concept as trader_ve_swing_v1.py, applied to
VEV_4000 instead of VE.

Why VEV_4000 should also work
-----------------------------
VEV_4000 is a delta-1 ITM call on VE: VEV_4000_mid ≈ VE_mid - 4000. So
whenever VE makes its end-of-day rally (avg +30-40 ticks across 3 days),
VEV_4000 should rally the same +30-40 ticks. With VEV's 300 position
limit (vs VE's 200), the maximum theoretical capture is bigger.

Caveats vs VE swing
-------------------
1. VEV_4000 has ~10x lower trade volume than VE. Mark 38 sells ~78
   times/day on VEV_4000. In a 200k-tick entry window we might only see
   ~16 sells × 2 lots ≈ 32 lots accumulated passively. Likely WAY less
   than the 300 limit.
2. VEV_4000 spread is 21 (vs VE's 6) — entry/exit cost is higher.

Net expected ($30-40 swing × 50-150 lots accumulated - spread costs)
should still produce a few thousand $/day on top of any other strategy.

Architecture: identical to trader_ve_swing_v1 with renamed product.
"""
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple


class Trader:
    SYM = "VEV_4000"
    LIMIT = 300

    # Phase boundaries (timestamps)
    ENTRY_START = 500_000
    ENTRY_END = 700_000
    EXIT_START = 900_000
    EXIT_FORCE = 970_000

    # Order sizing
    QUOTE_SIZE = 80

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {self.SYM: []}

        book = state.order_depths.get(self.SYM)
        if not self._book_ok(book):
            return result, 0, ""

        bb = max(book.buy_orders)
        ba = min(book.sell_orders)
        if bb >= ba:
            return result, 0, ""

        pos = int(state.position.get(self.SYM, 0))
        ts = int(state.timestamp)

        if ts < self.ENTRY_START:
            pass

        elif self.ENTRY_START <= ts < self.ENTRY_END:
            room = max(0, self.LIMIT - pos)
            if room > 0:
                size = min(self.QUOTE_SIZE, room)
                result[self.SYM].append(Order(self.SYM, bb + 1, +size))

        elif self.ENTRY_END <= ts < self.EXIT_START:
            pass

        elif self.EXIT_START <= ts < self.EXIT_FORCE:
            if pos > 0:
                size = min(self.QUOTE_SIZE, pos)
                result[self.SYM].append(Order(self.SYM, ba - 1, -size))

        else:
            if pos > 0:
                avail_at_bid = int(book.buy_orders[bb])
                size = min(pos, avail_at_bid)
                if size > 0:
                    result[self.SYM].append(Order(self.SYM, bb, -size))
            elif pos < 0:
                avail_at_ask = -int(book.sell_orders[ba])
                size = min(-pos, avail_at_ask)
                if size > 0:
                    result[self.SYM].append(Order(self.SYM, ba, +size))

        return result, 0, ""

    @staticmethod
    def _book_ok(book) -> bool:
        return book is not None and book.buy_orders and book.sell_orders

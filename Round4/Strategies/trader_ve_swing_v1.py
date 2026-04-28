"""
VE-only intraday swing trader, R4 v1.  *** FROZEN REFERENCE — DO NOT MODIFY ***

This is the verified-working baseline of the structural intraday-swing idea.
Backtest result on R4 dataset (3 days, v0.4.9 binary): $21,067 / 3 days = $7,022/day.
All future iterations should be saved as v2, v3, etc. Keep v1 as-is for
regression comparison.

Trades ONLY VELVETFRUIT_EXTRACT. No HG, no VEV.

Strategy
--------
3-day data shows VE has a structural intraday pattern across all 3 days:
  - bucket 8-12 (ts 400-650k):  mid avg ≈ 5228 (-22 deviation)
  - bucket 18-19 (ts 900-1000k): mid avg ≈ 5267 (+17 deviation)
  - average swing: +38 ticks from mid-day trough → end-of-day high
  - holds even on Day 3 where VE went -64 net for the day

Plan: long VE during the trough window, exit during the recovery window.

Phases (driven purely by state.timestamp):
  ts < 500k:           IDLE (just observe)
  ts 500k - 700k:      ACCUMULATE long up to 200 limit (passive bids at bb+1)
  ts 700k - 900k:      HOLD (no orders, ride the recovery)
  ts 900k - 970k:      UNWIND passively (asks at ba-1)
  ts >= 970k:          FORCE LIQUIDATE (cross spread to ensure flat by close)

If the structural pattern holds, expected PnL = 200 lots × ~38 ticks ≈ $7,600.

Architecture notes
------------------
- Single product, no traderData state needed (phase determined by timestamp)
- Phase boundaries are constants, easy to retune
- Position limit checked before every order
- Defensive empty-book handling
"""
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple


class Trader:
    VE_SYM = "VELVETFRUIT_EXTRACT"
    VE_LIMIT = 200

    # Phase boundaries (timestamps)
    ENTRY_START = 500_000
    ENTRY_END = 700_000
    EXIT_START = 900_000
    EXIT_FORCE = 970_000

    # Order sizes
    QUOTE_SIZE = 80     # passive quote size each tick during accumulate/unwind

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {self.VE_SYM: []}

        book = state.order_depths.get(self.VE_SYM)
        if not self._book_ok(book):
            return result, 0, ""

        bb = max(book.buy_orders)
        ba = min(book.sell_orders)
        if bb >= ba:
            return result, 0, ""

        pos = int(state.position.get(self.VE_SYM, 0))
        ts = int(state.timestamp)

        if ts < self.ENTRY_START:
            # Phase 1: IDLE — early in the day, before the trough sets in
            pass

        elif self.ENTRY_START <= ts < self.ENTRY_END:
            # Phase 2: ACCUMULATE LONG. Quote a passive bid 1 inside the
            # current best bid. Don't quote ask — we don't want to sell.
            room = max(0, self.VE_LIMIT - pos)
            if room > 0:
                size = min(self.QUOTE_SIZE, room)
                result[self.VE_SYM].append(Order(self.VE_SYM, bb + 1, +size))

        elif self.ENTRY_END <= ts < self.EXIT_START:
            # Phase 3: HOLD. No orders. Let VE drift (recover toward end-of-day high).
            pass

        elif self.EXIT_START <= ts < self.EXIT_FORCE:
            # Phase 4: PASSIVE UNWIND. Quote a passive ask 1 inside the best ask.
            if pos > 0:
                size = min(self.QUOTE_SIZE, pos)
                result[self.VE_SYM].append(Order(self.VE_SYM, ba - 1, -size))

        else:  # ts >= EXIT_FORCE
            # Phase 5: FORCE FLAT. Cross the spread on whatever inventory remains
            # so we close the day at zero (no overnight exposure).
            if pos > 0:
                avail_at_bid = int(book.buy_orders[bb])
                size = min(pos, avail_at_bid)
                if size > 0:
                    result[self.VE_SYM].append(Order(self.VE_SYM, bb, -size))
            elif pos < 0:
                avail_at_ask = -int(book.sell_orders[ba])
                size = min(-pos, avail_at_ask)
                if size > 0:
                    result[self.VE_SYM].append(Order(self.VE_SYM, ba, +size))

        return result, 0, ""

    @staticmethod
    def _book_ok(book) -> bool:
        return book is not None and book.buy_orders and book.sell_orders

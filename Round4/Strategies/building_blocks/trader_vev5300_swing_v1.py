"""
VEV_5300-only swing trader, R4 v1.

Same intraday-swing structure as trader_ve_swing_v1, applied to VEV_5300.
Mass-produced from generator — see trader_vev4000_swing_v1.py for design notes.
"""
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple


class Trader:
    SYM = "VEV_5300"
    LIMIT = 300

    ENTRY_START = 500_000
    ENTRY_END = 700_000
    EXIT_START = 900_000
    EXIT_FORCE = 970_000
    QUOTE_SIZE = 80

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {self.SYM: []}
        book = state.order_depths.get(self.SYM)
        if not (book and book.buy_orders and book.sell_orders):
            return result, 0, ""

        bb = max(book.buy_orders); ba = min(book.sell_orders)
        if bb >= ba:
            return result, 0, ""

        pos = int(state.position.get(self.SYM, 0))
        ts = int(state.timestamp)

        if ts < self.ENTRY_START:
            pass
        elif ts < self.ENTRY_END:
            room = max(0, self.LIMIT - pos)
            if room > 0:
                size = min(self.QUOTE_SIZE, room)
                result[self.SYM].append(Order(self.SYM, bb + 1, +size))
        elif ts < self.EXIT_START:
            pass
        elif ts < self.EXIT_FORCE:
            if pos > 0:
                size = min(self.QUOTE_SIZE, pos)
                result[self.SYM].append(Order(self.SYM, ba - 1, -size))
        else:
            if pos > 0:
                avail = int(book.buy_orders[bb])
                size = min(pos, avail)
                if size > 0:
                    result[self.SYM].append(Order(self.SYM, bb, -size))
            elif pos < 0:
                avail = -int(book.sell_orders[ba])
                size = min(-pos, avail)
                if size > 0:
                    result[self.SYM].append(Order(self.SYM, ba, +size))

        return result, 0, ""

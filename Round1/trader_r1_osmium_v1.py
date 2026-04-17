"""
IMC Prosperity 4 — Round 1, OSMIUM-only trader v1.

Scope: ASH_COATED_OSMIUM only. No PEPPER logic. If INTARIAN_PEPPER_ROOT
appears in the state, we simply don't trade it.

Strategy (based on the team's findings):

  1. Mean-reverting — no drift to chase, no directional target.
  2. Tick-level return autocorr is strongly negative (≈ -0.5). Prices bounce,
     so sitting one tick inside the touch captures the bounce repeatedly.
  3. frontier_imbalance (level-1 bid_vol vs ask_vol) is the strongest
     short-horizon signal we found — use it to tilt our quotes.
  4. Rolling-mean "fair value" MM underperformed naive quote-inside MM in
     the teammate's test. So we drop the rolling-mean FV entirely.

Design:

  * Passive only. No aggressive take in v1 (save that for v2 after we
    confirm passive PnL on backtest).
  * Reference centre = touch_mid + K_IMB * imbalance - K_INV * position.
      - imbalance tilt leans our quotes toward the side with pressure,
        which is where the next tick is more likely to go.
      - inventory tilt pulls our centre away from neutral when we build
        a position, naturally mean-reverting our book back to zero.
  * Quotes sit at centre ± half_spread, clamped to be one tick inside the
    touch so we sit at the top of the queue.
  * Hard safety: skip when the book is one-sided, touch spread is too
    tight, or our own quotes would cross. Also a soft position cap < 80
    to avoid the "stuck at limit" artefact the teammate flagged.
  * Lightweight diagnostics written to traderData so we can plot what
    the strategy actually did after a backtest.
"""

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    PRODUCT = "ASH_COATED_OSMIUM"
    POSITION_LIMIT = 80

    # Reference-centre tilts
    K_IMB = 2.0         # ticks of centre shift per unit of frontier imbalance
    K_INV = 0.12        # ticks of centre shift per unit of position (negated)

    # Quote shape
    BASE_HALF_SPREAD = 3      # half-spread around centre, in ticks
    ORDER_SIZE = 12           # max size per side

    # Safety
    MIN_TOUCH_SPREAD = 3      # need this much spread to quote inside
    SOFT_POS_LIMIT = 60       # buffer vs hard 80 to avoid order rejects
    MAX_QUOTE_OFFSET = 8      # clamp |quote - touch| per side

    # ==================================================================
    # Entry point
    # ==================================================================
    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        result: Dict[str, List[Order]] = {}

        order_depth = state.order_depths.get(self.PRODUCT)
        if order_depth is not None:
            result[self.PRODUCT] = self._trade_osmium(state, order_depth, data)

        trader_data = self._dump_data(data)
        conversions = 0
        return result, conversions, trader_data

    # ==================================================================
    # OSMIUM strategy
    # ==================================================================
    def _trade_osmium(
        self,
        state: TradingState,
        order_depth: OrderDepth,
        data: Dict,
    ) -> List[Order]:
        position = state.position.get(self.PRODUCT, 0)
        orders: List[Order] = []

        # ---------- 1. book sanity ----------
        best_bid, best_ask = self._best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return orders  # one-sided book, skip

        touch_spread = best_ask - best_bid
        if touch_spread < self.MIN_TOUCH_SPREAD:
            return orders  # not enough room to quote inside

        touch_mid = (best_bid + best_ask) / 2.0

        # ---------- 2. frontier imbalance ----------
        bid_vol = order_depth.buy_orders.get(best_bid, 0)
        ask_vol = abs(order_depth.sell_orders.get(best_ask, 0))
        denom = bid_vol + ask_vol
        imbalance = (bid_vol - ask_vol) / denom if denom > 0 else 0.0

        # ---------- 3. reference centre ----------
        imb_tilt = self.K_IMB * imbalance
        inv_tilt = -self.K_INV * position
        centre = touch_mid + imb_tilt + inv_tilt

        # ---------- 4. quotes, clamped inside the touch ----------
        raw_bid = int(math.floor(centre - self.BASE_HALF_SPREAD))
        raw_ask = int(math.ceil(centre + self.BASE_HALF_SPREAD))

        # snap inside the touch (top of queue on both sides)
        bid_q = max(best_bid + 1, min(raw_bid, best_ask - 1))
        ask_q = min(best_ask - 1, max(raw_ask, best_bid + 1))

        # don't drift too far from the touch on a tilt spike
        bid_q = max(best_bid - self.MAX_QUOTE_OFFSET,
                    min(bid_q, best_bid + self.MAX_QUOTE_OFFSET))
        ask_q = max(best_ask - self.MAX_QUOTE_OFFSET,
                    min(ask_q, best_ask + self.MAX_QUOTE_OFFSET))

        if bid_q >= ask_q:
            return orders  # tilts collapsed our spread — skip this tick

        # ---------- 5. size with soft + hard position guards ----------
        hard_buy_cap  = self.POSITION_LIMIT - position
        hard_sell_cap = self.POSITION_LIMIT + position
        soft_buy_cap  = max(0, self.SOFT_POS_LIMIT - position)
        soft_sell_cap = max(0, self.SOFT_POS_LIMIT + position)

        buy_qty  = min(self.ORDER_SIZE, hard_buy_cap,  soft_buy_cap)
        sell_qty = min(self.ORDER_SIZE, hard_sell_cap, soft_sell_cap)

        if buy_qty > 0:
            orders.append(Order(self.PRODUCT, int(bid_q), int(buy_qty)))
        if sell_qty > 0:
            orders.append(Order(self.PRODUCT, int(ask_q), int(-sell_qty)))

        # ---------- 6. diagnostics via traderData ----------
        diag = data.setdefault("diag", {})
        diag["ts"]       = state.timestamp
        diag["position"] = position
        diag["touch"]    = [best_bid, best_ask]
        diag["imb"]      = round(imbalance, 3)
        diag["centre"]   = round(centre, 2)
        diag["quotes"]   = [bid_q, ask_q]

        return orders

    # ==================================================================
    # Helpers
    # ==================================================================
    def _best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def _load_data(self, trader_data: str) -> Dict:
        if not trader_data:
            return {"diag": {}}
        try:
            data = json.loads(trader_data)
            data.setdefault("diag", {})
            return data
        except Exception:
            return {"diag": {}}

    def _dump_data(self, data: Dict) -> str:
        compact = {"diag": data.get("diag", {})}
        return json.dumps(compact, separators=(",", ":"))

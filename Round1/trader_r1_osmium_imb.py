"""
IMC Prosperity 4 — Round 1, OSMIUM trader with imbalance tilt (route A).

Built on top of `trader_r1_osmium.py` (the clean OSMIUM-only baseline).

Single-line change vs baseline:

    reservation_price = fair - INVENTORY_SKEW * position
                             + IMB_TILT * frontier_imbalance   <-- NEW

Rationale:
  - frontier_imbalance = (bid_vol_1 - ask_vol_1) / (bid_vol_1 + ask_vol_1)
  - In our 3-day historical analysis (analysis_v2.ipynb, Layer 3a / Interaction B)
    this was consistently the strongest single-tick predictor of forward price
    change AND of the next trade direction.
  - Shifting the reservation price in the direction of imbalance means:
      * imb > 0 (bid-heavy, upside pressure) -> reservation up
          -> TAKE more willing to buy (higher max buy price)
          -> TAKE less willing to sell (higher min sell price)
          -> MAKE quotes shift up (if clamp allows)
      * imb < 0 (ask-heavy, downside pressure) -> reservation down (mirror).
  - This is strictly additive to the existing FV + inventory skew. No other
    parameter is touched: TAKE_EDGE, QUOTE_EDGE, ORDER_SIZE, HISTORY_LEN,
    INVENTORY_SKEW all identical to baseline. One new parameter (IMB_TILT),
    chosen by theory (~1 tick shift on a strong-but-not-extreme imbalance),
    not by parameter search. Overfit risk is minimized.
"""

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    PRODUCT = "ASH_COATED_OSMIUM"
    POSITION_LIMIT = 80

    # Baseline params — unchanged
    ASH_HISTORY_LEN = 40
    ASH_TAKE_EDGE = 3
    ASH_QUOTE_EDGE = 2
    ASH_ORDER_SIZE = 12
    ASH_INVENTORY_SKEW = 0.05

    # NEW in this version
    ASH_IMB_TILT = 1.0   # ticks of reservation shift per unit of imbalance

    # ==================================================================
    # Entry point
    # ==================================================================
    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        result: Dict[str, List[Order]] = {}

        order_depth = state.order_depths.get(self.PRODUCT)
        if order_depth is not None:
            result[self.PRODUCT] = self._trade_ash_osmium(state, order_depth, data)

        trader_data = self._dump_data(data)
        conversions = 0
        return result, conversions, trader_data

    def _trade_ash_osmium(
        self,
        state: TradingState,
        order_depth: OrderDepth,
        data: Dict,
    ) -> List[Order]:
        product = self.PRODUCT
        limit = self.POSITION_LIMIT
        position = state.position.get(product, 0)
        orders: List[Order] = []

        best_bid, best_ask = self._best_bid_ask(order_depth)
        mid = self._mid_price(order_depth)
        if mid is None:
            return orders

        # --- rolling-mean fair value (unchanged) ---
        ash_mids = data.get("ash_mids", [])
        ash_mids.append(mid)
        if len(ash_mids) > self.ASH_HISTORY_LEN:
            ash_mids = ash_mids[-self.ASH_HISTORY_LEN :]
        data["ash_mids"] = ash_mids
        fair = sum(ash_mids) / len(ash_mids)

        # --- NEW: frontier imbalance at the touch ---
        imbalance = 0.0
        if best_bid is not None and best_ask is not None:
            bid_vol = order_depth.buy_orders.get(best_bid, 0)
            ask_vol = abs(order_depth.sell_orders.get(best_ask, 0))
            denom = bid_vol + ask_vol
            if denom > 0:
                imbalance = (bid_vol - ask_vol) / denom

        # --- reservation price: baseline + imbalance tilt ---
        reservation_price = (
            fair
            - self.ASH_INVENTORY_SKEW * position
            + self.ASH_IMB_TILT * imbalance
        )

        # --- Phase 1: take (unchanged formulas) ---
        buy_capacity = limit - position
        filled_buy = self._buy_through(
            product=product,
            order_depth=order_depth,
            max_price=math.floor(reservation_price - self.ASH_TAKE_EDGE),
            quantity=buy_capacity,
            orders=orders,
        )
        position += filled_buy

        sell_capacity = limit + position
        filled_sell = self._sell_through(
            product=product,
            order_depth=order_depth,
            min_price=math.ceil(reservation_price + self.ASH_TAKE_EDGE),
            quantity=sell_capacity,
            orders=orders,
        )
        position -= filled_sell

        buy_capacity = limit - position
        sell_capacity = limit + position

        # --- Phase 2: make (unchanged formulas) ---
        if best_bid is not None and best_ask is not None and best_bid < best_ask:
            bid_quote = min(best_bid + 1, math.floor(reservation_price - self.ASH_QUOTE_EDGE))
            ask_quote = max(best_ask - 1, math.ceil(reservation_price + self.ASH_QUOTE_EDGE))

            if bid_quote < ask_quote:
                buy_qty = min(self.ASH_ORDER_SIZE, buy_capacity)
                sell_qty = min(self.ASH_ORDER_SIZE, sell_capacity)

                if buy_qty > 0:
                    orders.append(Order(product, int(bid_quote), int(buy_qty)))
                if sell_qty > 0:
                    orders.append(Order(product, int(ask_quote), int(-sell_qty)))

        return orders

    # ==================================================================
    # Helpers — unchanged
    # ==================================================================
    def _buy_through(
        self,
        product: str,
        order_depth: OrderDepth,
        max_price: float,
        quantity: int,
        orders: List[Order],
    ) -> int:
        remaining = int(quantity)
        filled = 0
        for ask_price in sorted(order_depth.sell_orders.keys()):
            ask_volume = abs(int(order_depth.sell_orders[ask_price]))
            if ask_price > max_price or remaining <= 0:
                break
            take = min(remaining, ask_volume)
            if take > 0:
                orders.append(Order(product, int(ask_price), int(take)))
                filled += take
                remaining -= take
        return filled

    def _sell_through(
        self,
        product: str,
        order_depth: OrderDepth,
        min_price: float,
        quantity: int,
        orders: List[Order],
    ) -> int:
        remaining = int(quantity)
        filled = 0
        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            bid_volume = abs(int(order_depth.buy_orders[bid_price]))
            if bid_price < min_price or remaining <= 0:
                break
            take = min(remaining, bid_volume)
            if take > 0:
                orders.append(Order(product, int(bid_price), int(-take)))
                filled += take
                remaining -= take
        return filled

    def _best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def _mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask = self._best_bid_ask(order_depth)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return None

    def _load_data(self, trader_data: str) -> Dict:
        if not trader_data:
            return {"ash_mids": []}
        try:
            data = json.loads(trader_data)
            data.setdefault("ash_mids", [])
            return data
        except Exception:
            return {"ash_mids": []}

    def _dump_data(self, data: Dict) -> str:
        compact = {"ash_mids": data.get("ash_mids", [])[-self.ASH_HISTORY_LEN :]}
        return json.dumps(compact, separators=(",", ":"))

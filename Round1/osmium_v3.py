"""
IMC Prosperity 4 — Round 1, OSMIUM-only extract of submission 183008
(plus one structural improvement: one-sided book fallback).

Provenance:
  Extracted from /Users/nickzhu/Downloads/183008/183008.py (a Round 1
  submission downloaded from the Prosperity platform). All
  INTARIAN_PEPPER_ROOT code paths removed.

Strategy summary ("cautious wall-mid market maker"):

  1. Fair value = wall_mid + 0.25 * imbalance1
       - wall_mid = midpoint of OUTERMOST visible bid and ask
         (not touch_mid, not a rolling mean)
       - imbalance1 = top-of-book (bid_vol_1 - ask_vol_1) / total
  2. Three-phase tick loop:
       a. TAKE    — lift any ask <= fv - 2; hit any bid >= fv + 2;
                    plus free short-cover / long-unwind at fv
       b. FLATTEN — if |position| > 20, place a single flatten order at fv
       c. MAKE    — passive quotes one tick inside walls AND one tick inside
                    touch, capped so we never quote through fv; size scales
                    down as inventory pressure grows
  3. Safety rails:
       - refuse to quote when wall_width < 6
       - bid() / ask() helpers internally track per-side budget so you
         cannot place orders that would breach the position limit even
         across multiple calls within the same tick
       - _finalise_orders() is a belt-and-braces final clip, shrinking
         tail orders from the end if anything somehow slipped through
       - errors are counted and persisted via traderData

One-sided book fallback (NEW vs 183008):
  In the sample log (1000-tick sample run), 8.6% of ACO ticks had only one
  side of the book populated. The original strategy skipped those ticks
  entirely. We now handle them:

    - Use the LAST-KNOWN wall_mid (persisted each tick as `_prev_wall_mid`,
      which the original already wrote but never read on ACO) as the fair
      value anchor.
    - Post ONE passive quote on the side that still has a book, sized at
      half of base_quote_size. TAKE and FLATTEN are both skipped in this
      regime: without a full book we can't classify aggressive-take edges
      or price a flatten order safely.
    - The "never quote through fv" guard still applies.

  The fallback is disabled until the first full-book tick has seeded a
  usable `_prev_wall_mid` — no guessing from thin air.

Self-imposed position limit is 50 (half the official 80), matching the
original submission.
"""

import json
import math
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState


ASH = "ASH_COATED_OSMIUM"

POS_LIMIT: int = 50

CONFIG = {
    "take_edge": 2,            # only cross when clearly favorable
    "soft_inventory": 20,      # start flattening early
    "base_quote_size": 8,      # modest size
    "min_wall_width": 6,       # do not quote when book is too tight/odd
    "imbalance_weight": 0.25,  # tiny skew only
}


class ProductTrader:
    def __init__(self, name: str, state: TradingState, new_trader_data: dict):
        self.name = name
        self.state = state
        self.new_trader_data = new_trader_data
        self.orders: List[Order] = []

        self.last_data = self._get_last_trader_data()

        self.position_limit = POS_LIMIT
        self.initial_position = self.state.position.get(self.name, 0)

        self.mkt_buy_orders, self.mkt_sell_orders = self._get_order_depth()
        self.best_bid, self.best_ask = self._get_best_bid_ask()
        self.bid_wall, self.wall_mid, self.ask_wall = self._get_walls()
        self.wall_width = (
            None
            if self.bid_wall is None or self.ask_wall is None
            else self.ask_wall - self.bid_wall
        )

        self.max_allowed_buy_volume = self.position_limit - self.initial_position
        self.max_allowed_sell_volume = self.position_limit + self.initial_position

    # ---------- persistence / safety ----------
    def _record_error(self, where: str, exc: Exception) -> None:
        errs = self.new_trader_data.setdefault("errors", {})
        key = f"{self.name}:{where}:{type(exc).__name__}"
        errs[key] = errs.get(key, 0) + 1

    def _get_last_trader_data(self) -> dict:
        try:
            if self.state.traderData:
                return json.loads(self.state.traderData)
        except Exception as exc:
            self._record_error("load_traderData", exc)
        return {}

    def _get_order_depth(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        buy_orders: Dict[int, int] = {}
        sell_orders: Dict[int, int] = {}
        try:
            order_depth: OrderDepth = self.state.order_depths[self.name]
            buy_orders = {
                int(p): abs(int(v))
                for p, v in sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
            }
            sell_orders = {
                int(p): abs(int(v))
                for p, v in sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
            }
        except Exception as exc:
            self._record_error("order_depth", exc)
        return buy_orders, sell_orders

    def _get_best_bid_ask(self) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(self.mkt_buy_orders.keys()) if self.mkt_buy_orders else None
        best_ask = min(self.mkt_sell_orders.keys()) if self.mkt_sell_orders else None
        return best_bid, best_ask

    def _get_walls(self) -> Tuple[Optional[int], Optional[float], Optional[int]]:
        bid_wall = min(self.mkt_buy_orders.keys()) if self.mkt_buy_orders else None
        ask_wall = max(self.mkt_sell_orders.keys()) if self.mkt_sell_orders else None
        wall_mid = None if bid_wall is None or ask_wall is None else (bid_wall + ask_wall) / 2.0
        return bid_wall, wall_mid, ask_wall

    # ---------- order helpers ----------
    def bid(self, price: int, volume: int) -> None:
        v = min(abs(int(volume)), self.max_allowed_buy_volume)
        if v <= 0:
            return
        self.max_allowed_buy_volume -= v
        self.orders.append(Order(self.name, int(price), v))

    def ask(self, price: int, volume: int) -> None:
        v = min(abs(int(volume)), self.max_allowed_sell_volume)
        if v <= 0:
            return
        self.max_allowed_sell_volume -= v
        self.orders.append(Order(self.name, int(price), -v))

    def _finalise_orders(self) -> List[Order]:
        """
        Extra guard: if something went wrong, clip tail orders so total
        submitted volume on each side cannot breach limits.
        """
        pos = self.initial_position
        limit = self.position_limit

        def clip_side(sign: int) -> None:
            budget = (limit - pos) if sign > 0 else (limit + pos)
            budget = max(budget, 0)

            total = sum(
                o.quantity if sign > 0 else -o.quantity
                for o in self.orders
                if (o.quantity > 0 if sign > 0 else o.quantity < 0)
            )
            if total <= budget:
                return

            overflow = total - budget
            for i in range(len(self.orders) - 1, -1, -1):
                o = self.orders[i]
                if (sign > 0 and o.quantity > 0) or (sign < 0 and o.quantity < 0):
                    qty = o.quantity if sign > 0 else -o.quantity
                    if qty <= overflow:
                        overflow -= qty
                        self.orders[i] = Order(o.symbol, o.price, 0)
                    else:
                        new_qty = qty - overflow
                        overflow = 0
                        self.orders[i] = Order(
                            o.symbol, o.price, new_qty if sign > 0 else -new_qty
                        )
                    if overflow == 0:
                        break

        clip_side(+1)
        clip_side(-1)
        self.orders = [o for o in self.orders if o.quantity != 0]
        return self.orders

    # ---------- signals ----------
    def _imbalance1(self) -> float:
        if self.best_bid is None or self.best_ask is None:
            return 0.0
        bv = self.mkt_buy_orders.get(self.best_bid, 0)
        av = self.mkt_sell_orders.get(self.best_ask, 0)
        den = bv + av
        if den <= 0:
            return 0.0
        return (bv - av) / den

    def _estimate_fair_value(self) -> Optional[float]:
        if self.wall_mid is None:
            return None

        fv = self.wall_mid

        # Small top-of-book imbalance skew
        imb1 = self._imbalance1()
        fv += CONFIG["imbalance_weight"] * imb1

        return fv

    def _flatten_inventory(self, fv: float) -> None:
        soft = CONFIG["soft_inventory"]

        bid_flatten_price = int(math.floor(fv))
        ask_flatten_price = int(math.ceil(fv))

        if self.initial_position > soft:
            self.ask(ask_flatten_price, self.initial_position - soft)

        if self.initial_position < -soft:
            self.bid(bid_flatten_price, abs(self.initial_position) - soft)

    def get_orders(self) -> Dict[str, List[Order]]:
        raise NotImplementedError


class Round1Trader(ProductTrader):
    def _one_sided_make(self, fv: float) -> None:
        """
        Post a passive quote on whichever side(s) of the book still exist.
        Called when wall_mid is None (at least one side is empty); uses the
        last-known wall_mid (passed in as `fv`) as the reference price. No
        TAKE and no FLATTEN in this regime.

        Size is half of base_quote_size. The "never quote through fv" guard
        is preserved exactly.
        """
        half_size = max(1, CONFIG["base_quote_size"] // 2)

        if self.best_bid is not None:
            bid_price = int(self.best_bid + 1)
            if bid_price >= fv:
                bid_price = int(math.floor(fv - 1))
            if bid_price < fv:
                self.bid(bid_price, half_size)

        if self.best_ask is not None:
            ask_price = int(self.best_ask - 1)
            if ask_price <= fv:
                ask_price = int(math.ceil(fv + 1))
            if ask_price > fv:
                self.ask(ask_price, half_size)

    def get_orders(self) -> Dict[str, List[Order]]:
        # ------------------------------------------------
        # Book missing / one-sided -> fallback branch
        # ------------------------------------------------
        # Both sides empty: truly nothing we can do, even with history.
        if self.best_bid is None and self.best_ask is None:
            return {self.name: self._finalise_orders()}

        # One side empty: fall back to last-known wall_mid and post only on
        # the side that still has a book. Needs a history seed.
        if self.wall_mid is None:
            prev_wm = self.last_data.get(f"{self.name}_prev_wall_mid")
            if prev_wm is not None:
                self._one_sided_make(float(prev_wm))
            # Do NOT overwrite prev_wall_mid with None — keep the last
            # valid reading alive for future one-sided ticks.
            return {self.name: self._finalise_orders()}

        # Full book present but too thin to be trustworthy.
        if self.wall_width is None or self.wall_width < CONFIG["min_wall_width"]:
            self.new_trader_data[f"{self.name}_prev_wall_mid"] = self.wall_mid
            return {self.name: self._finalise_orders()}

        fv = self._estimate_fair_value()
        if fv is None:
            return {self.name: self._finalise_orders()}

        take_edge = CONFIG["take_edge"]

        # ------------------------------------------------
        # 1) TAKE phase
        # ------------------------------------------------
        # Buy clearly cheap asks
        for ask_price, ask_vol in self.mkt_sell_orders.items():
            if ask_price <= fv - take_edge:
                self.bid(ask_price, ask_vol)
            elif ask_price <= fv and self.initial_position < 0:
                # free/cheap short-cover
                cover = min(ask_vol, abs(self.initial_position))
                self.bid(ask_price, cover)
            else:
                break

        # Sell clearly rich bids
        for bid_price, bid_vol in self.mkt_buy_orders.items():
            if bid_price >= fv + take_edge:
                self.ask(bid_price, bid_vol)
            elif bid_price >= fv and self.initial_position > 0:
                unwind = min(bid_vol, self.initial_position)
                self.ask(bid_price, unwind)
            else:
                break

        # ------------------------------------------------
        # 2) Flatten overlay
        # ------------------------------------------------
        self._flatten_inventory(fv)

        # ------------------------------------------------
        # 3) MAKE phase (cautious)
        # ------------------------------------------------
        # Start from walls
        bid_price = int(self.bid_wall + 1)
        ask_price = int(self.ask_wall - 1)

        # Improve to current best by one tick if possible
        if self.best_bid is not None:
            bid_price = max(bid_price, int(self.best_bid + 1))
        if self.best_ask is not None:
            ask_price = min(ask_price, int(self.best_ask - 1))

        # Never quote through fair value
        if bid_price >= fv:
            bid_price = min(bid_price, int(math.floor(fv - 1)))
        if ask_price <= fv:
            ask_price = max(ask_price, int(math.ceil(fv + 1)))

        # If the resulting quote is invalid/crossed, skip that side
        post_bid = bid_price < ask_price and bid_price < fv
        post_ask = bid_price < ask_price and ask_price > fv

        # Inventory-aware sizing
        base_size = CONFIG["base_quote_size"]

        long_pressure = max(self.initial_position, 0) / max(CONFIG["soft_inventory"], 1)
        short_pressure = max(-self.initial_position, 0) / max(CONFIG["soft_inventory"], 1)

        bid_size = max(1, int(round(base_size * max(0.25, 1.0 - long_pressure))))
        ask_size = max(1, int(round(base_size * max(0.25, 1.0 - short_pressure))))

        if post_bid:
            self.bid(bid_price, bid_size)

        if post_ask:
            self.ask(ask_price, ask_size)

        # Persist minimal state
        self.new_trader_data[f"{self.name}_prev_wall_mid"] = self.wall_mid

        return {self.name: self._finalise_orders()}


class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        new_trader_data: Dict = {}

        try:
            if state.traderData:
                prev = json.loads(state.traderData)
                if isinstance(prev, dict):
                    # carry forward all simple scalar state + errors
                    for k, v in prev.items():
                        new_trader_data[k] = v
        except Exception:
            new_trader_data = {}

        if ASH in state.order_depths:
            trader = Round1Trader(ASH, state, new_trader_data)
            result.update(trader.get_orders())

        try:
            trader_data = json.dumps(new_trader_data)
        except Exception:
            trader_data = ""

        return result, 0, trader_data

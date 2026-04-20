from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Optional, Tuple
import json
import math


ASH = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"

ASH_POS_LIMIT = 80   # changed from 50 to 80 for Round 2

ASH_CONFIG = {
    "take_edge": 4,
    "soft_inventory": 20,
    "base_quote_size": 8,
    "min_wall_width": 6,
    "imbalance_weight": 0.25,

    # 3-layer flattening only
    "flatten_zone2": 20,
    "flatten_zone3": 40,
}


class ProductTrader:
    def __init__(self, name: str, state: TradingState, new_trader_data: dict):
        self.name = name
        self.state = state
        self.new_trader_data = new_trader_data
        self.orders: List[Order] = []

        self.last_data = self._get_last_trader_data()

        self.position_limit = ASH_POS_LIMIT
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
        fv += ASH_CONFIG["imbalance_weight"] * self._imbalance1()
        return fv

    def _flatten_inventory(self, fv: float) -> None:
        """
        3-layer flattening only:

        Layer 1: |pos| <= flatten_zone2
            no explicit flatten

        Layer 2: flatten_zone2 < |pos| <= flatten_zone3
            flatten passively near fair, down to flatten_zone2

        Layer 3: |pos| > flatten_zone3
            flatten aggressively through fair, down to flatten_zone2
        """
        zone2 = ASH_CONFIG["flatten_zone2"]
        zone3 = ASH_CONFIG["flatten_zone3"]

        pos = self.initial_position
        abs_pos = abs(pos)

        if abs_pos <= zone2:
            return

        if pos > 0:
            if abs_pos <= zone3:
                # layer 2: passive flatten down to zone2
                flatten_px = int(math.ceil(fv))
                flatten_qty = pos - zone2
                self.ask(flatten_px, flatten_qty)
            else:
                # layer 3: aggressive flatten down to zone2
                flatten_px = int(math.floor(fv))
                flatten_qty = pos - zone2
                self.ask(flatten_px, flatten_qty)

        elif pos < 0:
            if abs_pos <= zone3:
                # layer 2: passive flatten down to zone2
                flatten_px = int(math.floor(fv))
                flatten_qty = abs(pos) - zone2
                self.bid(flatten_px, flatten_qty)
            else:
                # layer 3: aggressive flatten down to zone2
                flatten_px = int(math.ceil(fv))
                flatten_qty = abs(pos) - zone2
                self.bid(flatten_px, flatten_qty)


class Round1AshTrader(ProductTrader):
    def _one_sided_make(self, fv: float) -> None:
        half_size = max(1, ASH_CONFIG["base_quote_size"] // 2)

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
        if self.best_bid is None and self.best_ask is None:
            return {self.name: self._finalise_orders()}

        if self.wall_mid is None:
            prev_wm = self.last_data.get(f"{self.name}_prev_wall_mid")
            if prev_wm is not None:
                self._one_sided_make(float(prev_wm))
            return {self.name: self._finalise_orders()}

        if self.wall_width is None or self.wall_width < ASH_CONFIG["min_wall_width"]:
            self.new_trader_data[f"{self.name}_prev_wall_mid"] = self.wall_mid
            return {self.name: self._finalise_orders()}

        fv = self._estimate_fair_value()
        if fv is None:
            return {self.name: self._finalise_orders()}

        take_edge = ASH_CONFIG["take_edge"]

        for ask_price, ask_vol in self.mkt_sell_orders.items():
            if ask_price <= fv - take_edge:
                self.bid(ask_price, ask_vol)
            elif ask_price <= fv and self.initial_position < 0:
                cover = min(ask_vol, abs(self.initial_position))
                self.bid(ask_price, cover)
            else:
                break

        for bid_price, bid_vol in self.mkt_buy_orders.items():
            if bid_price >= fv + take_edge:
                self.ask(bid_price, bid_vol)
            elif bid_price >= fv and self.initial_position > 0:
                unwind = min(bid_vol, self.initial_position)
                self.ask(bid_price, unwind)
            else:
                break

        self._flatten_inventory(fv)

        bid_price = int(self.bid_wall + 1)
        ask_price = int(self.ask_wall - 1)

        if self.best_bid is not None:
            bid_price = max(bid_price, int(self.best_bid + 1))
        if self.best_ask is not None:
            ask_price = min(ask_price, int(self.best_ask - 1))

        if bid_price >= fv:
            bid_price = min(bid_price, int(math.floor(fv - 1)))
        if ask_price <= fv:
            ask_price = max(ask_price, int(math.ceil(fv + 1)))

        post_bid = bid_price < ask_price and bid_price < fv
        post_ask = bid_price < ask_price and ask_price > fv

        base_size = ASH_CONFIG["base_quote_size"]
        long_pressure = max(self.initial_position, 0) / max(ASH_CONFIG["soft_inventory"], 1)
        short_pressure = max(-self.initial_position, 0) / max(ASH_CONFIG["soft_inventory"], 1)

        bid_size = max(1, int(round(base_size * max(0.25, 1.0 - long_pressure))))
        ask_size = max(1, int(round(base_size * max(0.25, 1.0 - short_pressure))))

        if post_bid:
            self.bid(bid_price, bid_size)

        if post_ask:
            self.ask(ask_price, ask_size)

        self.new_trader_data[f"{self.name}_prev_wall_mid"] = self.wall_mid
        return {self.name: self._finalise_orders()}


class Trader:
    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    PEPPER_SLOPE = 0.00099
    PEPPER_ENTRY_BONUS_EARLY = 12
    PEPPER_ENTRY_BONUS_LATE = 6
    PEPPER_PASSIVE_SIZE = 20
    PEPPER_SCALP_EDGE = 7
    PEPPER_SCALP_SIZE = 10
    PEPPER_ANCHOR_ALPHA = 0.05

    def bid(self) -> int:
        return 15

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        result: Dict[str, List[Order]] = {}

        if PEPPER in state.order_depths:
            result[PEPPER] = self._trade_pepper_root(
                state,
                state.order_depths[PEPPER],
                data,
            )

        if ASH in state.order_depths:
            ash_trader = Round1AshTrader(ASH, state, data)
            result.update(ash_trader.get_orders())

        trader_data = self._dump_data(data)
        conversions = 0
        return result, conversions, trader_data

    def _trade_pepper_root(
        self,
        state: TradingState,
        order_depth: OrderDepth,
        data: Dict,
    ) -> List[Order]:
        product = PEPPER
        limit = self.POSITION_LIMITS[product]
        position = state.position.get(product, 0)
        orders: List[Order] = []

        best_bid, best_ask = self._best_bid_ask(order_depth)
        mid = self._mid_price(order_depth)
        trade_price = self._latest_trade_price(state, product)

        if trade_price is not None:
            data["pepper_last_trade_price"] = trade_price
        else:
            trade_price = data.get("pepper_last_trade_price")

        if mid is not None and trade_price is not None:
            price_obs = max(mid, trade_price)
        elif trade_price is not None:
            price_obs = trade_price
        else:
            price_obs = mid

        if price_obs is None:
            return orders

        anchor_obs = price_obs - self.PEPPER_SLOPE * state.timestamp
        anchor = data.get("pepper_anchor")
        if anchor is None or state.timestamp == 0:
            anchor = anchor_obs
        else:
            anchor = (
                (1.0 - self.PEPPER_ANCHOR_ALPHA) * float(anchor)
                + self.PEPPER_ANCHOR_ALPHA * anchor_obs
            )
        data["pepper_anchor"] = anchor

        fair = anchor + self.PEPPER_SLOPE * state.timestamp
        early = state.timestamp <= 5_000
        entry_premium = (
            self.PEPPER_ENTRY_BONUS_EARLY
            if early
            else self.PEPPER_ENTRY_BONUS_LATE
        )

        target_position = limit

        if position < target_position:
            buy_capacity = target_position - position
            max_buy_price = fair + entry_premium
            filled = self._buy_through(
                product=product,
                order_depth=order_depth,
                max_price=max_buy_price,
                quantity=buy_capacity,
                orders=orders,
            )
            position += filled

        if position > 40:
            sellable = min(self.PEPPER_SCALP_SIZE, position - 40)
            filled = self._sell_through(
                product=product,
                order_depth=order_depth,
                min_price=fair + self.PEPPER_SCALP_EDGE,
                quantity=sellable,
                orders=orders,
            )
            position -= filled

        if position < target_position and best_bid is not None:
            quote_bid = min(best_bid + 1, math.floor(fair + 4))
            if best_ask is None or quote_bid < best_ask:
                quote_qty = min(self.PEPPER_PASSIVE_SIZE, target_position - position)
                if quote_qty > 0:
                    orders.append(Order(product, int(quote_bid), int(quote_qty)))

        return orders

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

    def _latest_trade_price(self, state: TradingState, product: str) -> Optional[float]:
        trades = state.market_trades.get(product, [])
        if not trades:
            return None
        latest_trade = max(trades, key=lambda tr: tr.timestamp)
        return float(latest_trade.price)

    def _load_data(self, trader_data: str) -> Dict:
        if not trader_data:
            return {
                "pepper_anchor": None,
                "pepper_last_trade_price": None,
            }
        try:
            data = json.loads(trader_data)
            if "pepper_anchor" not in data:
                data["pepper_anchor"] = None
            if "pepper_last_trade_price" not in data:
                data["pepper_last_trade_price"] = None
            return data
        except Exception:
            return {
                "pepper_anchor": None,
                "pepper_last_trade_price": None,
            }

    def _dump_data(self, data: Dict) -> str:
        compact = {
            "pepper_anchor": data.get("pepper_anchor"),
            "pepper_last_trade_price": data.get("pepper_last_trade_price"),
        }

        ash_prev_key = f"{ASH}_prev_wall_mid"
        if ash_prev_key in data:
            compact[ash_prev_key] = data.get(ash_prev_key)

        if "errors" in data:
            compact["errors"] = data.get("errors")

        return json.dumps(compact, separators=(",", ":"))
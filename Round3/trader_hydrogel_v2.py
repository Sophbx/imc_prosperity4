"""
HYDROGEL_PACK passive market maker (v2).

Same archetype as v1 (passive MM, improve by 1 over the touch, depth-imbalance
fade, end-of-day flatten), but with **tighter inventory bands**:

  v1: skew_scale=100, unwind=120, hard=180 (60% / 90% of position limit)
  v2: skew_scale=50,  unwind=60,  hard=120 (30% / 60% of position limit)

The strategy reacts earlier to adverse inventory accumulation, preventing
position from swinging as far during volatile regimes. Backtest comparison
on Round 3 historical data:

  v1: total=23,978, mean=7,993, stdev=5,852, sharpe-like=1.37, worst DD=-7,561
  v2: total=23,309, mean=7,770, stdev=4,732, sharpe-like=1.64, worst DD=-3,969

Trades ~3% of total PnL for a 47% reduction in worst drawdown.

Variants explored and abandoned:
- Vol-conditional widening: tick-level mid volatility is identical across
  day 0/1/2, so vol-based regime detection has no signal in this product.
- MTM stop-loss: halts new quotes during drawdowns, but drawdowns in
  mean-reverting products recover — halting locks in losses.
"""
from datamodel import Order, OrderDepth, TradingState


class Trader:
    HYDROGEL = "HYDROGEL_PACK"
    POSITION_LIMITS = {HYDROGEL: 200}

    HYDROGEL_CONFIG = {
        "improve_ticks": 1,
        "min_quote_spread": 6,
        "quote_size": 10,
        "min_quote_size": 2,
        "skew_inventory_scale": 50,         # halved from 100 -> stronger size skew earlier
        "unwind_inventory_limit": 60,       # halved from 120 -> aggressive mid-quote at |pos|>60
        "hard_inventory_limit": 120,        # halved from 180 -> cross book at |pos|>120
        "skew_imbalance_threshold": 0.10,
        "take_distance": 4,
        "flatten_window_start_ts": 980_000,
        "flatten_quote_offset": 2,
    }

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {self.HYDROGEL: []}
        depth = state.order_depths.get(self.HYDROGEL)
        if depth is None:
            return result, 0, ""
        position = int(state.position.get(self.HYDROGEL, 0))
        result[self.HYDROGEL] = self._trade_hydrogel(depth, position, int(state.timestamp))
        return result, 0, ""

    def _trade_hydrogel(self, depth, position, timestamp):
        cfg = self.HYDROGEL_CONFIG
        limit = self.POSITION_LIMITS[self.HYDROGEL]
        best_bid, best_ask = self._best_prices(depth)
        if best_bid is None or best_ask is None:
            return []
        spread = best_ask - best_bid
        touch_mid = (best_bid + best_ask) / 2.0
        bid_room = max(0, limit - position)
        ask_room = max(0, limit + position)

        if position > cfg["hard_inventory_limit"]:
            return self._take_to_neutralize(depth, position, touch_mid, side="sell")
        if position < -cfg["hard_inventory_limit"]:
            return self._take_to_neutralize(depth, position, touch_mid, side="buy")
        if spread < cfg["min_quote_spread"]:
            return []

        bid_price, ask_price = self._choose_quote_prices(
            depth, position, best_bid, best_ask, touch_mid, timestamp,
        )
        if bid_price >= best_ask:
            bid_price = best_ask - 1
        if ask_price <= best_bid:
            ask_price = best_bid + 1

        bid_qty, ask_qty = self._sized_quotes(position, bid_room, ask_room)

        orders = []
        if bid_qty > 0:
            orders.append(Order(self.HYDROGEL, int(bid_price), int(bid_qty)))
        if ask_qty > 0:
            orders.append(Order(self.HYDROGEL, int(ask_price), -int(ask_qty)))
        return orders

    def _choose_quote_prices(self, depth, position, best_bid, best_ask, touch_mid, timestamp):
        cfg = self.HYDROGEL_CONFIG
        if timestamp > cfg["flatten_window_start_ts"]:
            return (
                int(round(touch_mid - cfg["flatten_quote_offset"])),
                int(round(touch_mid + cfg["flatten_quote_offset"])),
            )
        if abs(position) > cfg["unwind_inventory_limit"]:
            if position > 0:
                return best_bid + cfg["improve_ticks"], int(round(touch_mid - 1))
            else:
                return int(round(touch_mid + 1)), best_ask - cfg["improve_ticks"]
        di = self._depth_imbalance(depth)
        bid_improve = cfg["improve_ticks"]
        ask_improve = cfg["improve_ticks"]
        if di >= cfg["skew_imbalance_threshold"]:
            bid_improve = 0
        elif di <= -cfg["skew_imbalance_threshold"]:
            ask_improve = 0
        return best_bid + bid_improve, best_ask - ask_improve

    def _sized_quotes(self, position, bid_room, ask_room):
        cfg = self.HYDROGEL_CONFIG
        scale = cfg["skew_inventory_scale"]
        pressure = max(-2.0, min(2.0, position / scale))
        bid_factor = max(0.0, 1.0 - max(0.0, pressure))
        ask_factor = max(0.0, 1.0 + min(0.0, pressure))
        base = cfg["quote_size"]
        bid_qty = max(cfg["min_quote_size"], int(round(base * bid_factor)))
        ask_qty = max(cfg["min_quote_size"], int(round(base * ask_factor)))
        return min(bid_qty, bid_room), min(ask_qty, ask_room)

    def _take_to_neutralize(self, depth, position, touch_mid, side):
        cfg = self.HYDROGEL_CONFIG
        target_qty = abs(position) - cfg["unwind_inventory_limit"]
        orders = []
        if side == "sell":
            for price in sorted(depth.buy_orders.keys(), reverse=True):
                if price < touch_mid - cfg["take_distance"]:
                    break
                avail = int(depth.buy_orders[price])
                qty = min(target_qty, max(0, avail))
                if qty <= 0:
                    continue
                orders.append(Order(self.HYDROGEL, int(price), -int(qty)))
                target_qty -= qty
                if target_qty <= 0:
                    break
        else:
            for price in sorted(depth.sell_orders.keys()):
                if price > touch_mid + cfg["take_distance"]:
                    break
                avail = abs(int(depth.sell_orders[price]))
                qty = min(target_qty, max(0, avail))
                if qty <= 0:
                    continue
                orders.append(Order(self.HYDROGEL, int(price), int(qty)))
                target_qty -= qty
                if target_qty <= 0:
                    break
        return orders

    @staticmethod
    def _best_prices(depth):
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        return best_bid, best_ask

    @staticmethod
    def _depth_imbalance(depth):
        bids = sum(depth.buy_orders.values()) if depth.buy_orders else 0
        asks = sum(abs(v) for v in depth.sell_orders.values()) if depth.sell_orders else 0
        total = bids + asks
        if total == 0:
            return 0.0
        return (bids - asks) / total

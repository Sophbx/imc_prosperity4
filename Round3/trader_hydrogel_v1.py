"""
HYDROGEL_PACK passive market maker (v1).

Strategy: improve by 1 tick on each side of the touch with inventory-skewed
sizing, depth-imbalance fade, aggressive mid-quotes when |pos| > 120, and
book-crossing when |pos| > 180.

Designed against Round 3 historical book; no persisted state across ticks
to avoid overfitting to micro-features. See Round3/Analysis/output/ for
the research artifacts that motivated the parameters.
"""
from datamodel import Order, OrderDepth, TradingState


class Trader:
    HYDROGEL = "HYDROGEL_PACK"
    POSITION_LIMITS = {HYDROGEL: 200}

    HYDROGEL_CONFIG = {
        "improve_ticks": 1,                # post inside the touch by this many ticks
        "min_quote_spread": 6,             # don't quote when market spread is below this
        "quote_size": 10,                  # base quote size per side
        "min_quote_size": 2,               # floor when inventory skew shrinks a side
        "skew_inventory_scale": 100,       # |pos| / this gives the skew pressure (capped at ±2)
        "unwind_inventory_limit": 120,     # |pos| above this -> aggressive mid quotes
        "hard_inventory_limit": 180,       # |pos| above this -> cross the book
        "skew_imbalance_threshold": 0.10,  # |depth_imbalance| above this -> fade the heavier side
        "take_distance": 4,                # when crossing, hit prices within touch_mid ± this
        "flatten_window_start_ts": 980_000,  # last 2% of timestamps -> tighten quotes
        "flatten_quote_offset": 2,         # mid ± this in flatten window
    }

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {self.HYDROGEL: []}

        depth = state.order_depths.get(self.HYDROGEL)
        if depth is None:
            return result, 0, ""

        position = int(state.position.get(self.HYDROGEL, 0))
        result[self.HYDROGEL] = self._trade_hydrogel(depth, position, int(state.timestamp))
        return result, 0, ""

    def _trade_hydrogel(
        self,
        depth: OrderDepth,
        position: int,
        timestamp: int,
    ) -> list[Order]:
        cfg = self.HYDROGEL_CONFIG
        limit = self.POSITION_LIMITS[self.HYDROGEL]

        best_bid, best_ask = self._best_prices(depth)
        if best_bid is None or best_ask is None:
            return []

        spread = best_ask - best_bid
        touch_mid = (best_bid + best_ask) / 2.0
        bid_room = max(0, limit - position)
        ask_room = max(0, limit + position)

        # ----- HARD UNWIND: cross the book when above hard limit -----
        if position > cfg["hard_inventory_limit"]:
            return self._take_to_neutralize(depth, position, touch_mid, side="sell")
        if position < -cfg["hard_inventory_limit"]:
            return self._take_to_neutralize(depth, position, touch_mid, side="buy")

        # ----- Skip if market spread is too tight -----
        if spread < cfg["min_quote_spread"]:
            return []

        # ----- Decide quote prices -----
        bid_price, ask_price = self._choose_quote_prices(
            depth, position, best_bid, best_ask, touch_mid, timestamp,
        )

        # Sanity: never post a quote that would cross the book passively.
        if bid_price >= best_ask:
            bid_price = best_ask - 1
        if ask_price <= best_bid:
            ask_price = best_bid + 1

        # ----- Sizing with inventory skew -----
        bid_qty, ask_qty = self._sized_quotes(position, bid_room, ask_room)

        orders: list[Order] = []
        if bid_qty > 0:
            orders.append(Order(self.HYDROGEL, int(bid_price), int(bid_qty)))
        if ask_qty > 0:
            orders.append(Order(self.HYDROGEL, int(ask_price), -int(ask_qty)))
        return orders

    def _choose_quote_prices(
        self,
        depth: OrderDepth,
        position: int,
        best_bid: int,
        best_ask: int,
        touch_mid: float,
        timestamp: int,
    ) -> tuple[int, int]:
        cfg = self.HYDROGEL_CONFIG

        # End-of-day flatten window: tighten to mid ± offset to maximize unwind.
        if timestamp > cfg["flatten_window_start_ts"]:
            return (
                int(round(touch_mid - cfg["flatten_quote_offset"])),
                int(round(touch_mid + cfg["flatten_quote_offset"])),
            )

        # Moderate inventory: post aggressive mid quotes on the side we want to reduce.
        if abs(position) > cfg["unwind_inventory_limit"]:
            if position > 0:
                # Long → quote sell aggressively, normal bid.
                return best_bid + cfg["improve_ticks"], int(round(touch_mid - 1))
            else:
                # Short → quote buy aggressively, normal ask.
                return int(round(touch_mid + 1)), best_ask - cfg["improve_ticks"]

        # Normal MM: improve by 1 on both sides, with depth-imbalance fade.
        di = self._depth_imbalance(depth)
        bid_improve = cfg["improve_ticks"]
        ask_improve = cfg["improve_ticks"]
        if di >= cfg["skew_imbalance_threshold"]:
            # More total bids than asks → research says price tends down → widen our bid.
            bid_improve = 0
        elif di <= -cfg["skew_imbalance_threshold"]:
            ask_improve = 0
        return best_bid + bid_improve, best_ask - ask_improve

    def _sized_quotes(
        self,
        position: int,
        bid_room: int,
        ask_room: int,
    ) -> tuple[int, int]:
        cfg = self.HYDROGEL_CONFIG
        scale = cfg["skew_inventory_scale"]
        pressure = max(-2.0, min(2.0, position / scale))
        # When long (pressure > 0): shrink bid; when short (< 0): shrink ask.
        bid_factor = max(0.0, 1.0 - max(0.0, pressure))
        ask_factor = max(0.0, 1.0 + min(0.0, pressure))
        base = cfg["quote_size"]
        bid_qty = max(cfg["min_quote_size"], int(round(base * bid_factor)))
        ask_qty = max(cfg["min_quote_size"], int(round(base * ask_factor)))
        return min(bid_qty, bid_room), min(ask_qty, ask_room)

    def _take_to_neutralize(
        self,
        depth: OrderDepth,
        position: int,
        touch_mid: float,
        side: str,
    ) -> list[Order]:
        cfg = self.HYDROGEL_CONFIG
        target_qty = abs(position) - cfg["unwind_inventory_limit"]
        orders: list[Order] = []

        if side == "sell":
            # Long: hit bids above touch_mid - take_distance, walking down from best.
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
        else:  # side == "buy"
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
    def _best_prices(depth: OrderDepth) -> tuple[int | None, int | None]:
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        return best_bid, best_ask

    @staticmethod
    def _depth_imbalance(depth: OrderDepth) -> float:
        """(total bid volume - total ask volume) / total, in [-1, 1]."""
        bids = sum(depth.buy_orders.values()) if depth.buy_orders else 0
        asks = sum(abs(v) for v in depth.sell_orders.values()) if depth.sell_orders else 0
        total = bids + asks
        if total == 0:
            return 0.0
        return (bids - asks) / total

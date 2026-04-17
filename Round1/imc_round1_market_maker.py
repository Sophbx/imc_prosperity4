import json
from typing import Any

from datamodel import Order, OrderDepth, TradingState


class Trader:
    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"
    PRODUCTS = (OSMIUM, PEPPER)

    POSITION_LIMITS = {
        OSMIUM: 12,
        PEPPER: 8,
    }

    OSMIUM_CONFIG = {
        "ema_alpha": 0.45,
        "inventory_skew": 3.0,
        "min_edge": 4,
        "order_size": 18,
    }

    PEPPER_CONFIG = {
        "ema_alpha": 0.12,
        "trend_alpha": 0.22,
        "inventory_skew": 0.25,
        "signal_threshold": 2.5,
        "prediction_horizon": 10.0,
        "passive_size": 3,
        "max_spread": 14,
    }

    def bid(self):
        return 15

    def run(self, state: TradingState):
        memory = self._load_memory(state.traderData)
        result: dict[str, list[Order]] = {}

        for product in self.PRODUCTS:
            depth = state.order_depths.get(product)
            if depth is None:
                result[product] = []
                continue

            position = state.position.get(product, 0)
            best_bid, best_ask = self._best_prices(depth)
            if best_bid is None or best_ask is None:
                result[product] = []
                continue

            mid = (best_bid + best_ask) / 2.0
            product_memory = memory.setdefault(product, {})
            self._update_memory(product_memory, mid)

            if product == self.OSMIUM:
                orders = self._trade_osmium(depth, position, product_memory)
            else:
                orders = self._trade_pepper(depth, position, product_memory)

            result[product] = orders

        trader_data = json.dumps(memory, separators=(",", ":"))
        conversions = 0
        return result, conversions, trader_data

    def _trade_osmium(self, depth: OrderDepth, position: int, memory: dict[str, Any]) -> list[Order]:
        best_bid, best_ask = self._best_prices(depth)
        if best_bid is None or best_ask is None:
            return []

        cfg = self.OSMIUM_CONFIG
        spread = best_ask - best_bid
        if spread <= 1 or spread > cfg["max_spread"]:
            return []
        
        fair_value = self._microprice(depth)
        smoothed_mid = memory["ema_mid"]
        reservation_price = 0.55 * fair_value + 0.45 * smoothed_mid - cfg["inventory_skew"] * position

        buy_limit = self.POSITION_LIMITS[self.OSMIUM] - position
        sell_limit = self.POSITION_LIMITS[self.OSMIUM] + position
        inside_bid = best_bid + 1
        inside_ask = best_ask - 1

        orders: list[Order] = []

        if buy_limit > 0 and inside_bid < best_ask and reservation_price - inside_bid >= cfg["min_edge"]:
            qty = min(cfg["order_size"], buy_limit)
            orders.append(Order(self.OSMIUM, inside_bid, qty))

        if sell_limit > 0 and inside_ask > best_bid and inside_ask - reservation_price >= cfg["min_edge"]:
            qty = min(cfg["order_size"], sell_limit)
            orders.append(Order(self.OSMIUM, inside_ask, -qty))

        return orders

    def _trade_pepper(self, depth: OrderDepth, position: int, memory: dict[str, Any]) -> list[Order]:
        best_bid, best_ask = self._best_prices(depth)
        if best_bid is None or best_ask is None:
            return []

        cfg = self.PEPPER_CONFIG
        spread = best_ask - best_bid
        if spread <= 1 or spread > cfg["max_spread"]:
            return []
        
        slope = memory["ema_slope"]
        fair_now = memory["ema_mid"]
        predicted_fair = fair_now + cfg["prediction_horizon"] * slope - cfg["inventory_skew"] * position

        buy_limit = self.POSITION_LIMITS[self.PEPPER] - position
        sell_limit = self.POSITION_LIMITS[self.PEPPER] + position
        signal = predicted_fair - ((best_bid + best_ask) / 2.0)
        orders: list[Order] = []

        # Pepper Root drifts upward in the capsule, but the drift is too small
        # to justify lifting the ask. Only quote passively and bias toward bids.
        passive_bid = min(best_ask - 1, best_bid + 1)
        passive_ask = max(best_bid + 1, best_ask - 1)

        if signal >= cfg["signal_threshold"] and buy_limit > 0:
            if passive_bid < best_ask and passive_bid <= predicted_fair - 1:
                orders.append(Order(self.PEPPER, passive_bid, min(cfg["passive_size"], buy_limit)))
        
        # Only offer inventory back out when we are already long enough or the
        # short-term signal turns clearly against us.
        if position > 0 and sell_limit > 0:
            should_offer = signal <= 0.5 or position >= self.POSITION_LIMITS[self.PEPPER] // 2
            if should_offer and passive_ask > best_bid and passive_ask >= predicted_fair:
                orders.append(Order(self.PEPPER, passive_ask, -min(cfg["passive_size"], sell_limit, position)))

        return orders

    def _update_memory(self, product_memory: dict[str, Any], mid: float) -> None:
        product = product_memory.get("product")
        if product is None:
            return

        if "ema_mid" not in product_memory:
            product_memory["last_mid"] = mid
            product_memory["ema_mid"] = mid
            product_memory["ema_slope"] = 0.0
            return

        last_mid = product_memory["last_mid"]
        if "ASH_COATED_OSMIUM" in str(product):
            alpha = self.OSMIUM_CONFIG["ema_alpha"]
            slope_alpha = 0.0
        else:
            alpha = self.PEPPER_CONFIG["ema_alpha"]
            slope_alpha = self.PEPPER_CONFIG["trend_alpha"]

        ema_mid = product_memory["ema_mid"] + alpha * (mid - product_memory["ema_mid"])
        raw_slope = mid - last_mid
        ema_slope = product_memory["ema_slope"] + slope_alpha * (raw_slope - product_memory["ema_slope"])

        product_memory["last_mid"] = mid
        product_memory["ema_mid"] = ema_mid
        product_memory["ema_slope"] = ema_slope

    def _load_memory(self, trader_data: str) -> dict[str, dict[str, Any]]:
        if not trader_data:
            return {
                self.OSMIUM: {"product": self.OSMIUM},
                self.PEPPER: {"product": self.PEPPER},
            }

        try:
            memory = json.loads(trader_data)
        except json.JSONDecodeError:
            memory = {}

        for product in self.PRODUCTS:
            product_memory = memory.setdefault(product, {})
            product_memory.setdefault("product", product)
        return memory

    @staticmethod
    def _best_prices(depth: OrderDepth) -> tuple[int | None, int | None]:
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        return best_bid, best_ask

    @staticmethod
    def _microprice(depth: OrderDepth) -> float:
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        bid_volume = depth.buy_orders[best_bid]
        ask_volume = abs(depth.sell_orders[best_ask])
        total = bid_volume + ask_volume
        if total == 0:
            return (best_bid + best_ask) / 2.0
        return (best_ask * bid_volume + best_bid * ask_volume) / total
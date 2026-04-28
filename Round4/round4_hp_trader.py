from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    POSITION_LIMITS = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
    }

    HP_ALPHA = {
        "Mark 14": 1.5,
        "Mark 38": -1.5,
    }

    VE_ALPHA = {
        "Mark 67": 2.0,
        "Mark 55": 0.4,
        "Mark 01": 0.1,
        "Mark 14": -0.7,
        "Mark 22": -1.5,
        "Mark 49": -1.8,
    }

    VOUCHER_STRIKES = {
        "VEV_4000": 4000,
        "VEV_4500": 4500,
        "VEV_5000": 5000,
        "VEV_5100": 5100,
        "VEV_5200": 5200,
        "VEV_5300": 5300,
        "VEV_5400": 5400,
        "VEV_5500": 5500,
        "VEV_6000": 6000,
        "VEV_6500": 6500,
    }

    for p in VOUCHER_STRIKES:
        POSITION_LIMITS[p] = 300

    BASE_IV = 0.23
    T_DAYS = 4.0

    def norm_cdf(self, x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_call(self, s, k, sigma, t_days):
        if s <= 0 or k <= 0:
            return 0.0
        total_vol = sigma * math.sqrt(t_days / 365.0)
        if total_vol <= 1e-9:
            return max(s - k, 0.0)
        d1 = (math.log(s / k) + 0.5 * total_vol * total_vol) / total_vol
        d2 = d1 - total_vol
        return s * self.norm_cdf(d1) - k * self.norm_cdf(d2)

    def trade_voucher(self, state, product, strike, s_fair):
        depth = state.order_depths.get(product)
        if depth is None:
            return []

        bid, ask = self.best_bid_ask(depth)
        if bid is None or ask is None:
            return []

        pos = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        theo = self.bs_call(s_fair, strike, self.BASE_IV, self.T_DAYS)
        fair = theo - 0.08 * pos

        orders = []

        if ask <= fair - 1:
            qty = self.clamp_buy(pos, limit, -depth.sell_orders[ask])
            if qty > 0:
                orders.append(Order(product, ask, qty))
                pos += qty

        if bid >= fair + 1:
            qty = self.clamp_sell(pos, limit, depth.buy_orders[bid])
            if qty > 0:
                orders.append(Order(product, bid, -qty))

        return orders

    def load_memory(self, trader_data: str) -> Dict:
        if trader_data:
            try:
                data = json.loads(trader_data)
                if isinstance(data, dict):
                    data.setdefault("ema", {})
                    data.setdefault("hist", {})
                    return data
            except Exception:
                pass
        return {"ema": {}, "hist": {}}

    def save_memory(self, mem: Dict) -> str:
        try:
            return json.dumps(mem, separators=(",", ":"))
        except Exception:
            return ""

    def best_bid_ask(self, depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        return best_bid, best_ask

    def mid_price(self, depth: OrderDepth) -> Optional[float]:
        bid, ask = self.best_bid_ask(depth)
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def update_ema(self, mem: Dict, key: str, value: float, alpha: float) -> float:
        prev = mem["ema"].get(key)
        ema = value if prev is None else alpha * value + (1 - alpha) * prev
        mem["ema"][key] = ema
        return ema

    def mark_signal(self, trades, alpha_map: Dict[str, float]) -> float:
        signal = 0.0
        for t in trades:
            if t.buyer in alpha_map:
                signal += alpha_map[t.buyer]
            if t.seller in alpha_map:
                signal -= alpha_map[t.seller]
        return max(-3.0, min(3.0, signal))

    def clamp_buy(self, position: int, limit: int, qty: int) -> int:
        return max(0, min(qty, limit - position))

    def clamp_sell(self, position: int, limit: int, qty: int) -> int:
        return max(0, min(qty, limit + position))

    def trade_product(
        self,
        state: TradingState,
        mem: Dict,
        product: str,
        alpha_map: Dict[str, float],
        ema_alpha: float,
        take_edge: float,
        quote_edge: int,
        quote_size: int,
        inv_skew: float,
    ) -> List[Order]:

        depth = state.order_depths.get(product)
        if depth is None:
            return []

        best_bid, best_ask = self.best_bid_ask(depth)
        mid = self.mid_price(depth)

        if best_bid is None or best_ask is None or mid is None:
            return []

        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        orders: List[Order] = []

        ema = self.update_ema(mem, product, mid, ema_alpha)
        mark_adj = self.mark_signal(state.market_trades.get(product, []), alpha_map)

        fair = 0.65 * mid + 0.35 * ema + mark_adj
        fair -= inv_skew * position

        pos = position

        # take cheap asks
        for ask in sorted(depth.sell_orders.keys()):
            ask_qty = -depth.sell_orders[ask]
            if ask <= fair - take_edge:
                qty = self.clamp_buy(pos, limit, ask_qty)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    pos += qty

        # hit expensive bids
        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            bid_qty = depth.buy_orders[bid]
            if bid >= fair + take_edge:
                qty = self.clamp_sell(pos, limit, bid_qty)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    pos -= qty

        # passive quotes
        buy_px = min(best_bid + 1, int(math.floor(fair - quote_edge)))
        sell_px = max(best_ask - 1, int(math.ceil(fair + quote_edge)))

        if buy_px < best_ask:
            qty = self.clamp_buy(pos, limit, quote_size)
            if qty > 0:
                orders.append(Order(product, buy_px, qty))

        if sell_px > best_bid:
            qty = self.clamp_sell(pos, limit, quote_size)
            if qty > 0:
                orders.append(Order(product, sell_px, -qty))

        return orders

    def run(self, state: TradingState):
        mem = self.load_memory(state.traderData)
        result: Dict[str, List[Order]] = {}

        hp_orders = self.trade_product(
            state,
            mem,
            "HYDROGEL_PACK",
            self.HP_ALPHA,
            ema_alpha=0.08,
            take_edge=2.0,
            quote_edge=6,
            quote_size=25,
            inv_skew=0.12,
        )
        if hp_orders:
            result["HYDROGEL_PACK"] = hp_orders

        ve_orders = self.trade_product(
            state,
            mem,
            "VELVETFRUIT_EXTRACT",
            self.VE_ALPHA,
            ema_alpha=0.10,
            take_edge=1.0,
            quote_edge=1,
            quote_size=25,
            inv_skew=0.08,
        )
        if ve_orders:
            result["VELVETFRUIT_EXTRACT"] = ve_orders

        conversions = 0
        trader_data = self.save_memory(mem)

        ve_mid = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            ve_mid = self.mid_price(state.order_depths["VELVETFRUIT_EXTRACT"])

        if ve_mid is not None:
            for product, strike in self.VOUCHER_STRIKES.items():
                orders = self.trade_voucher(state, product, strike, ve_mid)
                if orders:
                    result[product] = orders
        
        return result, conversions, trader_data


        
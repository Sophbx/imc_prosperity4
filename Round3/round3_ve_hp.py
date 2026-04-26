from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    POSITION_LIMITS: Dict[str, int] = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300,
        "VEV_4500": 300,
        "VEV_5000": 300,
        "VEV_5100": 300,
        "VEV_5200": 300,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
        "VEV_6000": 300,
        "VEV_6500": 300,
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

    # Round 3 final simulation starts with TTE = 5 days
    T_DAYS = 5.0

    # Training-data-based volatility smile, centered around ATM
    BASE_SIGMA = 0.01315
    SMILE_RATIO = {
        4000: 1.00,
        4500: 1.00,
        5000: 1.001,
        5100: 0.993,
        5200: 1.004,
        5300: 1.015,
        5400: 0.952,
        5500: 1.033,
        6000: 1.08,
        6500: 1.12,
    }

    HYDRO_ALPHA = 0.03
    EXTRACT_ALPHA = 0.05

    def load_memory(self, raw: str) -> Dict:
        if not raw:
            return {"ema": {}}
        try:
            mem = json.loads(raw)
            if not isinstance(mem, dict):
                return {"ema": {}}
            mem.setdefault("ema", {})
            return mem
        except Exception:
            return {"ema": {}}

    def save_memory(self, mem: Dict) -> str:
        try:
            return json.dumps(mem, separators=(",", ":"))
        except Exception:
            return '{"ema":{}}'

    @staticmethod
    def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        return best_bid, best_ask

    def get_mid(self, depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask = self.best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2.0

    def get_wall_mid(self, depth: OrderDepth) -> Optional[float]:
        if not depth.buy_orders or not depth.sell_orders:
            return self.get_mid(depth)

        bid_wall_price = max(depth.buy_orders.items(), key=lambda x: abs(x[1]))[0]
        ask_wall_price = min(depth.sell_orders.items(), key=lambda x: -abs(x[1]))[0]

        return (bid_wall_price + ask_wall_price) / 2.0

    def update_ema(self, mem: Dict, key: str, x: float, alpha: float) -> float:
        ema_map = mem["ema"]
        prev = ema_map.get(key)
        if prev is None:
            ema = x
        else:
            ema = alpha * x + (1.0 - alpha) * prev
        ema_map[key] = ema
        return ema

    @staticmethod
    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_call(self, s: float, k: float, sigma: float, t_days: float) -> float:
        if s <= 0 or k <= 0:
            return 0.0

        if t_days <= 1e-9 or sigma <= 1e-9:
            return max(s - k, 0.0)

        total_vol = sigma * math.sqrt(t_days)
        if total_vol <= 1e-12:
            return max(s - k, 0.0)

        d1 = (math.log(s / k) + 0.5 * total_vol * total_vol) / total_vol
        d2 = d1 - total_vol
        return s * self.norm_cdf(d1) - k * self.norm_cdf(d2)

    def fair_from_wall_and_ema(
        self,
        mem: Dict,
        product: str,
        depth: OrderDepth,
        alpha: float,
        wall_weight: float,
    ) -> Optional[float]:
        mid = self.get_mid(depth)
        wall = self.get_wall_mid(depth)

        if mid is None and wall is None:
            return None
        if wall is None:
            wall = mid
        if mid is None:
            mid = wall

        ema = self.update_ema(mem, product, wall, alpha)
        fair = wall_weight * wall + (1.0 - wall_weight) * ema
        return fair

    def clamp_buy(self, position: int, limit: int, desired: int) -> int:
        return max(0, min(desired, limit - position))

    def clamp_sell(self, position: int, limit: int, desired: int) -> int:
        return max(0, min(desired, limit + position))

    def take_crossed_quotes(
        self,
        product: str,
        depth: OrderDepth,
        fair: float,
        take_edge: float,
        position: int,
        limit: int,
    ) -> List[Order]:
        orders: List[Order] = []
        pos = position

        for ask in sorted(depth.sell_orders.keys()):
            ask_qty = -depth.sell_orders[ask]
            if ask <= fair - take_edge:
                qty = self.clamp_buy(pos, limit, ask_qty)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    pos += qty

        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            bid_qty = depth.buy_orders[bid]
            if bid >= fair + take_edge:
                qty = self.clamp_sell(pos, limit, bid_qty)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    pos -= qty

        return orders

    def make_quotes(
        self,
        product: str,
        depth: OrderDepth,
        fair: float,
        position: int,
        limit: int,
        edge: int,
        size: int,
        inv_skew: float,
    ) -> List[Order]:
        orders: List[Order] = []
        best_bid, best_ask = self.best_bid_ask(depth)

        if best_bid is None or best_ask is None:
            return orders

        fair_adj = fair - inv_skew * position

        buy_px = int(math.floor(fair_adj - edge))
        sell_px = int(math.ceil(fair_adj + edge))

        buy_px = min(buy_px, best_bid + 1)
        sell_px = max(sell_px, best_ask - 1)

        if buy_px < best_ask:
            buy_qty = self.clamp_buy(position, limit, size)
            if buy_qty > 0:
                orders.append(Order(product, buy_px, buy_qty))

        if sell_px > best_bid:
            sell_qty = self.clamp_sell(position, limit, size)
            if sell_qty > 0:
                orders.append(Order(product, sell_px, -sell_qty))

        return orders

    def trade_hydrogel(self, state: TradingState, mem: Dict) -> List[Order]:
        product = "HYDROGEL_PACK"
        depth = state.order_depths.get(product)
        if depth is None:
            return []

        fair = self.fair_from_wall_and_ema(
            mem=mem,
            product=product,
            depth=depth,
            alpha=self.HYDRO_ALPHA,
            wall_weight=0.65,
        )
        if fair is None:
            return []

        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]

        orders: List[Order] = []
        orders += self.take_crossed_quotes(
            product=product,
            depth=depth,
            fair=fair,
            take_edge=5.0,
            position=position,
            limit=limit,
        )

        pos_after = position + sum(o.quantity for o in orders)

        orders += self.make_quotes(
            product=product,
            depth=depth,
            fair=fair,
            position=pos_after,
            limit=limit,
            edge=4,
            size=20,
            inv_skew=0.10,
        )

        return orders

    def trade_extract(self, state: TradingState, mem: Dict) -> Tuple[List[Order], Optional[float]]:
        product = "VELVETFRUIT_EXTRACT"
        depth = state.order_depths.get(product)
        if depth is None:
            return [], None

        fair = self.fair_from_wall_and_ema(
            mem=mem,
            product=product,
            depth=depth,
            alpha=self.EXTRACT_ALPHA,
            wall_weight=0.70,
        )
        if fair is None:
            return [], None

        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]

        orders: List[Order] = []
        orders += self.take_crossed_quotes(
            product=product,
            depth=depth,
            fair=fair,
            take_edge=1.0,
            position=position,
            limit=limit,
        )

        pos_after = position + sum(o.quantity for o in orders)

        orders += self.make_quotes(
            product=product,
            depth=depth,
            fair=fair,
            position=pos_after,
            limit=limit,
            edge=1,
            size=25,
            inv_skew=0.08,
        )

        return orders, fair

    def trade_voucher(
        self,
        state: TradingState,
        product: str,
        strike: int,
        s_fair: float,
    ) -> List[Order]:
        depth = state.order_depths.get(product)
        if depth is None:
            return []

        best_bid, best_ask = self.best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return []

        sigma = self.BASE_SIGMA * self.SMILE_RATIO[strike]
        theo = self.bs_call(s_fair, float(strike), sigma, self.T_DAYS)

        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        spread = best_ask - best_bid

        # inventory-adjusted fair
        fair = theo - 0.10 * position

        # tighter / safer handling for dead far OTM options
        if strike >= 6000:
            take_edge = 1.5
            quote_edge = 1
            quote_size = 8
        elif strike <= 4500:
            take_edge = 2.0
            quote_edge = max(1, spread // 2)
            quote_size = 10
        else:
            take_edge = 1.0
            quote_edge = max(1, spread // 2)
            quote_size = 15

        orders: List[Order] = []
        pos = position

        # Aggressive take
        for ask in sorted(depth.sell_orders.keys()):
            ask_qty = -depth.sell_orders[ask]
            if ask <= fair - take_edge:
                qty = self.clamp_buy(pos, limit, ask_qty)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    pos += qty

        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            bid_qty = depth.buy_orders[bid]
            if bid >= fair + take_edge:
                qty = self.clamp_sell(pos, limit, bid_qty)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    pos -= qty

        # Passive make
        fair_adj = fair - 0.12 * pos
        buy_px = int(math.floor(fair_adj - quote_edge))
        sell_px = int(math.ceil(fair_adj + quote_edge))

        buy_px = min(buy_px, best_bid + 1)
        sell_px = max(sell_px, best_ask - 1)

        if buy_px < best_ask:
            buy_qty = self.clamp_buy(pos, limit, quote_size)
            if buy_qty > 0:
                orders.append(Order(product, buy_px, buy_qty))

        if sell_px > best_bid:
            sell_qty = self.clamp_sell(pos, limit, quote_size)
            if sell_qty > 0:
                orders.append(Order(product, sell_px, -sell_qty))

        return orders

    def run(self, state: TradingState):
        mem = self.load_memory(state.traderData)

        result: Dict[str, List[Order]] = {}

        hydro_orders = self.trade_hydrogel(state, mem)
        if hydro_orders:
            result["HYDROGEL_PACK"] = hydro_orders

        extract_orders, s_fair = self.trade_extract(state, mem)
        if extract_orders:
            result["VELVETFRUIT_EXTRACT"] = extract_orders

        if s_fair is not None:
            for product, strike in self.VOUCHER_STRIKES.items():
                voucher_orders = self.trade_voucher(state, product, strike, s_fair)
                if voucher_orders:
                    result[product] = voucher_orders

        conversions = 0
        trader_data = self.save_memory(mem)
        return result, conversions, trader_data
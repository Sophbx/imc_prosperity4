from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Optional, Tuple
import json
import math


SYM = "HYDROGEL_PACK"


class Trader:
    # Hydrogel asymmetric swing model with z-score entry ramp.
    #
    # Core winning structure preserved:
    #   flat  -> short only when Hydrogel is rich
    #   short -> long only after a large favorable reversal
    #   long  -> flat after rebound
    #
    # Improvement:
    #   Instead of entering full -200 immediately at z >= 0.85,
    #   ramp into the short:
    #       z >= 0.85 -> -120
    #       z >= 1.25 -> -160
    #       z >= 1.65 -> -200
    #
    # Important:
    #   Once short, we only increase size if the price gets richer.
    #   We do NOT reduce short size just because z falls.
    #   That avoids the bad dynamic/choppy behavior.

    FAIR = 10000.0
    SIGMA = 32.0
    LIMIT = 200

    ENTRY_Z_1 = 0.85
    ENTRY_Z_2 = 1.25
    ENTRY_Z_3 = 1.65

    SIZE_1 = 120
    SIZE_2 = 160
    SIZE_3 = 200

    REVERSAL_Z = 3.35
    REBOUND_Z = 1.20

    MAX_TRADE = 40

    COOLDOWN_TIME = 1200
    RESET_Z = 0.35

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

    def _load(self, td: str) -> Dict:
        if td:
            try:
                data = json.loads(td)
                if isinstance(data, dict):
                    data.setdefault("mode", "flat")
                    data.setdefault("extreme", None)
                    data.setdefault("cooldown_until", -1)
                    data.setdefault("needs_reset", False)
                    data.setdefault("short_size", 0)
                    data.setdefault("ema", {})
                    data.setdefault("hist", {})
                    return data
            except Exception:
                pass

        return {
            "mode": "flat",
            "extreme": None,
            "cooldown_until": -1,
            "needs_reset": False,
            "short_size": 0,
            "ema": {},
            "hist": {},
        }

    def _save(self, data: Dict) -> str:
        try:
            return json.dumps(data, separators=(",", ":"))[:50000]
        except Exception:
            return ""

    def _bb_ba(self, od: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        bb = max(od.buy_orders) if od.buy_orders else None
        ba = min(od.sell_orders) if od.sell_orders else None
        return bb, ba

    def _z(self, mid: float) -> float:
        return (mid - self.FAIR) / self.SIGMA

    def _desired_short_size_from_z(self, z: float) -> int:
        if z >= self.ENTRY_Z_3:
            return self.SIZE_3
        if z >= self.ENTRY_Z_2:
            return self.SIZE_2
        if z >= self.ENTRY_Z_1:
            return self.SIZE_1
        return 0

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

    def update_stat(self, mem, product, x):
        hist_map = mem.setdefault("hist", {})
        hist = hist_map.setdefault(product, [])

        hist.append(x)
        if len(hist) > 100:
            hist.pop(0)

        mean = sum(hist) / len(hist)
        var = sum((v - mean) ** 2 for v in hist) / max(1, len(hist) - 1)
        std = math.sqrt(max(var, 1e-6))

        return mean, std
    

    def mark_signal(self, trades, alpha_map):
        sig = 0.0
        for t in trades:
            if t.buyer in alpha_map:
                sig += alpha_map[t.buyer]
            if t.seller in alpha_map:
                sig -= alpha_map[t.seller]
        return max(-3.0, min(3.0, sig))

    def best_bid_ask(self, depth):
        bb = max(depth.buy_orders) if depth.buy_orders else None
        ba = min(depth.sell_orders) if depth.sell_orders else None
        return bb, ba

    def get_mid(self, depth):
        bb, ba = self.best_bid_ask(depth)
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2

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

        orders: List[Order] = []
        allow_buy = True
        allow_sell = True

        product = "HYDROGEL_PACK"
        depth = state.order_depths.get(product)
        if depth is None:
            return []

        wall = self.get_wall_mid(depth)
        if wall is None:
            return []

        short = self.update_ema(mem, product + "_short", wall, 0.12)
        long = self.update_ema(mem, product + "_long", wall, 0.02)
        mean, std = self.update_stat(mem, product, wall)
        z = (wall - mean) / std

        trend = short - long
        trend_strength = abs(trend)
        TREND_THRESHOLD = 2.0

        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        best_bid, best_ask = self.best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return []

        fair = 0.50 * wall + 0.30 * short + 0.20 * long

        mark_sig = self.mark_signal(state.market_trades.get(product, []),
                                    self.HP_ALPHA)
        fair += mark_sig

        if z > 1.2:
            fair -= 4
        elif z < -1.2:
            fair += 4

        if long < wall < short and trend > 0:
            fair -= 3
        elif short < wall < long and trend < 0:
            fair += 3

        fair -= 0.25 * position

        if position > 120:
            fair -= 0.8 * (position - 120)
        elif position < -120:
            fair -= 0.8 * (position + 120)

            if best_ask <= fair - 2 and position < 120:
                buy_qty = self.clamp_buy(position, limit, min(30, -depth.sell_orders[best_ask]))
                if buy_qty > 0:
                    orders.append(Order(product, best_ask, buy_qty))
                    position += buy_qty

        else:
            if allow_buy and best_ask <= fair - 1 and position < 120:
                buy_qty = self.clamp_buy(position, limit, min(25, -depth.sell_orders[best_ask]))
                if buy_qty > 0:
                    orders.append(Order(product, best_ask, buy_qty))
                    position += buy_qty

            if allow_sell and best_bid >= fair + 1 and position > -120:
                sell_qty = self.clamp_sell(position, limit, min(25, depth.buy_orders[best_bid]))
                if sell_qty > 0:
                    orders.append(Order(product, best_bid, -sell_qty))
                    position -= sell_qty

        edge = max(6, (best_ask - best_bid) // 2 - 1)
        buy_px = min(best_bid + 1, int(math.floor(fair - edge)))
        sell_px = max(best_ask - 1, int(math.ceil(fair + edge)))

            if buy_qty > 0:
                orders.append(Order(product, buy_px, buy_qty))

        if allow_sell and sell_px > best_bid and position > -120:
            sell_qty = self.clamp_sell(position, limit, 25)
            if sell_qty > 0:
                orders.append(Order(product, sell_px, -sell_qty))

        return orders

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

    def trade_extract(self, state: TradingState, mem: Dict) -> Tuple[List[Order], Optional[float]]:
        product = "VELVETFRUIT_EXTRACT"
        depth = state.order_depths.get(product)
        if depth is None:
            return [], None

        best_bid, best_ask = self.best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return [], None
        
        mark_sig = self.mark_signal(state.market_trades.get(product, []),
                                    self.VE_ALPHA)
        fair += mark_sig

        mid = self.get_mid(depth)
        wall = self.get_wall_mid(depth)

        if mid is None and wall is None:
            return [], None
        if wall is None:
            wall = mid
        if mid is None:
            mid = wall

        ema = self.update_ema(mem, product, wall, 0.05)
        ema_fast, ema_slow = self.update_ema_pair(
            mem,
            "VELVETFRUIT_EXTRACT_fast",
            "VELVETFRUIT_EXTRACT_slow",
            mid,
        )

        wall_signal = wall - mid
        trend_signal = ema_fast - ema_slow

    # softer fair
        fair = 0.60 * wall + 0.25 * ema + 0.15 * mid + 0.35 * wall_signal

        position = state.position.get(product, 0)

        SOFT_CAP = 45
        HARD_CAP = 75

        orders: List[Order] = []
        pos = position

    # 1) only reduce inventory if position is already fairly large
        if pos > 25 and trend_signal < -1.2:
            sell_qty = min(pos - 20, depth.buy_orders.get(best_bid, 0))
            if sell_qty > 0:
                orders.append(Order(product, best_bid, -sell_qty))
                pos -= sell_qty

        if pos < -25 and trend_signal > 1.2:
            buy_qty = min((-20 - pos), -depth.sell_orders.get(best_ask, 0))
            if buy_qty > 0:
                orders.append(Order(product, best_ask, buy_qty))
                pos += buy_qty

    # 2) aggressive taking with softer inventory adjustment
        buy_edge = 0.7 + 0.012 * max(pos, 0)
        sell_edge = 0.7 + 0.012 * max(-pos, 0)

        if trend_signal < -2.0:
            buy_edge += 0.4
        if trend_signal > 2.0:
            sell_edge += 0.4

        for ask in sorted(depth.sell_orders.keys()):
            ask_qty = -depth.sell_orders[ask]
            if ask <= fair - buy_edge and pos < HARD_CAP:
                qty = min(ask_qty, HARD_CAP - pos)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    pos += qty

        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            bid_qty = depth.buy_orders[bid]
            if bid >= fair + sell_edge and pos > -HARD_CAP:
                qty = min(bid_qty, pos + HARD_CAP)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    pos -= qty

    # 3) passive making with milder skew
        inv_skew = 0.05
        fair_adj = fair - inv_skew * pos

        quote_size = 18
        if abs(pos) >= 25:
            quote_size = 12
        if abs(pos) >= 45:
            quote_size = 7

        buy_px = int(math.floor(fair_adj - 1))
        sell_px = int(math.ceil(fair_adj + 1))

        buy_px = min(buy_px, best_bid + 1)
        sell_px = max(sell_px, best_ask - 1)

        allow_buy = True
        allow_sell = True

        if pos >= SOFT_CAP:
            allow_buy = False
        if pos <= -SOFT_CAP:
            allow_sell = False

        if trend_signal < -1.6 and pos > 20:
            allow_buy = False
        if trend_signal > 1.6 and pos < -20:
            allow_sell = False

        if allow_buy and buy_px < best_ask:
            buy_qty = min(quote_size, HARD_CAP - pos)
            if buy_qty > 0:
                orders.append(Order(product, buy_px, buy_qty))

        if allow_sell and sell_px > best_bid:
            sell_qty = min(quote_size, pos + HARD_CAP)
            if sell_qty > 0:
                orders.append(Order(product, sell_px, -sell_qty))

        return orders, fair
    
    def update_ema_pair(self, mem: Dict, key_fast: str, key_slow: str, x: float) -> Tuple[float, float]:
        ema_map = mem["ema"]

        prev_fast = ema_map.get(key_fast)
        prev_slow = ema_map.get(key_slow)

        fast = x if prev_fast is None else 0.18 * x + 0.82 * prev_fast
        slow = x if prev_slow is None else 0.04 * x + 0.96 * prev_slow

        ema_map[key_fast] = fast
        ema_map[key_slow] = slow
        return fast, slow

    def run(self, state: TradingState):
        
        mem = self._load(state.traderData)

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
        trader_data = self._save(mem)
        return result, conversions, trader_data
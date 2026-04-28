"""
Trader v4_full — same as v4 but with vouchers 5000-5500 re-enabled.

Components:
  - HYDROGEL_PACK: swing trader from 16kbase.py (fixed fair=10000, z-score
    state machine, asymmetric short-first entry, big-reversal flip to long).
    Backtest: $75,011 / 3 days standalone.
  - VELVETFRUIT_EXTRACT: passive MM (wall_mid + EMA fair, take + make).
    Backtest: $18,603 / 3 days.
  - VEV_4000: specialized ITM passive MM (min_quote_spread=12 to skip flickers).
    Backtest: $8,845 / 3 days.
  - VEV_4500: specialized ITM MM (min_quote_spread=14, effectively no-op safety).
    Backtest: $0 / 3 days.
  - VEV_5000-5500: BS+momentum-reversal with velvet_weight=0 + hard inventory
    cap at limit/5. Backtest: ~+$276 / 3 days (barely positive on aggregate).
  - VEV_6000/6500: still skipped (dead — mid pinned at 0.5).

This is the "trade everything you can" variant. v4 (without 5000-5500) is
slightly cleaner; v4_full keeps the small voucher block in case live regime
differs from backtest.

Expected combined PnL: ~$102.7K / 3 days.
"""
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

    # Only these options use the new combined residual-reversal logic.
    IMPROVED_OPTION_PRODUCTS = {
        "VEV_5000",
        "VEV_5100",
        "VEV_5200",
        "VEV_5300",
        "VEV_5400",
        "VEV_5500",
    }

    # Deep ITM vouchers: delta=1, wide spread, treat as separate delta-1 MM.
    # Per voucher_research, VEV_4000 contributes ~$8.6k/3d when MM'd correctly;
    # VEV_4500 has zero bot flow so we set a strict spread filter (no-op safety).
    ITM_VOUCHER_PRODUCTS = {"VEV_4000", "VEV_4500"}

    ITM_VOUCHER_CONFIG = {
        4000: {
            # Filter the ~1.8% of "flicker" ticks where spread briefly tightens
            # to 7-12 (1-2 tick events with adverse selection but no bot flow).
            # Normal regime is spread 20-22.
            "min_quote_spread": 12,
            "improve_ticks": 1,
            "quote_size": 10,
            "hard_inventory_limit": 200,
            "take_distance": 4,
        },
        4500: {
            # 16-tick spread, ~1 bot trade per 3 days. This config yields 0
            # fills in backtest (vs combined v1's -$33 from a flicker fill).
            # Strict spread filter is a no-op safety net.
            "min_quote_spread": 14,
            "improve_ticks": 1,
            "quote_size": 10,
            "hard_inventory_limit": 200,
            "take_distance": 4,
        },
    }

    # Round 3 final simulation starts at TTE = 5 days.
    # We now decay TTE dynamically through the day.
    START_T_DAYS = 5.0
    TIMESTAMPS_PER_DAY = 1_000_000

    # Same smile framework as r3_test.
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

    # Signal-history settings.
    HISTORY_LEN = 120
    VELVET_LOOKBACK = 40
    VELVET_VOL_WINDOW = 80
    RESID_LOOKBACK = 30
    RESID_VOL_WINDOW = 80

    # Product-specific blending.
    # Higher velvet_weight means more underlying-driven.
    # Higher residual_weight means more option-specific fair-value residual driven.
    IMPROVED_OPTION_CONFIG = {
        5000: {
            "velvet_weight": 0.0,
            "residual_weight": 0.20,
            "entry_z": 1.0,
            "strong_z": 1.8,
            "base_size": 24,
            "take_edge": 1.0,
            "quote_edge_min": 1,
            "inv_skew": 0.06,
            "resid_edge_mult": 0.10,
        },
        5100: {
            "velvet_weight": 0.0,
            "residual_weight": 0.40,
            "entry_z": 1.0,
            "strong_z": 1.8,
            "base_size": 24,
            "take_edge": 1.0,
            "quote_edge_min": 1,
            "inv_skew": 0.06,
            "resid_edge_mult": 0.10,
        },
        5200: {
            "velvet_weight": 0.0,
            "residual_weight": 0.60,
            "entry_z": 1.0,
            "strong_z": 1.8,
            "base_size": 22,
            "take_edge": 1.0,
            "quote_edge_min": 1,
            "inv_skew": 0.07,
            "resid_edge_mult": 0.12,
        },
        5300: {
            "velvet_weight": 0.0,
            "residual_weight": 0.75,
            "entry_z": 1.0,
            "strong_z": 1.8,
            "base_size": 18,
            "take_edge": 1.0,
            "quote_edge_min": 1,
            "inv_skew": 0.08,
            "resid_edge_mult": 0.15,
        },
        5400: {
            "velvet_weight": 0.0,
            "residual_weight": 0.90,
            "entry_z": 1.1,
            "strong_z": 2.0,
            "base_size": 14,
            "take_edge": 1.0,
            "quote_edge_min": 1,
            "inv_skew": 0.10,
            "resid_edge_mult": 0.18,
        },
        5500: {
            "velvet_weight": 0.00,
            "residual_weight": 1.00,
            "entry_z": 1.2,
            "strong_z": 2.2,
            "base_size": 10,
            "take_edge": 1.0,
            "quote_edge_min": 1,
            "inv_skew": 0.12,
            "resid_edge_mult": 0.20,
        },
    }

    OPTION_TREND_GUARD = {
        5000: {"lookback": 10, "threshold": 1.0},
        5100: {"lookback": 10, "threshold": 1.0},
        5200: {"lookback": 10, "threshold": 0.8},
        5300: {"lookback": 10, "threshold": 0.5},
        5400: {"lookback": 10, "threshold": 0.4},
        5500: {"lookback": 10, "threshold": 0.3},
    }
    
    def load_memory(self, raw: str) -> Dict:
        empty = {"ema": {}, "hist": {}, "hydro": {}}
        if not raw:
            return dict(empty)

        try:
            mem = json.loads(raw)
            if not isinstance(mem, dict):
                return dict(empty)

            mem.setdefault("ema", {})
            mem.setdefault("hist", {})
            mem.setdefault("hydro", {})
            return mem

        except Exception:
            return dict(empty)

    def save_memory(self, mem: Dict) -> str:
        compact = {
            "ema": mem.get("ema", {}),
            "hist": {},
            # HYDROGEL swing-trader state machine (mode/extreme/cooldown/...).
            # Critical to persist; without it the strategy resets every tick.
            "hydro": mem.get("hydro", {}),
        }

        hist = mem.get("hist", {})
        keep_keys = ["VELVETFRUIT_EXTRACT"]
        for product in self.IMPROVED_OPTION_PRODUCTS:
            keep_keys.append(f"{product}_residual")
            keep_keys.append(f"{product}_mid")

        for key in keep_keys:
            if key in hist:
                compact["hist"][key] = hist[key][-self.HISTORY_LEN:]

        try:
            return json.dumps(compact, separators=(",", ":"))
        except Exception:
            return '{"ema":{},"hist":{}}'

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

    def append_hist(self, mem: Dict, key: str, x: float) -> None:
        hist_map = mem.setdefault("hist", {})
        arr = hist_map.setdefault(key, [])
        arr.append(float(x))

        if len(arr) > self.HISTORY_LEN:
            del arr[:-self.HISTORY_LEN]

    def jump_score_from_hist(
        self,
        mem: Dict,
        key: str,
        lookback: int,
        vol_window: int,
    ) -> Optional[float]:
        hist = mem.get("hist", {}).get(key, [])

        if len(hist) < max(lookback + 1, 8):
            return None

        current = float(hist[-1])
        past = float(hist[-1 - lookback])
        short_ret = current - past

        start = max(1, len(hist) - vol_window)
        diffs = []

        for i in range(start, len(hist)):
            diffs.append(float(hist[i]) - float(hist[i - 1]))

        if len(diffs) < 5:
            return None

        mean_diff = sum(diffs) / len(diffs)
        var = sum((x - mean_diff) ** 2 for x in diffs) / max(len(diffs) - 1, 1)
        vol = math.sqrt(var)

        if vol <= 1e-9:
            return None

        return short_ret / (vol * math.sqrt(lookback))

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

    def current_tte_days(self, state: TradingState) -> float:
        # At timestamp 0, TTE = 5.
        # At timestamp 1_000_000, TTE = 4.
        # This protects against holding a fake fixed day-5 time value.
        elapsed_days = state.timestamp / self.TIMESTAMPS_PER_DAY
        return max(0.0, self.START_T_DAYS - elapsed_days)

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
        return max(0, min(int(desired), limit - position))

    def clamp_sell(self, position: int, limit: int, desired: int) -> int:
        return max(0, min(int(desired), limit + position))

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

    # ---- HYDROGEL swing-trader constants (from 16kbase.py) ----
    HYDRO_FAIR = 10000.0
    HYDRO_SIGMA = 32.0
    HYDRO_LIMIT = 200
    HYDRO_ENTRY_Z_1 = 0.85
    HYDRO_ENTRY_Z_2 = 1.25
    HYDRO_ENTRY_Z_3 = 1.65
    HYDRO_SIZE_1 = 120
    HYDRO_SIZE_2 = 160
    HYDRO_SIZE_3 = 200
    HYDRO_REVERSAL_Z = 3.35
    HYDRO_REBOUND_Z = 1.20
    HYDRO_MAX_TRADE = 40
    HYDRO_COOLDOWN = 1200
    HYDRO_RESET_Z = 0.35

    def _hydro_desired_short_size(self, z: float) -> int:
        if z >= self.HYDRO_ENTRY_Z_3:
            return self.HYDRO_SIZE_3
        if z >= self.HYDRO_ENTRY_Z_2:
            return self.HYDRO_SIZE_2
        if z >= self.HYDRO_ENTRY_Z_1:
            return self.HYDRO_SIZE_1
        return 0

    def _hydro_target(self, mid: float, timestamp: int, hd: Dict) -> int:
        """Compute target HYDROGEL position from the swing-trader state machine.
        hd is the per-strategy state dict held inside mem["hydro"]."""
        mode = hd.get("mode", "flat")
        extreme = hd.get("extreme")
        cooldown_until = int(hd.get("cooldown_until", -1))
        needs_reset = bool(hd.get("needs_reset", False))

        z = (mid - self.HYDRO_FAIR) / self.HYDRO_SIGMA
        reversal = self.HYDRO_REVERSAL_Z * self.HYDRO_SIGMA
        rebound = self.HYDRO_REBOUND_Z * self.HYDRO_SIGMA
        reset_band = self.HYDRO_RESET_Z * self.HYDRO_SIGMA
        reset_low = self.HYDRO_FAIR - reset_band
        reset_high = self.HYDRO_FAIR + reset_band

        if mode == "flat":
            hd["extreme"] = None
            hd["short_size"] = 0
            if needs_reset:
                if reset_low <= mid <= reset_high:
                    hd["needs_reset"] = False
                    needs_reset = False
                else:
                    return 0
            if timestamp < cooldown_until:
                return 0
            short_size = self._hydro_desired_short_size(z)
            if short_size > 0:
                hd["mode"] = "short"
                hd["extreme"] = mid
                hd["short_size"] = short_size
                return -short_size
            return 0

        if mode == "short":
            high = max(float(extreme or mid), mid)
            hd["extreme"] = high
            old_short_size = int(hd.get("short_size", self.HYDRO_SIZE_1))
            desired_short_size = self._hydro_desired_short_size(z)
            # Only scale IN, never scale OUT while short.
            short_size = max(old_short_size, desired_short_size)
            if short_size <= 0:
                short_size = self.HYDRO_SIZE_1
            short_size = min(short_size, self.HYDRO_LIMIT)
            hd["short_size"] = short_size
            if mid <= high - reversal:
                hd["mode"] = "long"
                hd["extreme"] = mid
                hd["short_size"] = 0
                return self.HYDRO_LIMIT
            return -short_size

        if mode == "long":
            low = min(float(extreme or mid), mid)
            hd["extreme"] = low
            if mid >= low + rebound:
                hd["mode"] = "flat"
                hd["extreme"] = None
                hd["cooldown_until"] = timestamp + self.HYDRO_COOLDOWN
                hd["needs_reset"] = True
                hd["short_size"] = 0
                return 0
            return self.HYDRO_LIMIT

        # Unknown mode — recover to flat.
        hd["mode"] = "flat"
        hd["extreme"] = None
        hd["needs_reset"] = False
        hd["short_size"] = 0
        return 0

    def trade_hydrogel(self, state: TradingState, mem: Dict) -> List[Order]:
        """HYDROGEL swing trader (replaces v3_fix's MM with 16kbase logic).

        Mechanism:
          - Fixed fair=10000, σ=32. Z-score = (mid - fair) / σ.
          - From flat: enter SHORT only when rich (z >= 0.85). Ramp size with z.
          - Once short: hold/scale in. Wait for big reversal (mid drops 3.35σ
            from peak) → flip to FULL LONG (+200).
          - From long: rebound 1.20σ → flat + cooldown.
          - Cross the spread to fill. MAX_TRADE=40 lots/tick.
        """
        product = "HYDROGEL_PACK"
        depth = state.order_depths.get(product)
        if depth is None:
            return []

        bb = max(depth.buy_orders) if depth.buy_orders else None
        ba = min(depth.sell_orders) if depth.sell_orders else None
        if bb is None or ba is None:
            return []

        mid = (bb + ba) / 2.0
        position = int(state.position.get(product, 0))

        # Pull/initialize per-strategy state inside mem.
        hd = mem.setdefault("hydro", {
            "mode": "flat",
            "extreme": None,
            "cooldown_until": -1,
            "needs_reset": False,
            "short_size": 0,
        })
        target = self._hydro_target(mid, int(state.timestamp), hd)

        buy_cap = max(0, self.HYDRO_LIMIT - position)
        sell_cap = max(0, self.HYDRO_LIMIT + position)

        orders: List[Order] = []
        if target > position and buy_cap > 0:
            visible = max(0, -depth.sell_orders.get(ba, 0))
            qty = min(target - position, buy_cap, self.HYDRO_MAX_TRADE, visible)
            if qty > 0:
                orders.append(Order(product, ba, qty))
        elif target < position and sell_cap > 0:
            visible = max(0, depth.buy_orders.get(bb, 0))
            qty = min(position - target, sell_cap, self.HYDRO_MAX_TRADE, visible)
            if qty > 0:
                orders.append(Order(product, bb, -qty))
        return orders

    def trade_extract(
        self,
        state: TradingState,
        mem: Dict,
    ) -> Tuple[List[Order], Optional[float]]:
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

        mid = self.get_mid(depth)
        if mid is not None:
            self.append_hist(mem, product, mid)

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

    def trade_voucher_itm(
        self,
        state: TradingState,
        product: str,
        strike: int,
    ) -> List[Order]:
        """Specialized passive MM for deep-ITM vouchers (VEV_4000, VEV_4500).

        Replaces trade_voucher_baseline for these strikes. Key differences:
          - Uses min_quote_spread filter to skip "flicker" ticks (spread<12/14)
            that have adverse selection but no genuine bot flow.
          - Quotes anchored to best_bid+1 / best_ask-1 (not BS theo, which is
            unstable for near-intrinsic deep ITM).
          - No fair-value-based take logic; just simple inside-the-touch making.
        """
        depth = state.order_depths.get(product)
        if depth is None:
            return []
        cfg = self.ITM_VOUCHER_CONFIG[strike]

        bb, ba = self.best_bid_ask(depth)
        if bb is None or ba is None:
            return []

        spread = ba - bb
        touch_mid = (bb + ba) / 2.0
        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]

        # Hard inventory safety net (rarely binds in normal operation).
        if position > cfg["hard_inventory_limit"]:
            return self._itm_cross_to_neutralize(product, depth, position, touch_mid, cfg, "sell")
        if position < -cfg["hard_inventory_limit"]:
            return self._itm_cross_to_neutralize(product, depth, position, touch_mid, cfg, "buy")

        # Skip if spread too tight (flicker filter).
        if spread < cfg["min_quote_spread"]:
            return []

        # Quote inside the touch.
        bid_price = bb + cfg["improve_ticks"]
        ask_price = ba - cfg["improve_ticks"]
        if bid_price >= ba:
            bid_price = ba - 1
        if ask_price <= bb:
            ask_price = bb + 1

        bid_qty = self.clamp_buy(position, limit, cfg["quote_size"])
        ask_qty = self.clamp_sell(position, limit, cfg["quote_size"])

        orders: List[Order] = []
        if bid_qty > 0:
            orders.append(Order(product, int(bid_price), int(bid_qty)))
        if ask_qty > 0:
            orders.append(Order(product, int(ask_price), -int(ask_qty)))
        return orders

    def _itm_cross_to_neutralize(self, product, depth, position, touch_mid, cfg, side):
        """Emergency cross-spread liquidation if hard limit breached on ITM voucher."""
        target_qty = abs(position) - cfg["hard_inventory_limit"] // 2
        if target_qty <= 0:
            return []
        orders: List[Order] = []
        if side == "sell":
            for price in sorted(depth.buy_orders.keys(), reverse=True):
                if price < touch_mid - cfg["take_distance"]:
                    break
                avail = int(depth.buy_orders[price])
                qty = min(target_qty, max(0, avail))
                if qty <= 0:
                    continue
                orders.append(Order(product, int(price), -int(qty)))
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
                orders.append(Order(product, int(price), int(qty)))
                target_qty -= qty
                if target_qty <= 0:
                    break
        return orders

    def trade_voucher_baseline(
        self,
        state: TradingState,
        product: str,
        strike: int,
        s_fair: float,
    ) -> List[Order]:
        """
        Original r3_test-style voucher logic.

        This is kept for:
        VEV_4000, VEV_4500, VEV_6000, VEV_6500.
        """
        depth = state.order_depths.get(product)
        if depth is None:
            return []

        best_bid, best_ask = self.best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return []

        tte_days = self.current_tte_days(state)
        sigma = self.BASE_SIGMA * self.SMILE_RATIO[strike]
        theo = self.bs_call(s_fair, float(strike), sigma, tte_days)

        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        spread = best_ask - best_bid

        fair = theo - 0.10 * position

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
    
    # Helper function 
    def short_return_from_hist(
        self,
        mem: Dict,
        key: str,
        lookback: int,
    ) -> Optional[float]:
        hist = mem.get("hist", {}).get(key, [])

        if len(hist) < lookback + 1:
            return None

        return float(hist[-1]) - float(hist[-1 - lookback])

    def trade_voucher_improved(
        self,
        state: TradingState,
        mem: Dict,
        product: str,
        strike: int,
        s_fair: float,
    ) -> List[Order]:
        """
        New combined strategy for VEV_5000 to VEV_5500.

        Core idea:
        1. Use dynamic BS fair value.
        2. Compute residual = option_mid - BS_fair.
        3. Use residual jump/reversal signal.
        4. Blend residual signal with VELVET signal.
        5. Trade aggressively only when signal and fair-value residual agree.
        6. Still do small passive market making near fair when no strong signal.
        """
        depth = state.order_depths.get(product)
        if depth is None:
            return []

        best_bid, best_ask = self.best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return []

        mid = self.get_mid(depth)
        if mid is None:
            return []

        mid_key = f"{product}_mid"
        self.append_hist(mem, mid_key, mid)

        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        spread = best_ask - best_bid

        cfg = self.IMPROVED_OPTION_CONFIG[strike]

        tte_days = self.current_tte_days(state)
        sigma = self.BASE_SIGMA * self.SMILE_RATIO[strike]
        theo = self.bs_call(s_fair, float(strike), sigma, tte_days)

        # Inventory-adjusted fair for execution.
        fair = theo - cfg["inv_skew"] * position

        # Residual should use non-inventory-adjusted theoretical fair.
        residual = mid - theo
        residual_key = f"{product}_residual"
        self.append_hist(mem, residual_key, residual)

        residual_signal = self.jump_score_from_hist(
            mem=mem,
            key=residual_key,
            lookback=self.RESID_LOOKBACK,
            vol_window=self.RESID_VOL_WINDOW,
        )

        velvet_signal = self.jump_score_from_hist(
            mem=mem,
            key="VELVETFRUIT_EXTRACT",
            lookback=self.VELVET_LOOKBACK,
            vol_window=self.VELVET_VOL_WINDOW,
        )

        if residual_signal is None:
            residual_signal = 0.0
        if velvet_signal is None:
            velvet_signal = 0.0

        combined_signal = (
            cfg["velvet_weight"] * velvet_signal
            + cfg["residual_weight"] * residual_signal
        )

        entry_z = cfg["entry_z"]
        strong_z = cfg["strong_z"]
        base_size = cfg["base_size"]
        take_edge = cfg["take_edge"]

        quote_edge = max(cfg["quote_edge_min"], spread // 2)

        # Fair-value residual threshold.
        # This prevents buying merely because of reversal signal while option is still rich,
        # and prevents selling merely because of reversal signal while option is still cheap.

        orders: List[Order] = []
        pos = position

        theo_fair = theo
        exec_fair = theo - cfg["inv_skew"] * position

        fair = exec_fair

        resid_edge = max(0.5, cfg["resid_edge_mult"] * spread)
        guard = self.OPTION_TREND_GUARD[strike]

        option_short_ret = self.short_return_from_hist(
            mem=mem,
            key=mid_key,
            lookback=guard["lookback"],
        )

        if option_short_ret is None:
            option_short_ret = 0.0

        falling_fast = option_short_ret < -guard["threshold"]
        rising_fast = option_short_ret > guard["threshold"]

        # Residual short-return: how the residual itself has moved over the lookback.
        # Used by the position-aware guards below so we don't keep adding to an
        # already-large position when the trade is already going our way (and the
        # signal hasn't flipped yet because the lookback window still includes the
        # initial spike).
        residual_short_ret = self.short_return_from_hist(
            mem=mem,
            key=residual_key,
            lookback=guard["lookback"],
        )
        if residual_short_ret is None:
            residual_short_ret = 0.0

        if combined_signal < -entry_z:
            # Symmetric guard:
            # Do not add new long exposure while the option is already rising fast.
            # This prevents buying near the top during a sharp rise.
            if rising_fast and pos >= 0:
                return orders

            # HARD inventory cap on the long side, symmetric to SELL branch.
            HARD_LONG_CAP = limit // 5
            if pos >= HARD_LONG_CAP:
                return orders

            # Soft guard: if meaningfully long and trade is already going our
            # way (option rising or residual normalizing), skip new buys.
            if pos > base_size and (option_short_ret > 0 or residual_short_ret >= 0):
                return orders

            desired_qty = base_size

            # If already short, allow reducing short, but do not flip aggressively long during a fast rise.
            if rising_fast and pos < 0:
                desired_qty = min(desired_qty, abs(pos))

            # If option is truly cheap, cross/take more willingly.
            if residual < -resid_edge:
                max_buy_price = theo_fair + take_edge
            else:
                # Still allow entry, but only at/under theoretical fair.
                max_buy_price = theo_fair

            if abs(combined_signal) >= strong_z or residual < -resid_edge:
                for ask in sorted(depth.sell_orders.keys()):
                    ask_qty = -depth.sell_orders[ask]

                    if ask <= max_buy_price:
                        qty = self.clamp_buy(pos, limit, min(ask_qty, desired_qty))
                        if qty > 0:
                            orders.append(Order(product, ask, qty))
                            pos += qty
                            desired_qty -= qty

                    if desired_qty <= 0:
                        break

            # Passive bid even if residual is not very negative.
            if desired_qty > 0:
                buy_px = int(math.floor(exec_fair - quote_edge))
                buy_px = min(buy_px, best_bid + 1)

                # If signal is strong, allow quote at fair, not too far below.
                if abs(combined_signal) >= strong_z:
                    buy_px = min(buy_px, int(math.floor(theo_fair)))
                else:
                    buy_px = min(buy_px, int(math.floor(theo_fair - 0.5)))

                if buy_px < best_ask:
                    qty = self.clamp_buy(pos, limit, desired_qty)
                    if qty > 0:
                        orders.append(Order(product, buy_px, qty))
                        pos += qty
        
        elif combined_signal > entry_z:
            if falling_fast and pos <= 0:
                return orders

            # HARD inventory cap: never accumulate beyond 20% of position limit
            # on the short side from this branch. The death spiral observed on
            # VEV_5000 / 5100 in v3 piled shorts up to the -300 hard limit, then
            # any price reversal ate the full gain (-$38K voucher block on day
            # 2). Hard-cap at limit/5 (60 for vouchers) prevents that and lifts
            # total PnL by ~$29K (from $11K -> $40K) on Round 3 backtest.
            HARD_SHORT_CAP = limit // 5   # 60 for vouchers (limit=300)
            if pos <= -HARD_SHORT_CAP:
                return orders

            # Soft guard: if meaningfully short and trade is already going our
            # way (option dropping or residual normalizing), skip new shorts.
            if pos < -base_size and (option_short_ret < 0 or residual_short_ret <= 0):
                return orders

            desired_qty = base_size

            # If already long, allow reducing long, but do not flip aggressively short during a fast drop.
            if falling_fast and pos > 0:
                desired_qty = min(desired_qty, pos)

            # If option is truly rich, cross/take more willingly.
            if residual > resid_edge:
                min_sell_price = theo_fair - take_edge
            else:
                # Still allow entry, but only at/above theoretical fair.
                min_sell_price = theo_fair

            if abs(combined_signal) >= strong_z or residual > resid_edge:
                for bid in sorted(depth.buy_orders.keys(), reverse=True):
                    bid_qty = depth.buy_orders[bid]

                    if bid >= min_sell_price:
                        qty = self.clamp_sell(pos, limit, min(bid_qty, desired_qty))
                        if qty > 0:
                            orders.append(Order(product, bid, -qty))
                            pos -= qty
                            desired_qty -= qty

                    if desired_qty <= 0:
                        break

            # Passive ask even if residual is not very positive.
            if desired_qty > 0:
                sell_px = int(math.ceil(exec_fair + quote_edge))
                sell_px = max(sell_px, best_ask - 1)

                # If signal is strong, allow quote at fair, not too far above.
                if abs(combined_signal) >= strong_z:
                    sell_px = max(sell_px, int(math.ceil(theo_fair)))
                else:
                    sell_px = max(sell_px, int(math.ceil(theo_fair + 0.5)))

                if sell_px > best_bid:
                    qty = self.clamp_sell(pos, limit, desired_qty)
                    if qty > 0:
                        orders.append(Order(product, sell_px, -qty))
                        pos -= qty

        else:
            # Calm regime: smaller passive market making.
            fair_adj = exec_fair - cfg["inv_skew"] * pos

            buy_px = int(math.floor(fair_adj - quote_edge))
            sell_px = int(math.ceil(fair_adj + quote_edge))

            buy_px = min(buy_px, best_bid + 1)
            sell_px = max(sell_px, best_ask - 1)

            passive_size = max(1, base_size // 2)

            if buy_px < best_ask:
                buy_qty = self.clamp_buy(pos, limit, passive_size)
                if buy_qty > 0:
                    orders.append(Order(product, buy_px, buy_qty))
                    pos += buy_qty

            if sell_px > best_bid:
                sell_qty = self.clamp_sell(pos, limit, passive_size)
                if sell_qty > 0:
                    orders.append(Order(product, sell_px, -sell_qty))
                    pos -= sell_qty

        return orders

    def run(self, state: TradingState):
        mem = self.load_memory(state.traderData)

        result: Dict[str, List[Order]] = {}

        hydro_orders = self.trade_hydrogel(state, mem)
        if hydro_orders:
            result["HYDROGEL_PACK"] = hydro_orders

        # IMPORTANT: trade_extract returns s_fair (wall_mid + EMA-based fair value
        # for VELVETFRUIT). The voucher BS-momentum logic relies on s_fair as the
        # underlying anchor for theoretical pricing, so we use the same value here.
        extract_orders, s_fair = self.trade_extract(state, mem)
        if extract_orders:
            result["VELVETFRUIT_EXTRACT"] = extract_orders

        # v4_full: trade everything we have a strategy for.
        #   - VEV_4000/4500 → specialized ITM MM
        #   - VEV_5000-5500 → BS+momentum (with velvet_weight=0, hard inventory cap)
        #   - VEV_6000/6500 → still skipped (dead products)
        for product, strike in self.VOUCHER_STRIKES.items():
            if product in self.ITM_VOUCHER_PRODUCTS:
                voucher_orders = self.trade_voucher_itm(
                    state=state,
                    product=product,
                    strike=strike,
                )
            elif product in self.IMPROVED_OPTION_PRODUCTS and s_fair is not None:
                voucher_orders = self.trade_voucher_improved(
                    state=state,
                    mem=mem,
                    product=product,
                    strike=strike,
                    s_fair=s_fair,
                )
            else:
                # VEV_6000/6500: dead, skip.
                voucher_orders = []
            if voucher_orders:
                result[product] = voucher_orders

        conversions = 0
        trader_data = self.save_memory(mem)
        return result, conversions, trader_data
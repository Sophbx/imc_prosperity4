"""
ITM voucher trader (VEV_4000 + VEV_4500). Round 3, Prosperity 4.

Scope: Handles ONLY VEV_4000 and VEV_4500 (the deep ITM vouchers, delta=1).
The teammate handles the other products (HYDROGEL, VELVETFRUIT, VEV_5000-5400, etc.)
and will integrate this trader's output via product-set merging.

==========
KEY FINDINGS FROM RESEARCH (see report for details)
==========

VEV_4000:
  * 21-tick spread, 100% delta, 172/164/128 bot trades per day.
  * Combined v1 already captures every bot trade (improve+1 over touch).
  * The marginal improvement comes from removing the EOD flatten window:
    combined v1 widened quotes (post bb-2/ba+2) in the last 200 ticks of each
    day, effectively halting MM. Removing that yields +$210 over 3 days.
  * Adding a min_quote_spread filter of 12 (vs default 6) avoids the ~1.8% of
    "flicker" ticks where spread briefly tightens to 7-12 (1-2 tick events
    that carry adverse selection without bot flow). Yields +$35.
  * No further structural improvements: improve_ticks=1 is optimal,
    quote_size doesn't bind (bot trades are qty 1-3), inventory limits
    don't bind (position never exceeds +-50 in backtest), and signals like
    frontier_imbalance/depth_imbalance/synth_fair don't help fill PnL
    (the simulator only fills based on price priority, not imbalance).
  * Delta hedging via VELVETFRUIT loses $750 in spread cost — confirmed
    matches the prior research note finding.

VEV_4500:
  * 16-tick spread, 100% delta, 1 bot trade across 3 days (zero flow).
  * The book exists, but bots never aggress. Maker MM has no fills.
  * Static intrinsic arbitrage: ask never drops > 2 ticks below intrinsic;
    not enough to overcome cross-spread cost.
  * Stat-arb via VELVETFRUIT: net buy_edge after V hedge round-trip is
    negative; no signal.
  * The 1.8% of tight-spread ticks (spread 7-9) are 1-2 tick flickers,
    typically due to a counterparty briefly improving the touch then
    leaving. We DO occasionally get filled there in combined v1's logic
    (default min_quote_spread=6), and that fill is unfavorable
    (-$33.50 over 3 days). Filter min_quote_spread>=14 → 0 fills, 0 loss.
  * **Verdict: VEV_4500 is structurally untradable. Include with strict
    min_quote_spread=14 as a no-op safety net; if live behavior changes,
    we miss out on a few ticks but don't lose money.**

==========
EXPECTED PERFORMANCE (3-day backtest)
==========

Combined v1 (baseline):    VEV_4000=$8635.5  VEV_4500=$0     Total=$8635.5
This trader (trader_itm_v1): VEV_4000=$8845    VEV_4500=$0     Total=$8845
Improvement: +$209.5 (+2.4%) over 3 days.

Per-day:
  Day 0: $3011 (vs $2942 baseline, +$69)
  Day 1: $3291 (vs $3296.5 baseline, -$5.5)
  Day 2: $2543 (vs $2397 baseline, +$146)
"""
from datamodel import Order, OrderDepth, TradingState


# Symbols
VEV_4000 = "VEV_4000"
VEV_4500 = "VEV_4500"

# Position limits
LIMIT_VOUCHER = 300

# VEV_4000 config
# - improve_ticks=1: post inside the touch by 1 (bb+1 / ba-1). Optimal: any
#   wider sacrifices spread per RT, any tighter (improve=0) yields 0 fills.
# - min_quote_spread=12: filter the ~1.8% of tight-spread flicker ticks
#   (spread 7-12). Normal regime is spread 20-22; we only act in normal regime.
# - quote_size=10: caps fill volume at 10 lots; bot trades are qty 1-3 so
#   this is sufficient. Larger sizes don't help.
# - hard_inventory_limit=200: safety net only (position never exceeds +-50
#   in backtest). At 200, we cross-spread to neutralize.
VEV_4000_CFG = {
    "improve_ticks": 1,
    "min_quote_spread": 12,
    "quote_size": 10,
    "hard_inventory_limit": 200,
    "take_distance": 4,  # for emergency unwind
}

# VEV_4500 config
# - min_quote_spread=14: strict filter; only quote during normal-spread regime
#   (spread 15-17). At spread<14 we skip — those are flicker ticks where any
#   fill is adversely selected (cf. baseline -$33.50 day 1 trade).
# - In backtest (3 days), this config yields 0 fills 0 PnL — explicitly safer
#   than combined v1's -$33 loss. Live: if behavior shifts, we re-engage.
VEV_4500_CFG = {
    "improve_ticks": 1,
    "min_quote_spread": 14,
    "quote_size": 10,
    "hard_inventory_limit": 200,
    "take_distance": 4,
}


class Trader:
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}

        # VEV_4000: passive MM, the workhorse
        depth = state.order_depths.get(VEV_4000)
        if depth is not None:
            pos = int(state.position.get(VEV_4000, 0))
            orders = self._mm_delta1(VEV_4000, depth, pos, LIMIT_VOUCHER, VEV_4000_CFG)
            if orders:
                result[VEV_4000] = orders

        # VEV_4500: passive MM, but only at wide spread (no-op in normal regime)
        depth = state.order_depths.get(VEV_4500)
        if depth is not None:
            pos = int(state.position.get(VEV_4500, 0))
            orders = self._mm_delta1(VEV_4500, depth, pos, LIMIT_VOUCHER, VEV_4500_CFG)
            if orders:
                result[VEV_4500] = orders

        return result, 0, ""

    def _mm_delta1(self, sym, depth, position, limit, cfg):
        """Passive MM for delta=1 ITM vouchers.
        Strategy:
          - Post inside the touch: bid at bb+improve_ticks, ask at ba-improve_ticks.
          - Skip the tick if spread < min_quote_spread (filter flickers).
          - Cross-spread to neutralize if |position| > hard_inventory_limit (safety).
        """
        bb, ba = self._best_prices(depth)
        if bb is None or ba is None:
            return []
        spread = ba - bb
        touch_mid = (bb + ba) / 2.0

        # Hard inventory safety net (rarely binds in normal operation)
        if position > cfg["hard_inventory_limit"]:
            return self._cross_to_neutralize(sym, depth, position, touch_mid, cfg, "sell")
        if position < -cfg["hard_inventory_limit"]:
            return self._cross_to_neutralize(sym, depth, position, touch_mid, cfg, "buy")

        # Skip if spread is too tight (flicker filter)
        if spread < cfg["min_quote_spread"]:
            return []

        # Place quotes at bb+1 / ba-1 (inside the touch, never crossing)
        bid_price = bb + cfg["improve_ticks"]
        ask_price = ba - cfg["improve_ticks"]
        if bid_price >= ba:
            bid_price = ba - 1
        if ask_price <= bb:
            ask_price = bb + 1

        # Sizing: simple uniform quote_size, capped by remaining room to limit
        bid_room = max(0, limit - position)
        ask_room = max(0, limit + position)
        bid_qty = min(cfg["quote_size"], bid_room)
        ask_qty = min(cfg["quote_size"], ask_room)

        orders = []
        if bid_qty > 0:
            orders.append(Order(sym, int(bid_price), int(bid_qty)))
        if ask_qty > 0:
            orders.append(Order(sym, int(ask_price), -int(ask_qty)))
        return orders

    def _cross_to_neutralize(self, sym, depth, position, touch_mid, cfg, side):
        """Cross-spread market orders to liquidate inventory if hard limit breached.
        This is a safety net only; in normal operation this path doesn't fire.
        """
        target_qty = abs(position) - cfg["hard_inventory_limit"] // 2  # walk back to half-limit
        if target_qty <= 0:
            return []
        orders = []
        if side == "sell":
            for price in sorted(depth.buy_orders.keys(), reverse=True):
                if price < touch_mid - cfg["take_distance"]:
                    break
                avail = int(depth.buy_orders[price])
                qty = min(target_qty, max(0, avail))
                if qty <= 0:
                    continue
                orders.append(Order(sym, int(price), -int(qty)))
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
                orders.append(Order(sym, int(price), int(qty)))
                target_qty -= qty
                if target_qty <= 0:
                    break
        return orders

    @staticmethod
    def _best_prices(depth):
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        return best_bid, best_ask

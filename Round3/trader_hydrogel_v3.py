"""
HYDROGEL_PACK — EMA-anchored mean-reversion quoter (v3).

================================================================================
THESIS — step-change vs. v1/v2/v2b
================================================================================

v1/v2/v2b are passive market makers that quote at touch ± improve and skew
by inventory only. They cluster around $23-24K total PnL on 3-day historicals.
The empirical research established that HYDROGEL has STRONG mean reversion:

  - lag-1 autocorrelation of returns: -0.13 to -0.14 (all 3 days)
  - variance ratio at k=500 ticks: 0.33-0.47 (random walk would be 1.0)
  - touch_mid - EMA(2000) shows reliable predictability:
        |dev| > 4 ticks predicts -3.8 to +5.2 ticks fwd-100
  - 67-75% of ticks have |dev| > 10 ticks

But TAKING this signal is unprofitable: HYDROGEL spread is 16 ticks (93% of
ticks), and the touch is 8 ticks from mid — to take we cross 8 ticks, only
to capture a 4-5 tick reversion.

The insight: capture the signal **as a maker by aggressing the favorable
side and disabling the unfavorable side**. When dev > +threshold (touch is
rich vs. EMA), shift BOTH bid and ask DOWN by 1 tick:

  bid_price = best_bid + improve - 1 = best_bid       -> not filled (sim
                                                          fills strict-inside)
  ask_price = best_ask - improve - 1 = best_ask - 2   -> very aggressive sell

So on rich-dev ticks the bid effectively turns OFF (parked at touch, no fill)
and the ask becomes a very aggressive sell. Symmetric on cheap-dev ticks.
This is one-sided market-making aligned with the mean-reversion direction —
no taking required.

================================================================================
BACKTEST RESULTS (3 days, GeyzsoN simulator)
================================================================================

  Variant            | Total      | Day 0   | Day 1   | Day 2  | DD     | Sharpe
  -------------------|------------|---------|---------|--------|--------|-------
  v1                 |  $23,978   |  -      |  -      |  -     | -7,561 | 1.37
  v2 (tighter bands) |  $23,309   |  8,160  | 12,294  |  2,855 | -3,969 | 1.64
  v2b (Wall+A-S)     |  $23,478   |  -      |  -      |  -     | -2,437 | 1.59
  v3 (this trader)   | **$46,645**| 16,827  | 21,319  |  8,499 | -7,415 | ~1.5

  v3 nearly doubles total PnL. Day 2 (worst day for v2: $2,855) goes to
  $8,499 — a 3x improvement. Every day improves significantly.

The signal is robust across:
  - EMA window 1000-2000 ($43-46K, peak at 2000)
  - dev_threshold 3-5 ($46K all)
  - SMA vs. EMA (within 0.5%, EMA preferred for runtime)

It is NOT robust to:
  - Larger shift_size (>1) — crossing too aggressive, loses fills
  - Multiple shift levels — added complexity, hurts PnL
  - Adding a stat-arb taker — fwd reversion < cross cost, money-losing
  - improve_ticks=0 — simulator only fills strict-inside, no fills

================================================================================
PARAMETERS — chosen from empirical sweep, not curve-fit
================================================================================

  ema_window = 2000   : robust plateau 800-2000; longer windows preserve
                        signal stability vs. noisy short EMAs
  dev_threshold = 3   : binary trigger; threshold doesn't change shift
                        magnitude, only its activation rate; values 3-5 give
                        within $200 of each other
  shift_size = 1      : critical — shifts >1 tick cross too far past the
                        spread floor, losing more in adverse fills than
                        gained in mean reversion
  asym_factor = 3.0   : quote bigger on the favorable side; saturated at
                        ~3, marginal gain ~$300
  improve_ticks = 1   : forced by simulator (only fills strict-inside-touch)
  quote_size = 12     : doesn't matter much above 12 — fills are demand-
                        limited not size-limited
  skew_scale = 200    : effectively disables inventory size skew; the EMA
                        shift handles inventory naturally because position
                        accumulates only when one side is preferentially
                        filled, and the EMA shift turns off the other side
                        on the next big dev move
  hard_inventory_limit = 180 : safety cross-book trigger if EMA shift
                        somehow fails to manage inventory; in practice
                        position trajectory peaks at |q| ≤ 132 across all
                        3 days, so this never fires
  flatten at 980_000  : end-of-day position-flatten quotes inherited from
                        v1/v2 as universal hygiene

================================================================================
WHAT WAS TRIED AND ABANDONED
================================================================================

  * improve_ticks=0 (post AT touch). 18 fills total over 3 days. Simulator
    only fills strict-inside-touch. Untestable; might work live but no way
    to validate.
  * Stat-arb taker on |dev| > threshold. Confirmed unprofitable in backtest:
    walking the book to take touch crosses 8 ticks; the predicted reversion
    is 4-5 ticks; net -$45K disaster on aggressive taker tests.
  * Layered MM (post at improve+1 AND at deeper level). Deep layer never
    fills under the simulator's strict-inside rule. Identical to baseline.
  * Multi-level shifts (shift=1 at thr1=4, shift=2 at thr2=15). Worse than
    single-level by $400-1500. Two-tick crosses are too aggressive.
  * Frontier-imbalance taker. Only ~16-21 events/day at threshold 0.4 (not
    240/day as suggested in old notes). Even if alpha is real, sample size
    too small to add meaningful PnL.
  * SMA replacement for EMA. Within 0.5% of EMA. EMA preferred — JSON
    persistence of a 2000-element rolling buffer is wasteful.
  * Z-score band taking. |z|>2 events ~500/day with fwd-50 reversion of
    5-6 ticks. Same arithmetic as stat-arb taker — cross cost > reversion.

================================================================================
References
================================================================================
- Round3/Analysis/output/ — original signal research
- /tmp/hydrogel_analyze.py — pre-strategy signal verification script
- Empirical parameter sweeps: /tmp/sweep[1-9]*.py
"""
from datamodel import Order, OrderDepth, TradingState
import json


class Trader:
    HYDROGEL = "HYDROGEL_PACK"
    POSITION_LIMITS = {HYDROGEL: 200}

    CFG = {
        # ---- Quote placement ----
        "improve_ticks": 1,             # post 1 tick inside the touch
        "min_quote_spread": 6,          # only quote if market spread >= this
        "quote_size": 12,               # base clip per side
        "min_quote_size": 2,            # minimum even when heavily skewed

        # ---- Inventory management (mostly inactive at our typical positions) ----
        "skew_inventory_scale": 200,    # wide scale -> EMA shift does the work
        "unwind_inventory_limit": 150,  # quote-mode "unwind" trigger
        "hard_inventory_limit": 180,    # cross-book emergency trigger
        "take_distance": 4,             # max walk distance for emergency take

        # ---- End-of-day flatten ----
        "flatten_window_start_ts": 980_000,
        "flatten_quote_offset": 2,

        # ---- EMA mean-reversion shift (the new alpha) ----
        # alpha computed from window: alpha = 2/(window+1)
        "ema_window": 2000,             # long window — captures equilibrium
        "ema_warmup_ticks": 2000,       # only shift after warmup
        "dev_threshold": 3,             # |touch_mid - EMA| > 3 -> shift
        "shift_size": 1,                # 1-tick shift — crucial, do not raise
        "asym_factor": 3.0,             # multiply favorable-side size by this
    }

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {self.HYDROGEL: []}
        depth = state.order_depths.get(self.HYDROGEL)
        if depth is None:
            return result, 0, state.traderData or ""

        td = self._decode(state.traderData)
        ema = td.get("ema")
        n_seen = td.get("n_seen", 0)

        position = int(state.position.get(self.HYDROGEL, 0))
        ts = int(state.timestamp)
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        if best_bid is None or best_ask is None:
            return result, 0, self._encode({"ema": ema, "n_seen": n_seen})

        touch_mid = (best_bid + best_ask) / 2.0
        # Update EMA (pure-Python, single multiply-add per tick)
        alpha = 2.0 / (self.CFG["ema_window"] + 1)
        if ema is None:
            ema = touch_mid
        else:
            ema = alpha * touch_mid + (1.0 - alpha) * ema
        n_seen += 1

        result[self.HYDROGEL] = self._trade(
            depth, position, ts, best_bid, best_ask, touch_mid, ema, n_seen,
        )
        return result, 0, self._encode({"ema": ema, "n_seen": n_seen})

    def _trade(self, depth, position, ts, best_bid, best_ask, touch_mid, ema, n_seen):
        cfg = self.CFG
        limit = self.POSITION_LIMITS[self.HYDROGEL]
        spread = best_ask - best_bid

        # Hard safety: cross book if extreme inventory.
        if position > cfg["hard_inventory_limit"]:
            return self._cross_book(depth, position, touch_mid, "sell")
        if position < -cfg["hard_inventory_limit"]:
            return self._cross_book(depth, position, touch_mid, "buy")

        # Need at least min_quote_spread to make profitably.
        if spread < cfg["min_quote_spread"]:
            return []

        # Mean-reversion shift (the v3 alpha).
        if n_seen >= cfg["ema_warmup_ticks"] and ema is not None:
            dev = touch_mid - ema
            if dev > cfg["dev_threshold"]:
                shift = -cfg["shift_size"]   # touch rich -> shift quotes DOWN
            elif dev < -cfg["dev_threshold"]:
                shift = +cfg["shift_size"]   # touch cheap -> shift quotes UP
            else:
                shift = 0
        else:
            shift = 0

        bid_room = max(0, limit - position)
        ask_room = max(0, limit + position)

        # Quote prices
        if ts > cfg["flatten_window_start_ts"]:
            # End-of-day flatten — narrow around mid, plus the EMA shift
            bid_price = int(round(touch_mid - cfg["flatten_quote_offset"] + shift))
            ask_price = int(round(touch_mid + cfg["flatten_quote_offset"] + shift))
        elif abs(position) > cfg["unwind_inventory_limit"]:
            # Emergency unwind via tight one-sided quote
            if position > 0:
                bid_price = int(round(best_bid + cfg["improve_ticks"] + shift))
                ask_price = int(round(touch_mid - 1 + shift))
            else:
                bid_price = int(round(touch_mid + 1 + shift))
                ask_price = int(round(best_ask - cfg["improve_ticks"] + shift))
        else:
            # Normal regime: improve touch by 1, then apply EMA shift
            bid_price = int(round(best_bid + cfg["improve_ticks"] + shift))
            ask_price = int(round(best_ask - cfg["improve_ticks"] + shift))

        # Never cross
        if bid_price >= best_ask:
            bid_price = best_ask - 1
        if ask_price <= best_bid:
            ask_price = best_bid + 1

        # Inventory-pressure size skew (mostly inactive with skew_scale=200)
        scale = cfg["skew_inventory_scale"]
        pressure = max(-2.0, min(2.0, position / scale))
        bid_factor = max(0.0, 1.0 - max(0.0, pressure))
        ask_factor = max(0.0, 1.0 + min(0.0, pressure))
        base = cfg["quote_size"]
        bid_qty = max(cfg["min_quote_size"], int(round(base * bid_factor)))
        ask_qty = max(cfg["min_quote_size"], int(round(base * ask_factor)))

        # Asymmetric size aligned with the EMA shift
        af = cfg["asym_factor"]
        if af > 1.0 and shift != 0:
            if shift < 0:
                # Touch rich -> we want to SELL more aggressively
                ask_qty = int(round(ask_qty * af))
                bid_qty = max(cfg["min_quote_size"], int(round(bid_qty / af)))
            else:
                # Touch cheap -> we want to BUY more aggressively
                bid_qty = int(round(bid_qty * af))
                ask_qty = max(cfg["min_quote_size"], int(round(ask_qty / af)))

        bid_qty = min(bid_qty, bid_room)
        ask_qty = min(ask_qty, ask_room)

        orders: list[Order] = []
        if bid_qty > 0:
            orders.append(Order(self.HYDROGEL, int(bid_price), int(bid_qty)))
        if ask_qty > 0:
            orders.append(Order(self.HYDROGEL, int(ask_price), -int(ask_qty)))
        return orders

    def _cross_book(self, depth, position, touch_mid, side):
        """Walk the book up to take_distance ticks past touch_mid to
        unwind back to within unwind_inventory_limit."""
        cfg = self.CFG
        target_qty = abs(position) - cfg["unwind_inventory_limit"]
        orders: list[Order] = []
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
    def _decode(s):
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            return {}

    @staticmethod
    def _encode(d):
        try:
            return json.dumps(d)
        except Exception:
            return ""

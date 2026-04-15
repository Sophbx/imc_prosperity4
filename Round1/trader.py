"""
IMC Prosperity 4 - Round 1 TEST trader (v1).

This is NOT a production strategy — it's an instrumentation tool.
Its job is to:

  1. Run cleanly end-to-end on the live Prosperity server for Round 1,
     so we get a real submission.log back.
  2. Run cleanly through the local community backtester (jmerle's P3
     backtester, patched for P4 products), so we can cross-check it
     against the live log.
  3. Produce enough structured debug output that we can answer our
     open Round 1 questions by just reading the log:
       - How many iterations does an upload test actually run?
         (Wiki says 1,000; our Round 0 logs said 2,000.)
       - Does the PEPPER_ROOT drift pattern we saw in sample data
         (+1,000/day) show up on the live day too?
       - Does OSMIUM show the ~4% one-sided book rate and ~0.15% empty
         book rate on the live day, or was that a sample-day artefact?
       - Does the live order book match the sample data at the same
         timestamps (replay), or is it a fresh randomisation?
  4. NOT lose meaningful money. Target: slightly-positive or near-zero
     PnL on OSMIUM, zero on PEPPER_ROOT.

Strategy split by product:

  INTARIAN_PEPPER_ROOT  —>  OBSERVE ONLY, no orders.
    The sample data shows a +1,000/day persistent drift across all
    three sample days. A naive market maker accumulates the wrong-side
    inventory against a trend and bleeds. We don't yet know the drift
    mechanism well enough to take a side, so this test trader does not
    trade PEPPER_ROOT at all. It only logs book state.

  ASH_COATED_OSMIUM     —>  Conservative Wall Mid market making.
    Wall Mid = (bid_wall + ask_wall) / 2 where bid_wall is the deepest
    visible bid price and ask_wall the deepest visible ask. Post one
    bid at bid_wall+1 and one ask at ask_wall-1 (both one tick inside
    the wall), size 3, with a self-imposed ±15 position cap. Skip a
    side entirely when the corresponding quote would not have strictly
    positive edge relative to wall_mid, when the book is one-sided, or
    when the book is empty.

Logging:
  Always prints a line on empty-book and one-sided-book events (these
  are rare and we want to catch all of them). Samples normal-state
  lines every 1,000 timestamp units (= every ~10 ticks) to keep the
  log compact.  The timestamp is always included so we can count
  iterations from the log after submission.

Self-imposed limits:
  SELF_IMPOSED_LIMIT = 15 per product, well below any plausible
  exchange limit. We do not rely on knowing the real Round 1 position
  limit.  If the exchange limit turns out to be lower than 15, we'll
  find out when orders start getting silently rejected — the log will
  show it because own_trades won't accumulate as expected.
"""

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List


class Trader:
    # --- product list ---
    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"
    PRODUCTS = [PEPPER, OSMIUM]

    # --- risk parameters ---
    SELF_IMPOSED_LIMIT = 15   # absolute position cap (we impose this, not IMC)
    ORDER_SIZE = 3            # size per MAKE quote

    # --- logging parameters ---
    LOG_SAMPLE_STEP = 1000    # print normal-state lines every N timestamp units
                              # (= every ~10 ticks, since ticks are 100 apart)

    def bid(self):
        """Required placeholder for Round 2's auction mechanic. Ignored elsewhere."""
        return 15

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        ts = state.timestamp

        # sample flag: true roughly every LOG_SAMPLE_STEP timestamp units.
        # Always also log on interesting events (empty / one-sided book).
        sample_line = (ts % self.LOG_SAMPLE_STEP == 0)

        for product in self.PRODUCTS:
            orders: List[Order] = []

            # ---- defensive: product missing from state ----
            if product not in state.order_depths:
                # Always log this — it would be very surprising.
                print(f"[t={ts}] {product} NOT_IN_STATE")
                result[product] = orders
                continue

            od = state.order_depths[product]
            position = state.position.get(product, 0)

            has_bids = bool(od.buy_orders)
            has_asks = bool(od.sell_orders)

            # ---- empty book (both sides missing) ----
            if not has_bids and not has_asks:
                print(f"[t={ts}] {product} EMPTY_BOOK pos={position}")
                result[product] = orders
                continue

            # ---- one-sided book ----
            if not has_bids or not has_asks:
                side = "no_bid" if not has_bids else "no_ask"
                print(f"[t={ts}] {product} ONE_SIDED {side} pos={position}")
                # Don't MAKE on a one-sided book — we'd be the only
                # liquidity on that level. Observe only this tick.
                result[product] = orders
                continue

            # ---- normal two-sided book: compute walls and wall_mid ----
            bid_wall = min(od.buy_orders.keys())       # deepest visible bid
            ask_wall = max(od.sell_orders.keys())      # deepest visible ask
            best_bid = max(od.buy_orders.keys())       # highest visible bid
            best_ask = min(od.sell_orders.keys())      # lowest visible ask
            wall_mid = (bid_wall + ask_wall) / 2
            wall_spread = ask_wall - bid_wall

            if sample_line:
                # One structured line per product per sample tick.
                # Short enough to stay inside any log size limits, verbose
                # enough that we can reconstruct a lot offline.
                print(
                    f"[t={ts}] {product} "
                    f"wm={wall_mid:g} "
                    f"bw={bid_wall} aw={ask_wall} sp={wall_spread} "
                    f"bb={best_bid} ba={best_ask} "
                    f"pos={position}"
                )

            # ---- PEPPER: observe only, no orders ----
            if product == self.PEPPER:
                result[product] = orders
                continue

            # ---- OSMIUM: conservative Wall Mid market making ----
            # One bid and one ask, each one tick inside the wall, as long
            # as the quote has strictly positive edge against wall_mid.

            max_buy = self.SELF_IMPOSED_LIMIT - position
            max_sell = self.SELF_IMPOSED_LIMIT + position

            mm_bid_price = bid_wall + 1
            mm_ask_price = ask_wall - 1

            # Bid side
            if mm_bid_price < wall_mid and max_buy > 0:
                qty = min(self.ORDER_SIZE, max_buy)
                orders.append(Order(product, mm_bid_price, qty))

            # Ask side
            if mm_ask_price > wall_mid and max_sell > 0:
                qty = min(self.ORDER_SIZE, max_sell)
                # Negative quantity => SELL order
                orders.append(Order(product, mm_ask_price, -qty))

            result[product] = orders

        # No persistent trader_data for this test version — keep it simple
        # so any weirdness in the submission log is obviously about the
        # platform / our assumptions, not about our state handling.
        traderData = ""
        conversions = 0
        return result, conversions, traderData

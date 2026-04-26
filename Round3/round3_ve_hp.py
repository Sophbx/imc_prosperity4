from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Optional, Tuple
import json


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
                    return data
            except Exception:
                pass

        return {
            "mode": "flat",
            "extreme": None,
            "cooldown_until": -1,
            "needs_reset": False,
            "short_size": 0,
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

    def _target(self, mid: float, timestamp: int, data: Dict) -> int:
        mode = data.get("mode", "flat")
        extreme = data.get("extreme")
        cooldown_until = int(data.get("cooldown_until", -1))
        needs_reset = bool(data.get("needs_reset", False))

        z = self._z(mid)

        reversal = self.REVERSAL_Z * self.SIGMA
        rebound = self.REBOUND_Z * self.SIGMA

        reset_band = self.RESET_Z * self.SIGMA
        reset_low = self.FAIR - reset_band
        reset_high = self.FAIR + reset_band

        if mode == "flat":
            data["extreme"] = None
            data["short_size"] = 0

            if needs_reset:
                if reset_low <= mid <= reset_high:
                    data["needs_reset"] = False
                    needs_reset = False
                else:
                    return 0

            if timestamp < cooldown_until:
                return 0

            short_size = self._desired_short_size_from_z(z)

            if short_size > 0:
                data["mode"] = "short"
                data["extreme"] = mid
                data["short_size"] = short_size
                return -short_size

            return 0

        if mode == "short":
            high = max(float(extreme or mid), mid)
            data["extreme"] = high

            old_short_size = int(data.get("short_size", self.SIZE_1))
            desired_short_size = self._desired_short_size_from_z(z)

            # Important: only scale IN, never scale OUT while short.
            # If z falls, keep the existing short until reversal.
            short_size = max(old_short_size, desired_short_size)

            # Safety fallback.
            if short_size <= 0:
                short_size = self.SIZE_1

            short_size = min(short_size, self.LIMIT)
            data["short_size"] = short_size

            if mid <= high - reversal:
                data["mode"] = "long"
                data["extreme"] = mid
                data["short_size"] = 0
                return self.LIMIT

            return -short_size

        if mode == "long":
            low = min(float(extreme or mid), mid)
            data["extreme"] = low

            if mid >= low + rebound:
                data["mode"] = "flat"
                data["extreme"] = None
                data["cooldown_until"] = timestamp + self.COOLDOWN_TIME
                data["needs_reset"] = True
                data["short_size"] = 0
                return 0

            return self.LIMIT

        data["mode"] = "flat"
        data["extreme"] = None
        data["needs_reset"] = False
        data["short_size"] = 0
        return 0

    def run(self, state: TradingState):
        data = self._load(state.traderData)

        result: Dict[str, List[Order]] = {product: [] for product in state.order_depths}

        od = state.order_depths.get(SYM)
        if od is None:
            return result, 0, self._save(data)

        bb, ba = self._bb_ba(od)
        if bb is None or ba is None:
            return result, 0, self._save(data)

        mid = (bb + ba) / 2.0
        pos = int(state.position.get(SYM, 0))
        target = self._target(mid, int(state.timestamp), data)

        buy_cap = max(0, self.LIMIT - pos)
        sell_cap = max(0, self.LIMIT + pos)

        orders: List[Order] = []

        if target > pos and buy_cap > 0:
            visible = max(0, -od.sell_orders.get(ba, 0))
            qty = min(target - pos, buy_cap, self.MAX_TRADE, visible)

            if qty > 0:
                orders.append(Order(SYM, ba, qty))

        elif target < pos and sell_cap > 0:
            visible = max(0, od.buy_orders.get(bb, 0))
            qty = min(pos - target, sell_cap, self.MAX_TRADE, visible)

            if qty > 0:
                orders.append(Order(SYM, bb, -qty))

        result[SYM] = orders
        return result, 0, self._save(data)
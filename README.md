# IMC Prosperity 4 — Allium tuberosum

Our team repository for the [IMC Prosperity 4](https://prosperity.imc.com/) algorithmic trading competition (Spring 2026).

## The Team

<!-- TODO: replace placeholder rows with real teammate info when we finalise the roster -->

| Name | Role | Links |
|---|---|---|
| Nick Zhu | Team Lead / Strategy | [LinkedIn](TBD) · [GitHub](https://github.com/Sophbx) |
| Sophia Gu | Manager/Strategy/Validation | [LinkedIn](https://www.linkedin.com/in/sophia-gu-912bbb294/) · [GitHub](https://github.com/Sophbx)|
| _Teammate 3_ | _TBD_ | _TBD_ |
| _Teammate 4_ | _TBD_ | _TBD_ |

## Repo Structure

```
Round0/
├── Data/                      # Sample market data (Day -2 and Day -1)
├── Tutorial/                  # Strategy guide PDF + LaTeX source
├── trader.py                  # Original baseline
├── trader_v2.py ... v4.py     # Early experiments (rejected)
├── trader_eme6_tomato_v1.py   # EMERALDS spread tuning intermediate
├── trader_wallmid_v1.py       # First port of Hedgehogs Wall Mid approach
├── trader_hybrid_v1.py        # Wall Mid (EMERALDS) + EMA/skew (TOMATOES)
├── trader_phase1_A/B/C.py     # Multi-agent debate Phase 1 outputs
├── trader_phase2_D/E.py       # Multi-agent debate Phase 2 refinements
├── trader_final_v1.py         # ★ Current submission
└── WRITEUP.md                 # Round 0 writeup (teammates, read this!)
```

## Round 0 — Tutorial Round

Round 0 introduces the basic Prosperity simulation via two products: **EMERALDS** (a perfectly stable asset pinned at fair value 10,000) and **TOMATOES** (a volatile, drifting asset). Both have a position limit of ±80. The round is meant to teach order-book mechanics, position limits, and basic market making — but it also hides more subtle microstructure opportunities than the tutorial lets on.

Our final algorithm is `Round0/trader_final_v1.py`. It builds on the Wall Mid concept from [Frankfurt Hedgehogs](https://github.com/timodiehm/imc-prosperity-3) (Prosperity 3 runner-up), and extends their Rainforest Resin / Kelp approach with an adaptive noise-quote tracking mechanism we developed through an internal multi-agent code review. EMERALDS is fully saturated (every catchable historical trade × 7 ticks of edge = 14,945 on the sample days). TOMATOES is where the real innovation lives: we quote exactly one tick inside whichever bot is best-priced, plant ourselves in the "virgin zone" between the noise bot and the wall, and capture most of the flow that would otherwise hit the noise bot. A layer of defensive guards (noise-bot presence check, position-limit pre-clamp, error-counter logging) protects us against the most obvious ways live trading could diverge from backtest.

Backtested PnL on the two sample days (Jmerle backtester, conservative matching):

| Version | EMERALDS | TOMATOES | Total |
|---|---:|---:|---:|
| `trader.py` (baseline) | 4,150 | 7,741 | 11,890 |
| `trader_hybrid_v1` | 14,945 | 8,170 | 23,116 |
| **`trader_final_v1`** | **14,945** | **16,726** | **31,671** |

Detailed walkthrough, design rationale, and residual risks are in [`Round0/WRITEUP.md`](Round0/WRITEUP.md).

## Backtesting

No official Prosperity 4 backtester exists yet as far as we can tell, so we are running [Jmerle's **Prosperity 3** backtester](https://github.com/jmerle/imc-prosperity-3-backtester) (last year's version) as a stop-gap. We locally patched its hardcoded `LIMITS` dict to accept EMERALDS and TOMATOES; everything else is untouched from upstream. **Treat every backtest number in this repo as a reference point, not ground truth** — the P3 simulation mechanics may differ from P4 in ways we haven't verified, and the `--match-trades worse` mode is still somewhat optimistic about queue priority. Use backtester PnL for **relative** comparisons between trader versions, not as a forecast of live scores. Full caveats in the Round 0 writeup.

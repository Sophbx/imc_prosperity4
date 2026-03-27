# IMC Prosperity 4

Our team's repository for the [IMC Prosperity 4](https://prosperity.imc.com/) algorithmic trading competition.

## Repo Structure

```
Round0/
├── Data/                   # Sample market data (prices & trades for Day -2 and Day -1)
│   ├── prices_round_0_day_-1.csv
│   ├── prices_round_0_day_-2.csv
│   ├── trades_round_0_day_-1.csv
│   └── trades_round_0_day_-2.csv
├── trader.py               # Our trading algorithm for the Tutorial Round
└── Tutorial/
    ├── tutorial.tex         # LaTeX source for the strategy guide
    └── tutorial.pdf         # Compiled PDF — read this first!
```

## Round 0 (Tutorial Round)

### `trader.py` — Trading Algorithm

This is the algorithm we submit to the Prosperity platform. It trades two products:

- **EMERALDS** (position limit: 80) — Extremely stable asset, fair value locked at 10,000. Strategy: static fair-value market making with a tight spread (9,998 / 10,002).
- **TOMATOES** (position limit: 80) — Volatile, trending asset. Strategy: dynamic fair-value tracking via EMA (alpha=0.3) + inventory-skewed market making.

Both products use a **two-phase approach** each iteration:
1. **TAKE** — Aggressively pick off any mispriced bot quotes (buy below fair value, sell above).
2. **MAKE** — Post resting limit orders inside the bot spread to capture the spread as profit.

### `Tutorial/tutorial.pdf` — Strategy Guide

A comprehensive write-up covering:
- How the Prosperity simulation works
- Core trading concepts (order books, spreads, position limits)
- Data analysis of the sample CSVs
- Market making strategy explained with math
- EMA fair value estimation
- Inventory skew / risk management
- Code walkthrough of `trader.py`
- Common pitfalls to avoid
- Tips for preparing for future rounds

**If you're new to the project, start by reading the PDF.**

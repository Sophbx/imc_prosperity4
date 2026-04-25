# Round 3 Manual Trading — Bio-Pods

The Celestial Gardeners' Guild wants to sell us Ornamental Bio-Pods. We submit
two bids; bots auto-sell our purchases at 920 the next day.

## Mechanics

- Each NPC has a private **reserve price** uniformly distributed on
  `{670, 675, 680, ..., 920}` (51 values, step 5).
- We submit two bids `b1 < b2`. Each NPC trades with us at most once.
- If `b1 > reserve` → trade at `b1`.
- Else if `b2 > reserve`:
  - if `b2 > avg_b2` (mean of *all teams'* second bids) → trade at `b2`.
  - if `b2 ≤ avg_b2` → trade at `b2`, but PnL multiplied by
    `((920 - avg_b2) / (920 - b2))^3` (penalty rises sharply as b2 sits
    further below the average).
- Profit per trade = `920 - bid` (then × penalty if applicable).
- We don't know how many NPCs there are; their count just scales total profit.

## What's in this folder

- `manual_analysis.ipynb` — Stage I-III analysis. Final picks at the bottom
  under `Submission`.

"""Generate manual_analysis.ipynb from a deterministic cell list.

Run me: `python3 build_manual_notebook.py`
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()

C = []
md = lambda s: C.append(("md", s))
py = lambda s: C.append(("py", s))

# ---------- Setup ----------
md("""# Round 3 Manual — Bio-Pods Two-Bid Analysis

Closed-form derivation of the optimal first/second bid pair for the
Celestial Gardeners' Guild manual challenge, with sensitivity over
opponent-average-second-bid assumptions.

See `README.md` for the game mechanics.""")

py("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format", lambda x: f"{x:.4f}")
sns.set_theme(context="notebook", style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 4)

LOW, HIGH, STEP = 670, 920, 5
SELL_PRICE = 920
RESERVES = np.arange(LOW, HIGH + STEP, STEP)   # 51 values: 670, 675, ..., 920
N_RESERVES = len(RESERVES)                      # 51
B1_GRID = RESERVES + 1                          # 671, 676, ..., 921 (Pareto-optimal bids)

print(f"reserves: {N_RESERVES} grid points from {RESERVES[0]} to {RESERVES[-1]}")
print(f"bid grid: {len(B1_GRID)} points from {B1_GRID[0]} to {B1_GRID[-1]}")""")

# ---------- Stage I: closed-form ----------
md(r"""## Stage I — closed-form analytical optimum

If we only submit ONE bid `b1`, expected profit per NPC is

$$ E[\text{profit per NPC}] = \frac{k_1}{51}(920 - b_1), \quad k_1 = \#\{r \in \text{grid} : r < b_1\} $$

Within each "tier" (same `k_1`), bidding higher only burns money. So optimal `b1` sits one above a grid point: `b1 ∈ {671, 676, ..., 921}`.

For `k_1` reserve points captured, the cheapest such `b1` is `666 + 5·k_1`, giving

$$ E(k_1) = \frac{k_1}{51}\,(254 - 5k_1) $$

Maximize: $\frac{dE}{dk_1} = \frac{254 - 10k_1}{51} = 0 \implies k_1 = 25.4 \implies k_1^\star = 25$, so $b_1^\star = 791$ and $E^\star = \frac{25 \cdot 129}{51} \approx 63.24$.""")

py("""rows = []
for b1 in B1_GRID:
    k1 = int((RESERVES < b1).sum())
    e = (k1 / N_RESERVES) * (SELL_PRICE - b1)
    rows.append({"b1": int(b1), "k1": k1, "E": e})
single_bid = pd.DataFrame(rows)
opt = single_bid.loc[single_bid["E"].idxmax()]
print(f"optimal single-bid b1 = {int(opt['b1'])}  k1 = {int(opt['k1'])}  E = {opt['E']:.4f}")

# sanity: matches the closed-form
assert int(opt["b1"]) == 791
assert int(opt["k1"]) == 25
assert abs(opt["E"] - 25 * 129 / N_RESERVES) < 1e-9

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(single_bid["b1"], single_bid["E"], marker="o", lw=1)
ax.axvline(opt["b1"], color="red", linestyle="--", label=f"optimum b1 = {int(opt['b1'])}")
ax.set_xlabel("b1"); ax.set_ylabel("E[profit] per NPC")
ax.set_title("Single-bid expected profit per NPC")
ax.legend(); plt.tight_layout(); plt.show()""")

md(r"""### Two-bid extension

With both `b1 < b2`:

$$ E[\text{profit/NPC}] = \frac{k_1}{51}(920 - b_1) + \frac{k_2 - k_1}{51}\cdot \pi_{b_2} $$

where

$$ \pi_{b_2} = \begin{cases} 920 - b_2 & b_2 > \overline{b_2} \\ \dfrac{(920 - \overline{b_2})^3}{(920 - b_2)^2} & b_2 \le \overline{b_2} \end{cases} $$

Stages II and III explore how the optimum responds to different assumptions about $\overline{b_2}$.""")

# ---------- Stage II: grid search at one assumed avg_b2 ----------
md(r"""## Stage II — grid search at a fixed $\overline{b_2}$

Pick a starting guess for the population's average second bid. Plausible
anchors:

- $\overline{b_2} = 791$ — if everyone plays single-bid optimum and forgets b2.
- $\overline{b_2} \approx 850$ — midpoint of the upper half of the reserve range.

Below: heatmap of $E$ over the full $(b_1, b_2)$ grid at $\overline{b_2} = 791$.
Optimum marked.""")

py("""def expected_profit(b1, b2, avg_b2):
    k1 = int((RESERVES < b1).sum())
    k2 = int((RESERVES < b2).sum())
    e_b1 = (k1 / N_RESERVES) * (SELL_PRICE - b1)
    if b2 > avg_b2:
        prof_b2 = SELL_PRICE - b2
    else:
        # b2 <= avg_b2: penalised
        prof_b2 = (SELL_PRICE - b2) * ((SELL_PRICE - avg_b2) / (SELL_PRICE - b2)) ** 3
    e_b2 = ((k2 - k1) / N_RESERVES) * prof_b2
    return e_b1 + e_b2

ASSUMED_AVG_B2 = 791

records = []
for b1 in B1_GRID:
    for b2 in B1_GRID:
        if b2 <= b1:
            continue
        records.append({
            "b1": int(b1),
            "b2": int(b2),
            "E": expected_profit(int(b1), int(b2), ASSUMED_AVG_B2),
        })
sweep = pd.DataFrame(records)
opt2 = sweep.loc[sweep["E"].idxmax()]
print(f"At avg_b2 = {ASSUMED_AVG_B2}: optimal (b1, b2) = "
      f"({int(opt2['b1'])}, {int(opt2['b2'])}), E = {opt2['E']:.4f}")

heat = sweep.pivot(index="b1", columns="b2", values="E")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(heat, ax=ax, cmap="viridis", cbar_kws={"label": "E[profit] per NPC"})
ax.invert_yaxis()
ax.set_title(f"E[profit] heatmap at avg_b2 = {ASSUMED_AVG_B2}  |  "
             f"optimum (b1={int(opt2['b1'])}, b2={int(opt2['b2'])})  E={opt2['E']:.2f}")
plt.tight_layout(); plt.show()""")

md(r"""**Read:** the optimum at $\overline{b_2}=791$ sits at the printed $(b_1, b_2)$.
A wide bright region around it shows the surface is fairly flat near the peak,
so small misjudgements about the average won't blow up the strategy.""")

# ---------- Stage III: sweep over avg_b2 ----------
md(r"""## Stage III — sensitivity to $\overline{b_2}$

For a range of plausible $\overline{b_2}$ values, find the corresponding
optimal $(b_1, b_2)$ and expected profit. Two predictions to verify:

- $b_1^\star$ is roughly stable (the penalty doesn't touch the first bid).
- $b_2^\star$ tracks $\overline{b_2}$ closely (you want to barely beat the average).""")

py("""AVG_GRID = [750, 775, 800, 825, 850, 875, 900]

records = []
for avg_b2 in AVG_GRID:
    best = None
    for b1 in B1_GRID:
        for b2 in B1_GRID:
            if b2 <= b1:
                continue
            e = expected_profit(int(b1), int(b2), avg_b2)
            if best is None or e > best["E"]:
                best = {"avg_b2": avg_b2, "b1": int(b1), "b2": int(b2), "E": e}
    records.append(best)
sensitivity = pd.DataFrame(records)
display(sensitivity)""")

py("""fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(sensitivity["avg_b2"], sensitivity["b1"], marker="o", label="b1*")
axes[0].plot(sensitivity["avg_b2"], sensitivity["b2"], marker="s", label="b2*")
axes[0].set_xlabel("assumed avg_b2"); axes[0].set_ylabel("optimal bid")
axes[0].set_title("Optimal bids vs assumed avg_b2"); axes[0].legend()

axes[1].plot(sensitivity["avg_b2"], sensitivity["E"], marker="o", color="darkgreen")
axes[1].set_xlabel("assumed avg_b2"); axes[1].set_ylabel("E[profit] per NPC")
axes[1].set_title("Expected profit at each assumed avg_b2")
plt.tight_layout(); plt.show()""")

md(r"""## Submission

Goal: pick a single $(b_1, b_2)$ that doesn't fall apart if our guess of
$\overline{b_2}$ is off. The cell below maximizes the worst-case $E$ over a
plausible range $\overline{b_2} \in [800, 850]$ — robust pick rather than
point-optimal.""")

py("""ROBUST_RANGE = list(range(800, 851, 5))   # 800, 805, ..., 850

records = []
for b1 in B1_GRID:
    for b2 in B1_GRID:
        if b2 <= b1:
            continue
        worst = min(expected_profit(int(b1), int(b2), avg_b2) for avg_b2 in ROBUST_RANGE)
        records.append({"b1": int(b1), "b2": int(b2), "worst_E": worst})
robust = pd.DataFrame(records)
pick = robust.loc[robust["worst_E"].idxmax()]
print(f"Robust pick: b1 = {int(pick['b1'])}, b2 = {int(pick['b2'])}")
print(f"Worst-case E[profit] per NPC over avg_b2 in [800, 850]: {pick['worst_E']:.4f}")
print()
print("Sanity (E at each anchor):")
for avg_b2 in [791, 800, 825, 850, 875]:
    e = expected_profit(int(pick["b1"]), int(pick["b2"]), avg_b2)
    print(f"  avg_b2 = {avg_b2}: E = {e:.4f}")""")

md(r"""### Final picks

The cell above prints the robust $(b_1, b_2)$. Submit those numbers in the
Manual Challenge UI before the round closes.

Notes:
- $b_1$ should still be near 791 — the penalty doesn't touch the first bid,
  so the single-bid optimum dominates the first-bid choice.
- $b_2$ shifts up to barely beat the assumed range $[800, 850]$, sacrificing
  some best-case profit in exchange for not getting whacked by the penalty.""")

# ---------- emit ----------
for kind, src in C:
    if kind == "md":
        nb.cells.append(nbf.v4.new_markdown_cell(src))
    else:
        nb.cells.append(nbf.v4.new_code_cell(src))

nbf.write(nb, "manual_analysis.ipynb")
print(f"wrote manual_analysis.ipynb with {len(nb.cells)} cells")

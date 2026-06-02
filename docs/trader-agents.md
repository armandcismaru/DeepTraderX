# Trader Agents

All trading algorithms live in
[`src/tbse/tbse_trader_agents.py`](../deep_trader_tbse/src/tbse/tbse_trader_agents.py).
Each is a subclass of `Trader` and competes inside TBSE. A trader is handed a
*customer order* (an assignment with a limit price and a side) and must decide
what price to quote.

## The `Trader` superclass

Shared state and lifecycle methods (mostly unchanged from BSE):

- `add_order(order)` — accept a customer assignment; returns `"LOB_Cancel"` if
  the trader already has a live quote that must be withdrawn, else `"Proceed"`.
- `del_order(coid)` — drop a completed assignment.
- `bookkeep(trade, order, ...)` — bank profit and append to the blotter. Profit
  is `limit − price` (bids) or `price − limit` (asks). **Profit is asserted
  non-negative**: a negative profit triggers `sys.exit()`.
- `respond(time, lob, trade, ...)` — react to market events (default no-op).
- `get_order(time, countdown, lob)` — produce the next quote (default `None`).

Key invariant: each trader holds **at most one** customer order at a time
(quantity fixed at 1). Much of the bookkeeping is commented as "LAZY" because it
assumes this.

## Algorithm reference

| Code | Class | Origin | One-line summary |
|------|-------|--------|------------------|
| `GVWY` | `TraderGiveaway` | BSE | Quotes its limit price exactly — gives away all surplus. |
| `ZIC` | `TraderZic` | Gode & Sunder 1993 | Quotes a uniform-random price between the limit and the system bound. |
| `SHVR` | `TraderShaver` | BSE | Shaves a penny off the current best price to take priority. |
| `SNPR` | `TraderSniper` | BSE | "Lurks" until late in the session, then shaves increasingly aggressively. |
| `ZIP` | `TraderZip` | Cliff 1997 | Adaptive margin via a Widrow-Hoff rule with momentum. |
| `AA` | `TraderAA` | Vytelingum 2006 | Adaptive-Aggressive: estimates equilibrium, adjusts aggressiveness & target. |
| `GDX` | `TraderGdx` | Tesauro & Bredin 2002 | Dynamic-programming bid/ask using "belief" functions over past quotes. |
| `DTR` | `DeepTrader` | This project | LSTM that predicts a price from 13 LOB features. |

> **Schedule note:** the seven-slot trader schedule is ordered
> `ZIC, ZIP, GDX, AA, GVWY, SHVR, DTR`. `SNPR` (Sniper) exists in the code and in
> `populate_market()` but is **not** one of the seven schedulable slots, so it is
> not used by the standard config / CLI / CSV run modes.

## DeepTrader (`DTR`) in detail

`DeepTrader.__init__` loads a trained Keras model and its per-feature
normalization vectors:

```python
self.model = nn.load_network(self.filename)              # default "DeepTrader2_2"
self.max_vals, self.min_vals = nn.normalization_values(self.filename)
self.n_features = 13
```

On each `get_order()`:

1. `create_input(lob, otype)` assembles the 13-feature vector from the published
   LOB (see [data-and-ml.md](data-and-ml.md) for the exact feature list).
2. The vector is min-max normalized with the training-time bounds, reshaped to
   `(1, 1, 13)`, and run through the model.
3. The scalar output is de-normalized back to a price.
4. The price is **clamped to never cross the trader's own limit**: an ask is
   raised to at least `limit + 1` (and nudged just inside the best ask if room
   exists); a bid is lowered symmetrically. This guarantees DTR never quotes a
   loss-making price.

The model file is selected in `tbse.py`'s `create_trader()`:
`DeepTrader("DTR", name, 0.00, 0, "DeepTrader2_2")`. Three models ship in
`src/deep_trader/Models/` (`DeepTrader2_1`, `DeepTrader2_1_old`,
`DeepTrader2_2`); only `DeepTrader2_2` is wired up by default.

## Notes on the classical algorithms

- **ZIP** (`TraderZip`) keeps separate buy/sell margins so a single instance can
  both buy and sell, and updates them in `respond()` based on whether the best
  bid/ask improved or was hit/lifted.
- **AA** (`TraderAA`) maintains an estimated equilibrium price (weighted moving
  average of recent transactions), Smith's α volatility measure, an
  aggressiveness parameter `r`, and a learned `theta`. The `respond()` path only
  updates after a detected deal.
- **GDX** (`TraderGdx`) builds a dynamic-programming value table `values[m][n]`
  on its first turn and uses `belief_buy`/`belief_sell` (empirical acceptance
  probabilities over observed quotes) to choose a price that maximizes expected
  discounted surplus.

For the precise equations and parameter choices, consult the cited papers and
the dissertation PDF; the implementations are faithful (Snashall's AA/GDX ports)
but contain several pragmatic guards against division-by-zero and degenerate
states.

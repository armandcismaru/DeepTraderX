# Data & the DeepTrader Model

This page documents the machine-learning side of the project: the LOB features,
the model architecture, and the (offline) training pipeline.

## The 13 input features

DeepTrader observes Level-2 LOB microstructure. The feature vector is built in
two mirrored places — `Exchange.lob_data_out()`
([tbse_exchange.py](../deep_trader_tbse/src/tbse/tbse_exchange.py), used to
record **training** data) and `DeepTrader.create_input()`
([tbse_trader_agents.py](../deep_trader_tbse/src/tbse/tbse_trader_agents.py),
used at **inference** time). The order of features is:

| # | Feature | Meaning |
|---|---------|---------|
| 0 | `time` | Current virtual time |
| 1 | `trade_type` | 1 if the trader's order is an Ask, 0 if a Bid |
| 2 | `limit` | The trader's own limit (cost) price |
| 3 | `mid_price` | `(best_bid + best_ask) / 2` |
| 4 | `micro_price` | Size-weighted mid: `(n_bid·ask + n_ask·bid)/(n_bid+n_ask)` |
| 5 | `imbalance` | `(n_bid − n_ask)/(n_bid + n_ask)` |
| 6 | `spread` | `abs(best_ask − best_bid)` |
| 7 | `best_bid` | Best bid price (0 if none) |
| 8 | `best_ask` | Best ask price (0 if none) |
| 9 | `delta_t` | Time since the previous trade |
| 10 | `total_orders` | `n_bids + n_asks` resting on the book |
| 11 | `smiths_alpha` | Smith's α: RMS deviation of recent trade prices from `p_estimate` |
| 12 | `p_estimate` | EWMA of recent trade prices (decay 0.9, most-recent weighted highest) |

The **target** (feature index 13 in training rows) is the price actually quoted.

> **Keep the two builders in sync.** `lob_data_out()` (training) and
> `create_input()` (inference) must produce the *same* features in the *same
> order*; the model's normalization vectors assume it. If you change one, change
> both. Note they are independent reimplementations (one uses explicit
> conditionals, the other uses `or`/comprehensions), which makes silent drift
> easy.

## Normalization

Min/max normalization bounds are fixed constants in
[`utils.py`](../deep_trader_tbse/src/deep_trader/utils.py) (`MAX_VALUES`,
`MIN_VALUES`, 14 entries = 13 features + 1 target) and are also persisted
alongside each model as `<model>.csv`. At inference, `create_input` output is
scaled with `(x − min) / (max − min)` and the model output is rescaled back with
`out · (max_target − min_target) + min_target`.

## Model architecture

Defined in
[`lstm_architecture.py`](../deep_trader_tbse/src/deep_trader/lstm_architecture.py)
(`MultivariateLSTM`):

```
Input  (batch, 1 timestep, 13 features)
  └─ LSTM(10, activation="relu", unroll=True)
      └─ Dense(5, activation="relu")
          └─ Dense(3, activation="relu")
              └─ Dense(1)                # predicted price (regression)

Optimizer: Adam(learning_rate=1.5e-5)
Loss:      MSE   (metrics: mae, msle, mse)
Epochs:    20    Batch size: 16384
```

Trained artifacts live in `src/deep_trader/Models/<name>/` as a trio:
`<name>.json` (architecture), `<name>.h5` (weights), `<name>.csv`
(normalization bounds).

## Training / data pipeline (offline)

The pipeline is **not** exercised during normal simulation runs (the trained
model ships in the repo). It consists of:

1. **Collect data** — run TBSE with `lob_out=True` so `lob_data_out()` writes a
   feature row to a per-session CSV on each trade.
2. **Pickle** — `utils.pickle_files()` (local CSVs) or `utils.pickle_s3_files()`
   (CSVs in an S3 bucket) concatenate the rows into a single pickle.
3. **Normalize** — `utils.normalize_train()` applies the fixed bounds and writes
   `normalized_data.pkl`.
4. **Batch** — `DataGenerator` (a `keras.utils.Sequence`) streams batches from
   the pickle.
5. **Fit** — `MultivariateLSTM.create_model()` trains and calls `.save()`.

> The training scripts are research-grade. Several import/initialization bugs
> were fixed (run training as `python -m src.deep_trader.lstm_architecture` from
> `deep_trader_tbse/`), but the pipeline is **not** exercised by CI and a full
> training run is unverified — notably the save path (`./Models/…`) and the
> inference load path (`./src/deep_trader/Models/…`) still differ. Inference
> (running the shipped model inside TBSE) is the supported, working path. See the
> repo-root `AGENTS.md` §6 for details.

## Real-world data inputs

`src/tbse/RWD/` holds historical price series used as the order-schedule offset
function when `config.useInputFile = True`:

- `IBM-310817.csv` (default, referenced by `config.input_file`)
- `GBP-USD-110917.csv`
- `10YR-US-Bond.xlsx`

`get_offset_event_list()` in `tbse.py` parses the CSV (columns: index, `HH:MM:SS`
time, price), normalizes prices into the system range, and drives the
equilibrium price over the virtual trading day.

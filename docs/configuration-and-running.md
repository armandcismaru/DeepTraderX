# Configuration & Running

## Prerequisites

```console
$ python3 -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt
```

All commands below assume you are **inside `deep_trader_tbse/`** — TBSE resolves
both its module imports and the model path relative to the current working
directory.

## The three run modes

`tbse.py` chooses a mode from the number of command-line arguments:

### 1. From `config.py` (no arguments)

```console
$ python3 tbse.py
```

Uses the trader counts and order schedule declared in
[`src/config.py`](../deep_trader_tbse/src/config.py). Runs `numTrials` trials.
Writes a results file named after the schedule, e.g. `00-05-00-00-00-00-05.csv`.

### 2. From the command line (7 integers)

```console
$ python3 tbse.py <ZIC> <ZIP> <GDX> <AA> <GVWY> <SHVR> <DTR>
# e.g. 5 AA + 5 DTR on each side:
$ python3 tbse.py 0 0 0 5 0 0 5
```

Each integer is how many of that trader type to place on **each** side of the
book (buyers always mirror sellers). Order: `ZIC, ZIP, GDX, AA, GVWY, SHVR, DTR`.

### 3. From a CSV file (1 argument)

```console
$ python3 tbse.py markets.csv
```

Each row is a schedule of seven comma-separated integers. TBSE runs
`numSchedulesPerRatio` schedules × `numTrialsPerSchedule` trials per row. This is
the mode used for batch experiments and in the Docker/Kubernetes deployment.

## `config.py` reference

`config.py` is plain module-level globals plus a `parse_config()` validator that
runs at startup and aborts on misconfiguration.

| Setting | Default | Meaning |
|---------|---------|---------|
| `sessionLength` | `1` | Real wall-clock seconds per session. |
| `virtualSessionLength` | `3600` | Virtual seconds the trading day represents. |
| `numZIC … numDTR` | varies | Per-side counts for the config run mode. |
| `useOffset` | `True` | Apply a time-varying offset to the equilibrium price. |
| `useInputFile` | `True` | Drive the offset from a real-world CSV (overrides `useOffset`). |
| `input_file` | `src/tbse/RWD/IBM-310817.csv` | Real-world price series. |
| `stepmode` | `fixed` | Supply/demand curve stepping: `fixed`/`jittered`/`random`. |
| `timemode` | `periodic` | Order arrival timing: `periodic`/`drip-fixed`/`drip-jitter`/`drip-poisson`. |
| `interval` | `30` | Virtual seconds between customer-order replenishment. |
| `supply` / `demand` | ranges | Random price ranges for the order schedule. |
| `symmetric` | `True` | If true, demand schedule mirrors supply. |
| `numTrials` | `1` | Trials for config / command-line modes. |
| `numSchedulesPerRatio` | `1` | Schedules per CSV row. |
| `numTrialsPerSchedule` | `50` | Trials per schedule (CSV mode). |

> **Valid `timemode` values:** `periodic` (default), `drip-fixed`,
> `drip-jitter`, `drip-poisson`. The validator and the implementation use the
> same spellings (a prior `drip-jittered`/`drip-jitter` mismatch was fixed).

## Output formats

### End-of-session statistics (config / CLI modes, and CSV mode `dump_all`)

`trade_stats()` writes one row per trial summarizing each trader type:

```
trial_id, <ttype, total_balance, n_traders, avg_profit, avg_trades, time1, time2>, ...
```

A sample is checked in at [`data/sample.csv`](data/sample.csv).

### Per-trade LOB feature dump (training data, `lob_out=True`)

`lob_data_out()` writes a feature row on each trade — the 13 features plus the
quoted price target (see [data-and-ml.md](data-and-ml.md)). This path is enabled
when collecting training data; the standard run configs pass `lob_out=False`.

## `generate_schedules.py`

A helper that writes permutations of trader proportions to `markets.csv`. It
currently appends two trailing zeros (the `SHVR`/`DTR` slots) to each generated
five-trader proportion, so generated schedules do not include Shaver or
DeepTrader unless edited.

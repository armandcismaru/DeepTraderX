# AGENTS.md — Guidance for AI agents working on DeepTraderX

This file orients coding agents (and humans) working in this repository. It
covers what the project is, how to run and verify it, the conventions to follow,
and a catalogue of **known bugs, latent defects, and security concerns** found
by scanning the codebase. Detailed end-user/architecture documentation lives in
[`docs/`](docs/README.md).

> ⚠️ **`docs/` is published publicly via GitHub Pages**
> ([`.github/workflows/pages.yml`](.github/workflows/pages.yml)). Keep security
> findings and internal notes in *this* file and `CLAUDE.md` (repo root, not
> published) — never in `docs/`.

---

## 1. What this project is

DeepTraderX (DTX, trader code `DTR`) is an LSTM-based automated trading agent
that competes inside **TBSE** — the Threaded Bristol Stock Exchange, a Python
multi-threaded simulation of a continuous-double-auction limit-order-book
market. It is academic research code (an MEng dissertation + ICAART 2024 paper),
not a production trading system. There is no live money, no broker, no network
listener; it is a self-contained Monte-Carlo market simulator.

The codebase is largely a faithful fork of Dave Cliff's BSE / Michael Rollins's
TBSE, with the `DeepTrader` agent and an offline ML training pipeline added.

## 2. How to run & verify

```console
# from repo root
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd deep_trader_tbse          # IMPORTANT: run from here
python3 tbse.py              # config-driven run
python3 tbse.py 0 0 0 5 0 0 5   # 5 AA + 5 DTR per side
python3 tbse.py markets.csv     # batch run (prints "Done Now" when finished)
```

- **Always run from `deep_trader_tbse/`.** Imports (`import src.config`) and the
  model path (`./src/deep_trader/Models/...`) are CWD-relative.
- The CSV mode prints `Done Now` on success — that string is what
  `simulation-check.yml` greps for. Don't remove it.
- Lint/format before committing: `pylint` (config `.pylintrc`, max line 120) and
  `black`. `./checks.sh` runs both over every `*.py`.

### Expect occasional flaky trials

A market session that ends with the wrong live-thread count is **discarded and
retried automatically** (`if NUM_THREADS != trader_count + 2: trial -= 1`). This
is by design, not a regression — see §5.

## 3. Conventions

- Python ≥ 3.9 (Docker uses 3.9; CI uses 3.11). Formatting is **black**; linting
  is **pylint** with a permissive `.pylintrc` and liberal inline `# pylint:
  disable=` pragmas. Match the existing style.
- The exchange/trader code deliberately preserves BSE's original structure,
  comments, and (quirky) naming. When editing ported code, prefer minimal,
  surgical changes over refactors — the retry mechanism and the GIL are load-
  bearing (§5).
- No test suite exists. "Verification" = the simulation runs to completion and
  `pylint` passes. If you add logic, consider adding a smoke test.
- Files/paths NOT in git: `.csv` (gitignored!), `*.pkl`, `.env`, `*.pem`. Note
  the `.gitignore` rule `.csv` only matches a file literally named `.csv`, **not**
  `*.csv` — so result/data CSVs in subdirs (e.g. `experiment-data/`,
  `markets.csv`, `docs/data/sample.csv`) *are* tracked. Be deliberate about
  committing generated CSVs.

## 4. Security findings

None are remotely exploitable (no network surface), but they matter for anyone
running this on shared infra or extending it:

1. **Pickle deserialization of remote data (RCE class).**
   [`utils.py`](deep_trader_tbse/src/deep_trader/utils.py) `pickle_s3_files()` /
   `normalize_train()` and
   [`data_generator.py`](deep_trader_tbse/src/deep_trader/data_generator.py)
   `pickle.load()` data pulled from an S3 bucket. Anyone who can write to that
   bucket can achieve code execution on the training host. Only unpickle data
   you produced; prefer a safe format (Parquet/NPZ) if reworking the pipeline.

2. **`os.system()` with f-string interpolation.**
   [`configure_ec2.py`](configure_ec2.py) builds `kops` commands via
   `os.system(f"...{region}...")`. The interpolated values are hard-coded today,
   so it's not currently injectable, but it's a command-injection pattern — use
   `subprocess.run([...])` with a list if this is ever revived.

3. **Hard-coded cloud identifiers.** `configure_ec2.py` embeds a specific AWS
   region, AMI ID, key-pair name, and security-group ID; `.env.example` and the
   commented code reference real bucket names (`output-data-fz19792`,
   `experiment-data-fz19792`). No secret keys are committed (good — the code uses
   the default AWS provider chain), but these identifiers are author-specific and
   should be parameterized, not reused.

4. **`except Exception as e: terminate_instances(InstanceIds=instance_ids)`** in
   `configure_ec2.py` references `instance_ids` which may be undefined if the
   failure happens before it is assigned → a `NameError` masks the real error and
   the cleanup never runs.

5. **Over-broad CI permission.** `pylint.yml` grants `pull-requests: write` but
   the job only lints. Drop unused write scopes (least privilege). It triggers on
   `push` (not `pull_request_target`), so it isn't exposed to fork PRs.

6. **Container runs as root**, `EXPOSE 80/tcp|udp` though nothing listens on a
   port. Cosmetic/hardening, but misleading.

## 5. Concurrency model — read before touching the engine

- One `Exchange` object is **shared, mutable, and unsynchronized** across all
  trader threads and the exchange thread. Trader threads call
  `exchange.publish_lob(...)` (reads `lob_anon`, `tape`, best prices) while the
  exchange thread mutates them in `process_order2`/`add_order`/`delete_best`.
  **There are no locks.** Correctness relies on the CPython GIL + the retry loop.
  Do not add long compound reads over `tape`/`lob_anon` without a lock.
- **`run_exchange` blocks on `order_q.get()` with no timeout.** After
  `start_event.clear()`, if the queue is empty the exchange thread can block
  indefinitely; `ex_thread.join()` would then hang. In practice traders keep the
  queue fed, and the thread-count check discards incomplete sessions. If you
  rework shutdown, give `get()` a timeout and re-check `start_event`.
- Trader threads **busy-poll** with `time.sleep(0.01)`. With many traders and the
  GIL, throughput is CPU-bound; "parallel" traders are concurrent, not parallel.

## 6. Known bugs & latent defects

The simulation "runs fine" today mainly because the buggy paths were disabled in
the shipped run configs (`lob_out=False`, no Shaver/Sniper in the default
`markets.csv`, training scripts not invoked). The **correctness** bugs below were
fixed on 2026-06-02; the **dead/drifting-code** items remain open.

### Fixed (2026-06-02)

1. **Shaver & Sniper sold below cost (sign bug).** The Ask branches in
   [`tbse_trader_agents.py`](deep_trader_tbse/src/tbse/tbse_trader_agents.py) used
   `min(quote_price, limit_price)` — but a seller's limit is a *floor*, so it must
   be `max(...)` (as in the Bid branch and upstream BSE). With `min`, when
   `best_ask − 1 < limit` the trader quoted below cost → loss-making trade →
   `Trader.bookkeep()`'s `profit < 0` `sys.exit()` → run aborts and retries
   forever. Both Ask branches now use `max(...)`. Verified with a `0,0,0,0,0,2,0`
   (Shaver) run.

2. **Training entry point broken.**
   [`lstm_architecture.py`](deep_trader_tbse/src/deep_trader/lstm_architecture.py)
   called `NeuralNetwork.__init__(filename, model=Sequential())` (wrong signature
   → `TypeError`) and used non-package imports. It now calls
   `NeuralNetwork.__init__(self); self.model = Sequential()` and uses relative
   imports — run it as `python -m src.deep_trader.lstm_architecture` from
   `deep_trader_tbse/`. ⚠️ Training is still **not** exercised by CI and a full
   training run was not verified; the save path (`./Models/…`) and the inference
   load path (`./src/deep_trader/Models/…`) still differ — reconcile before
   relying on it.

3. **`pickle_files()` read zero rows for valid files.**
   [`utils.py`](deep_trader_tbse/src/deep_trader/utils.py) counted lines with
   `sum(1 for _ in file)` *after* creating the `csv.reader`, exhausting the
   handle. It now reads `list(csv.reader(file))` once, then checks the length.

4. **Divide-by-zero with one trader per side.**
   [`tbse_customer_orders.py`](deep_trader_tbse/src/tbse/tbse_customer_orders.py)
   `get_order_price()` now guards `step_size = p_range/(n-1) if n > 1 else 0`
   (was an unconditional `/(n-1)`). Verified with a `0,0,0,0,0,1,0` run.

5. **`config.timemode` value mismatch.** `parse_config()` accepted
   `'drip-jittered'` while `customer_orders()` implements `'drip-jitter'`. The
   validator and its error message now use `'drip-jitter'`, matching the
   implementation and the inline config comment. Valid values are now
   consistently `periodic` / `drip-fixed` / `drip-jitter` / `drip-poisson`.

6. **`bookkeep` error path referenced `self.orders[0]`** (a key that need not
   exist) — now prints the in-scope `coid`. Only reachable in the already-fatal
   `profit < 0` branch.

7. **`DataGenerator.__getitem__` could return `None`** on an EOF before a full
   batch was assembled — it now returns the gathered `(x[:i], y[:i])`. Latent;
   training path only.

### Open — dead / drifting code

8. **S3 upload/download is commented out** throughout `tbse.py` (and
   `configure_ec2.py`), so the README's claim that results are stored in S3 is
   stale — a default run writes local CSVs only. With `lob_out=True` the LOB file
   is even written and then `os.remove`d (lines ~554-560), so that work is
   discarded.

9. **Dockerfile diverges from `requirements.txt`.** It `pip install`s
   `tensorflow`/`keras`/etc. **unpinned** and never uses `requirements.txt`
   (which pins `keras==2.15.0`, `tensorflow==2.15.0`). Builds are non-reproducible
   and may load the shipped `.h5` model under an incompatible Keras 3.x. Prefer
   `COPY requirements.txt . && pip install -r requirements.txt`.

10. **macOS arm64 dependency risk.** `requirements.txt` pairs
    `tensorflow-macos==2.16.2` with `keras==2.15.0`; TF 2.16 bundles Keras 3,
    which differs from standalone Keras 2.15 (`model_from_json` API). Works on the
    author's machine but is a fragile combination — pin carefully if you touch it.

11. **Stale line-number references** in `README.md` ("config.py lines 16-22",
    "lines 931-934", "lines 67 onwards") have drifted from the current code.

## 7. Housekeeping observations

- ~~A stray, untracked `~/Desktop/` directory at the repo root~~ — **removed
  2026-06-02** (it was an accidental literal-`~` expansion, never tracked).
- Large binaries are committed: two dissertation/paper PDFs (~6.4 MB),
  `results.ipynb` (~150 KB), model `.h5` files, and `RWD/*.xlsx`. Fine for an
  archival research repo; relevant if you care about clone size.
- The bulk of recent commits are Dependabot dependency bumps.

## 8. When making changes

- After any engine/trader change, re-run `python3 tbse.py markets.csv` and
  confirm it still prints `Done Now`.
- Shaver/Sniper (`SHVR`/`SNPR`) now quote correctly, but the default
  `markets.csv` doesn't include them — test those types explicitly
  (e.g. `0,0,0,0,0,5,0`) if you touch their logic.
- Keep `lob_data_out()` (training features) and `DeepTrader.create_input()`
  (inference features) in lockstep — same features, same order (see
  [docs/data-and-ml.md](docs/data-and-ml.md)).
- Don't put secrets or vulnerability details in `docs/` (public). 
- Commit/push only when asked; this repo's default branch is `main`.

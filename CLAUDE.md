# CLAUDE.md

Guidance for Claude Code when working in this repository. The full guidance —
conventions, the concurrency model, security findings, and the catalogue of
known bugs — lives in `AGENTS.md` and is imported below. Read it before editing.

@AGENTS.md

## Quick reference

**What it is:** DeepTraderX — an LSTM trading agent (`DTR`) competing inside
TBSE, a multi-threaded simulated limit-order-book market. Academic research code;
no live trading, no network surface. Full docs in [`docs/`](docs/README.md).

**Run it (always from `deep_trader_tbse/`):**
```console
pip install -r requirements.txt
cd deep_trader_tbse
python3 tbse.py                 # config-driven
python3 tbse.py 0 0 0 5 0 0 5   # 7 ints: ZIC ZIP GDX AA GVWY SHVR DTR per side
python3 tbse.py markets.csv     # batch; prints "Done Now" on success
```

**Verify / check:** no test suite. Success = run completes (CSV mode prints
`Done Now`) and `pylint`/`black` pass (`./checks.sh`).

## Things to remember (high-signal)

- **Run from `deep_trader_tbse/`** — imports and the model path are CWD-relative.
- **Flaky trials are normal.** Sessions ending with the wrong thread count are
  auto-discarded and retried. The shared `Exchange` is unsynchronized; the GIL +
  retry loop are load-bearing. Don't add unlocked compound reads over
  `tape`/`lob_anon`.
- **`docs/` is published to public GitHub Pages** — keep security findings and
  internal notes out of it (they belong here / in `AGENTS.md`).
- **Bug catalogue is in `AGENTS.md` §6.** The high-severity correctness bugs
  (Shaver/Sniper selling below cost, single-trader divide-by-zero, `timemode`
  mismatch, broken training entry point, `pickle_files`/`bookkeep`/`DataGenerator`
  defects) were fixed on 2026-06-02. Still open: dead/drifting deployment code
  (S3 calls commented out, Dockerfile not using `requirements.txt`, macOS arm64
  TF/Keras version risk).
- **Keep features in sync:** `Exchange.lob_data_out()` (training data) and
  `DeepTrader.create_input()` (inference) must produce the same 13 features in
  the same order.
- **Style:** black + permissive pylint (`.pylintrc`, max line 120). Preserve the
  ported BSE/TBSE structure; prefer minimal surgical diffs.
- **Commit/push only when asked.** Default branch is `main`.

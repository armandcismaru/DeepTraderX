# Architecture

DeepTraderX runs experiments inside **TBSE**, a multi-threaded simulation of a
single-instrument continuous double-auction (CDA) limit-order-book exchange.
This page describes the components, the threading model, and the end-to-end
data flow of a single *market session*.

## Components

| Module | Role |
|--------|------|
| [`tbse.py`](../deep_trader_tbse/tbse.py) | Orchestrator. Parses input, builds the trader population, spawns threads, runs sessions, writes results. |
| [`src/config.py`](../deep_trader_tbse/src/config.py) | Declarative simulation parameters + a `parse_config()` validator. |
| [`src/tbse/tbse_exchange.py`](../deep_trader_tbse/src/tbse/tbse_exchange.py) | `Exchange` / `Orderbook` / `OrderbookHalf` — the matching engine and LOB. |
| [`src/tbse/tbse_trader_agents.py`](../deep_trader_tbse/src/tbse/tbse_trader_agents.py) | The `Trader` superclass and all eight algorithms. |
| [`src/tbse/tbse_customer_orders.py`](../deep_trader_tbse/src/tbse/tbse_customer_orders.py) | Generates the supply/demand schedule (assignments handed to traders). |
| [`src/tbse/tbse_msg_classes.py`](../deep_trader_tbse/src/tbse/tbse_msg_classes.py) | The `Order` quote object passed between traders and the exchange. |
| [`src/deep_trader/`](../deep_trader_tbse/src/deep_trader/) | The DeepTrader LSTM: model definition, loading, inference, training. |

## Threading model

A market session runs **one thread per trader plus one exchange thread**, all
coordinated through `queue.Queue` objects and a single `threading.Event`
(`start_event`). The total live-thread count is therefore `n_traders + 2`
(the +2 being the exchange thread and the main thread).

```
                       ┌──────────────────────────────────────────────┐
                       │              market_session() (main)           │
                       │  • generates customer orders every ~10 ms      │
                       │  • pushes cancellations onto kill_q            │
                       └───────────────┬───────────────────────────────┘
                                       │ start_event (set / clear)
        ┌──────────────────────────────┼───────────────────────────────┐
        ▼                              ▼                                ▼
 ┌─────────────┐   order_q   ┌──────────────────┐  trader_qs[i]  ┌─────────────┐
 │ run_trader  │ ──────────► │   run_exchange   │ ─────────────► │ run_trader  │  ...
 │  (Trader 0) │ ◄────────── │  (Exchange obj)  │ ◄───────────── │  (Trader N) │
 └─────────────┘  (trade,    └──────────────────┘   (trade,      └─────────────┘
                   order,                              order,
                   lob)         shared Exchange         lob)
                                  read by ALL
```

- **`order_q`** — a single shared queue. Every trader pushes new quotes onto it;
  the exchange thread consumes them one at a time.
- **`trader_qs[i]`** — one queue per trader. After a match, the exchange
  broadcasts `[trade, order, lob]` to *every* trader's queue so they can
  bookkeep and respond.
- **`kill_q`** — the main thread pushes orders here when a new customer order
  forces an existing live quote to be cancelled; the exchange drains it.
- **`start_event`** — set once at session start, cleared at session end. Every
  thread loops `while start_event.is_set()`.

### Virtual vs. real time

Each session has a short real-world wall-clock length (`sessionLength`, default
**1 second**) mapped onto a long *virtual* trading day (`virtualSessionLength`,
default **3600 virtual seconds**). Throughout the code:

```python
virtual_time = (time.time() - start_time) * (virtual_end / sess_length)
```

All schedule timings, order timestamps, and trader logic operate in virtual
time.

### Why sessions are retried

The threading design is cooperative and **not lock-protected** (see the caveat
below), so a session can occasionally end with a thread having crashed or hung.
`market_session()` returns the live-thread count, and the orchestrator compares
it to the expected `trader_count + 2`:

```python
if NUM_THREADS != trader_count + 2:
    trial = trial - 1          # discard and re-run this trial
```

This retry loop is a deliberate robustness mechanism — treat a "flaky" trial as
expected behaviour of the simulator, not necessarily a new bug.

## End-to-end data flow of one session

1. **`populate_market()`** instantiates the trader population from the schedule
   (`("ZIC", n), ("ZIP", n), ...`) and optionally shuffles trader IDs to remove
   positional bias.
2. **Threads start**, all blocked on `start_event.wait()`; the main thread then
   calls `start_event.set()`.
3. **Customer orders**: every ~10 ms the main thread calls `customer_orders()`,
   which uses the supply/demand schedule to assign each trader a limit price
   (its *assignment*). If a trader already had a live quote, its old quote is
   queued for cancellation on `kill_q`.
4. **Quoting**: in `run_trader()`, each trader repeatedly
   - reads the current LOB via `exchange.publish_lob(...)`,
   - calls `trader.respond(...)` to update internal state,
   - calls `trader.get_order(...)` to produce a quote,
   - pushes the quote onto `order_q`.
5. **Matching**: `run_exchange()` pops each order and calls
   `Exchange.process_order2()`, which adds the order to the book and, if it
   crosses the best counterparty quote, executes a trade at the resting price.
6. **Settlement**: the trade is broadcast to all `trader_qs`. The two
   counterparties call `Trader.bookkeep()` to bank profit
   (`limit − transaction_price` for asks, `transaction_price − limit` for bids).
7. **Session end**: when wall-clock time exceeds `start_time + sess_length`, the
   main thread clears `start_event`, joins all threads, and (optionally) writes
   end-of-session statistics via `trade_stats()`.

## The order book (`OrderbookHalf` / `Orderbook` / `Exchange`)

- **`OrderbookHalf`** holds one side (bids or asks). It maintains a dict of
  orders keyed by trader ID (**max one resting order per trader per side**), a
  price-indexed `lob`, and an anonymized `lob_anon` list published to traders.
- **`Orderbook`** bundles a bid half and an ask half plus the `tape` (the
  trade/cancel history) and a monotonically increasing `quote_id`.
- **`Exchange`** subclasses `Orderbook` and adds `add_order`, `del_order`,
  `process_order2` (matching), `publish_lob`, `lob_data_out` (feature dump for
  ML), `trade_stats`, and `tape_dump`.

`publish_lob()` is the single source of market state for traders. With
`data_out=True` it additionally computes derived microstructure features
(mid-price, micro-price, order-book imbalance, spread, Smith's α, an
EWMA price estimate `p_estimate`, and inter-trade time `dt`) — these are the
features the DeepTrader consumes and the training data records.

## ⚠️ Concurrency caveat (read before editing the engine)

The single `Exchange` instance is **shared, mutable, and unsynchronized**.
Trader threads call `exchange.publish_lob(...)` (reads `lob_anon`, `tape`,
best prices) at the same time the exchange thread mutates those structures in
`process_order2`/`add_order`/`delete_best`. There are **no locks**. Correctness
today relies on CPython's GIL making individual bytecode operations atomic and
on the retry mechanism above to discard the occasional corrupted session.

If you refactor the engine, do not assume reads are consistent snapshots, and do
not introduce long compound read sequences over `tape`/`lob_anon` without a lock
— that is the most likely place to introduce intermittent, hard-to-reproduce
failures.

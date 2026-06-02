# DeepTraderX (DTX) — Documentation

> Reference documentation for the DeepTraderX project: a deep-learning trading
> agent that competes inside the **Threaded Bristol Stock Exchange (TBSE)**, a
> multi-threaded simulated limit-order-book (LOB) market.

This folder is the source for the project's documentation site (published to
GitHub Pages by [`.github/workflows/pages.yml`](../.github/workflows/pages.yml)).
Anything committed here is **public** — do not put credentials, internal notes,
or security findings in this directory.

## What is this project?

DeepTraderX (DTX, internal trader code `DTR`) is an LSTM-based automated trader
trained purely on Level-2 LOB market data. It is benchmarked against well-known
public-domain trading strategies (ZIC, ZIP, GDX, AA, Giveaway, Shaver) inside
TBSE — a Python multi-threaded fork of Dave Cliff's Bristol Stock Exchange (BSE).

The project is the codebase behind the MEng dissertation and the ICAART 2024
paper *"DeepTraderX: Challenging Conventional Trading Strategies with Deep
Learning in Multi-Threaded Market Simulations."* Both PDFs are in the repo root.

## Documentation index

| Document | Contents |
|----------|----------|
| [architecture.md](architecture.md) | System components, threading model, end-to-end data flow |
| [trader-agents.md](trader-agents.md) | Every trading algorithm (ZIC/ZIP/GDX/AA/GVWY/SHVR/SNPR/DTR) |
| [data-and-ml.md](data-and-ml.md) | The 13 LOB features, the LSTM model, the training pipeline |
| [configuration-and-running.md](configuration-and-running.md) | `config.py`, the three run modes, output formats |
| [deployment.md](deployment.md) | Docker image, Kubernetes job, AWS/EC2 provisioning |

## Repository layout (high level)

```
DeepTraderX/
├── deep_trader_tbse/            # Main application
│   ├── tbse.py                  # Entry point: orchestrates market sessions
│   ├── markets.csv              # Default batch of trader schedules
│   ├── Dockerfile               # Container image for cloud runs
│   └── src/
│       ├── config.py            # Simulation configuration + validation
│       ├── tbse/                # The exchange simulation
│       │   ├── tbse_exchange.py        # Order book + matching engine
│       │   ├── tbse_trader_agents.py   # All trading algorithms
│       │   ├── tbse_customer_orders.py # Supply/demand order generation
│       │   ├── tbse_msg_classes.py     # Order dataclass
│       │   ├── tbse_sys_consts.py      # System price bounds
│       │   └── RWD/                    # Real-world price series (IBM, GBP/USD, bonds)
│       └── deep_trader/         # The neural network
│           ├── neural_network.py       # Model load / inference / test
│           ├── lstm_architecture.py    # Model definition + training (offline)
│           ├── data_generator.py       # Keras Sequence for batched training
│           ├── utils.py                # Data pickling + normalization
│           └── Models/                 # Trained Keras models (.h5/.json/.csv)
├── configure_ec2.py             # AWS EC2 / kops cluster bootstrap (one-off)
├── generate_schedules.py        # Produces markets.csv permutations
├── market-simulations.yaml      # Kubernetes Job spec
├── requirements.txt
└── docs/                        # ← you are here
```

## Quick start

```console
$ python3 -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt
$ cd deep_trader_tbse
$ python3 tbse.py            # run with the schedule in src/config.py
```

> **Important:** TBSE must be run from inside the `deep_trader_tbse/` directory.
> Module imports (`import src.config`) and the model-load path
> (`./src/deep_trader/Models/...`) are both resolved relative to the current
> working directory.

See [configuration-and-running.md](configuration-and-running.md) for the other
two ways to specify a market (command-line and CSV) and the output formats.

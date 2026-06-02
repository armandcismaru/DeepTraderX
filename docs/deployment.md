# Deployment

The simulation is embarrassingly parallel: each pod/container runs the same
image against `markets.csv` to accumulate many independent market sessions.
"Parallelism" at scale comes from running many identical instances, not from a
single distributed job.

## Docker

The image is built from
[`deep_trader_tbse/Dockerfile`](../deep_trader_tbse/Dockerfile):

```console
$ docker pull armandcismaru/deeptrader:deeptrader2.5
$ docker run armandcismaru/deeptrader:deeptrader2.5
```

The container entry point is `python3 tbse.py markets.csv`.

> **Build-context caveat:** the `CMD` runs `tbse.py` from the image working
> directory `/app`, and `COPY . /app` copies the build context's root. Because
> `tbse.py` lives **inside** `deep_trader_tbse/`, the image must be built with
> that directory as the build context (`docker build deep_trader_tbse/`), not the
> repository root. See the repo-root `AGENTS.md` for the full list of
> Docker/dependency caveats (the Dockerfile installs dependencies inline rather
> than from `requirements.txt`, so versions can drift).

## Kubernetes

[`market-simulations.yaml`](../market-simulations.yaml) defines a `batch/v1`
Job:

```console
$ kubectl apply -f market-simulations.yaml
$ kubectl get pods
```

It requests `completions: 9` / `parallelism: 9` — i.e. nine pods running the
same image. Adjust `completions`/`parallelism` to scale; update the `image:` tag
when you push a new version.

## AWS provisioning (`configure_ec2.py`)

[`configure_ec2.py`](../configure_ec2.py) is a one-off bootstrap script that
launches an EC2 instance (installing Docker + Kubernetes via user-data) and,
optionally, creates a `kops` cluster. It is **environment-specific** — it
contains hard-coded region, AMI, key-pair, and security-group identifiers tied
to the original author's AWS account, and much of it is commented out. Treat it
as a historical reference, not a turnkey tool, and review it before running.

## Credentials & buckets

The project uses the default AWS SDK credential provider chain (environment
variables, shared profile, or instance role) — **no credentials are hard-coded**
in the source. Copy `.env.example` to `.env` and set values as needed:

- `AWS_S3_INPUT_BUCKET` — where training CSVs are read from for pickling
  (`utils.pickle_s3_files`).
- `AWS_S3_OUTPUT_BUCKET` — where TBSE experiment outputs are uploaded (`tbse.py`).

Note: in the current source the S3 upload/download calls in `tbse.py` are
commented out, so a default run writes results to local CSV files only.

## Continuous integration

Three GitHub Actions workflows are defined in `.github/workflows/`:

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `pylint.yml` | push, weekly cron, manual | Lints every `*.py` file with pylint. |
| `simulation-check.yml` | PRs to `main`, weekly cron, manual | Runs `tbse.py markets.csv` and asserts it prints `Done Now`. |
| `pages.yml` | push to `main` touching `docs/**` | Builds `docs/` into a static site with Jekyll and deploys it to GitHub Pages. |

Dependabot (`.github/dependabot.yml`) opens weekly pip dependency-update PRs —
which is the bulk of the recent commit history.

### Enabling the documentation site (one-time)

`pages.yml` deploys via the GitHub Actions Pages flow, which requires the repo's
Pages source to be set accordingly: **Settings → Pages → Build and deployment →
Source = "GitHub Actions"**. Until that is set once, the workflow builds but
cannot publish. After enabling, every push to `main` that touches `docs/**`
rebuilds and redeploys the site (served at `https://<user>.github.io/DeepTraderX/`).

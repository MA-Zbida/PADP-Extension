# PADP: agreement-gated cooperative transport

A continuous 2D warehouse where objects have **weights** and agents have **capacities**. An object
too heavy for one agent must be lifted by a coalition of 2 to 4 agents, and carried as a rigid
group that only moves when every member chooses the same move. Agents are trained with
multi-agent PPO (MAPPO or IPPO).

**Question.** Do learned teams fail at this mainly because each agent samples its actions with its
own random noise, and does *sharing* that noise inside each coalition (derived from the object they
hold, with no communication) fix it? The `coupled` sampler in
[padp/marl/sampler.py](padp/marl/sampler.py) implements the shared noise.

## Install

```bash
pip install -r requirements.txt
pip install -e .
pytest            # all tests, about 20 seconds
```

## Quickstart

```bash
python scripts/visualize.py --scenario k3                          # watch three agents lift 140 kg
python scripts/train.py --config configs/experiments/smoke.yaml   # 3-minute training check
python -m tensorboard.main --logdir runs                            # live curves at http://localhost:6006
```

## Experiments

Each experiment is a YAML file in [configs/experiments/](configs/experiments/). Run it with
`python scripts/sweep.py configs/experiments/<name>.yaml --parallel 6`, then summarize it with
`python scripts/analyze.py <name>` (writes `results/<name>/summary.md` and figures).

| Config | What it varies | Runs |
|---|---|---|
| `smoke.yaml` | Pipeline check | 1 |
| `e1_mechanism.yaml` | Coalition size (k = 1, 2, 3) x carry length x sampler | 100 |
| `e1_sparse.yaml` | The same without reward shaping | 30 |
| `e1_k3_entropy.yaml` | k = 3 at a higher entropy coefficient | 40 |
| `e2_entropy.yaml` | Entropy coefficient x sampler (k = 2) | 30 |
| `e3_physics.yaml` | Carrying rule (strict, average, leader) x sampler | 30 |
| `e4_main.yaml` | Full warehouse with racks: five methods | 25 |
| `e4_pilot.yaml` | Difficulty ladder toward the full warehouse | 12 |
| `e4_coalition.yaml` | Open floor, heterogeneous coalitions: five methods | 50 |
| `e4_anchor_ablation.yaml` | Variants of the coupled sampler (global noise, movement only, wrong anchors) | 40 |
| `e5_lbf_pilot.yaml` | Level-Based Foraging (cooperative and standard tasks) x sampler, short pilot | 16 |
| `e5_lbf.yaml` | Level-Based Foraging (cooperative and standard tasks) x sampler | 80 |

The toy game in `scripts/toy_agreement_game.py` checks the sampler on a matrix-style agreement task
in a few minutes.

Sweeps skip finished runs and resume interrupted ones, so you can stop with Ctrl+C and rerun the
same command later. On a laptop CPU, a 2M-step run takes about 15-20 min and a 3M-step run about
20-25 min; on a 4-core / 8-thread CPU, `--parallel 6` gives the best throughput. Training is
deterministic given the seed, and `tests/test_reference_trajectories.py` pins the simulator's
trajectories bit for bit.

## Repository layout

```
padp/
  env/        warehouse.py (rules), lbf.py (Level-Based Foraging adapter), factory.py, config.py, rewards.py, ...
  marl/       trainer.py (PPO loop), sampler.py (independent / coupled noise), networks.py, buffer.py, logger.py
  utils/      config_io.py (YAML, grids, --set), stats.py (IQM, bootstrap, tests)
configs/      base.yaml and experiments/*.yaml
scripts/      train, sweep, analyze, evaluate, visualize, toy_agreement_game
tests/        rules, reward, seeding, reference trajectories, sampler, training loop, tooling
docs/         environment specification, Level-Based Foraging setup
runs/         training outputs (git-ignored)
```

## Configuration

Defaults live in [configs/base.yaml](configs/base.yaml). An experiment file overrides some of them
and declares a grid of factors; every combination is a variant (for example `coupled-k2-long`).

```bash
python scripts/sweep.py configs/experiments/e1_mechanism.yaml --dry-run                 # list the runs
python scripts/train.py --config configs/experiments/e1_mechanism.yaml \
    --variant coupled-k2-long --seed 3                                                    # one run
python scripts/train.py --config configs/base.yaml --name my_test \
    --set train.sampler=coupled env.n_agents=3 env.weights=[90]                          # ad-hoc run
```

## Outputs

Each run writes `runs/<experiment>/<variant>/seed<k>/`:

- `config.yaml`: the exact settings used
- `metrics.csv`: training curves (success, carry agreement, losses, ...), every 20k steps
- `eval.csv`: success on 20 fixed evaluation episodes, every 200k steps
- `tb/`: TensorBoard logs
- `latest.pt`, `final.pt`: checkpoints

To inspect a trained run: `python scripts/evaluate.py runs/<experiment>/<variant>/seed0 --render`.
The environment is specified in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md); the Level-Based Foraging
setup (observation, reward, anchors, metrics) in [docs/LBF.md](docs/LBF.md).

## License

MIT, see [LICENSE](LICENSE).

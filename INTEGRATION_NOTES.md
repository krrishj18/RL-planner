# Integration notes — `rlplanner.llm` (query hints) and `EnvConfig.queries_dynamic`

Written by the LLM-hints builder. `scripts/train.py`, `scripts/imitate.py`, `scripts/warmstart.sh`,
`src/rlplanner/train/scenes.py`, `src/rlplanner/sim/baselines.py` and `scripts/run_baselines.py`
are owned by another builder in this pass, so nothing there was edited. These are the one-line
hooks they need — none of them is required for the new code to work, and none of them changes
behaviour while the feature stays off.

## What already works with no change

- `EnvConfig` gained one block, `queries_dynamic` (`sim/config.py: QueryScheduleConfig`), default
  `enabled = False`. With it off `DisasterEnv` never touches the mission list, so every existing
  run, checkpoint and yaml reproduces bit-for-bit. `to_dict` / `from_dict` / `to_yaml` round-trip
  it (`pool` is a tuple in the dataclass and a list on disk, like `rayfronts.queries`).
- Variant yamls need no code change: `EnvConfig.from_dict` already routes a `queries_dynamic:`
  block, so a variant can switch it on by adding

  ```yaml
  queries_dynamic:
    enabled: true
    every: 10          # decisions between edit draws
    p_edit: 0.5
    n_init_min: 1
    n_init_max: 3
    w_min: 0.3
    w_max: 1.0
    noise_std: 0.0     # per-dim noise on a drawn query embedding
  ```

- `train/scenes.py` needs nothing: `SceneBank.make_env` hands the `EnvConfig` straight to
  `DisasterEnv`, which builds the sampler itself in `reset()` from its own seed. Verified through
  `VecEnv` and the `par_env` workers — the query block keeps its `Qmax` width, so nothing in the
  stacking, the policy or a checkpoint moves.
- `sim/baselines.py` and `scripts/run_baselines.py` need nothing: `Policy.query_scores(obs, r)`
  reads the live query block, so a baseline follows an edited mission list automatically.

## Hooks worth adding

### `scripts/train.py`

One flag mapping to the config block, plus the usual `args.json` record:

```python
ap.add_argument("--dynamic-queries", action="store_true",
                help="sample and edit the mission queries per episode (EnvConfig.queries_dynamic)")
ap.add_argument("--dq-every", type=int, default=10)      # optional
ap.add_argument("--dq-noise", type=float, default=0.0)   # optional
...
cfg.queries_dynamic.enabled = bool(a.dynamic_queries)
cfg.queries_dynamic.every = int(a.dq_every)
cfg.queries_dynamic.noise_std = float(a.dq_noise)
```

The variant yaml's `queries_dynamic` block should win when the flag is absent, exactly as the
comms block does (`--dynamic-queries` sets it, otherwise leave whatever the yaml said).

### `scripts/imitate.py`

Same two lines if the teacher rollouts should also meet a moving query list. The teachers read the
query block through `query_scores`, so they follow the edits without any other change; leaving it
off is also a defensible choice (the imitation target stays a fixed-query expert).

### `scripts/warmstart.sh`

Pass `--dynamic-queries` (and, if added, `--dq-every` / `--dq-noise`) through to `train.py` and
`imitate.py` in the same place the comms flags are forwarded.

### `scripts/sweep.py` (not owned by either of us, listed for completeness)

If the summary header prints the variant's execution switches, `queries_dynamic.enabled` belongs
next to `sequential_decode` and `comms.mode`.

## Evaluation

`scripts/llm_hints_eval.py` is self-contained: it runs held-out episodes serially in-process (the
hint loop needs the live env between decisions) and reuses `train/evaluate.py` for the table, the
CSV and the per-episode rows. It does not touch `eval_policy.py`. If a `--dynamic-queries`
equivalent is ever wanted in `eval_policy.py`, it is the same one line on `cfg`.

## Contract note

`sim/config.py` and `sim/env.py` are contract files (CONTRACTS.md preamble). The changes are
additive and backward compatible:

- `config.py`: new `QueryScheduleConfig` dataclass and the `EnvConfig.queries_dynamic` field,
  validated only when enabled.
- `env.py`: `DisasterEnv.set_queries` was split into a private `_set_queries` (swap the list, no
  observation rebuild) plus the public rebuild; `reset` draws the initial subset before the first
  observation is built and `step` applies an edit at the decision boundary, both no-ops while the
  block is disabled. The public `set_queries(names, weights) -> TeamObs` signature and semantics
  are unchanged. With the block enabled, `reset` first restores `cfg.rayfronts.queries` to the
  mission list the env was built with: the schedule writes its draw onto the config, and the next
  episode would otherwise build its embedding table from a noised name no fresh table knows.

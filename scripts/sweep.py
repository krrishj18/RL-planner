#!/usr/bin/env python
"""Run the DESIGN_VARIANTS.md E sweep: one training run per variant, then one held-out
evaluation of each at several comms ranges *and* at its own training comms setting, then a
summary table and a combined curve figure.

    uv run python scripts/sweep.py --variants central_full decentral_share_all \
        --scenes synthetic:0-200 --updates 400 --envs 32 --workers 10

Every variant gets the same seed, the same scene bank and therefore the same held-out split, so
the only difference between two rows of the summary is the variant's own config. Runs land in
`runs/sweep_<stamp>/<variant>/` and the summary in `runs/sweep_<stamp>/summary.md`.

The common ranges keep the rows comparable, but they hand a variant a radio it never trained with
(the blackout's row at 200 m is a different system). The `own` section evaluates each variant at
the comms setting it trained on — blackout at range 0, `central_full` on one shared belief, the
share_* variants with their randomised range — always with each variant's own payload flags.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import yaml

VARIANT_DIR = Path("configs/variants")
ALL_VARIANTS = ("central_full", "decentral_share_all", "decentral_share_pos_cov",
                "decentral_share_rays", "decentral_blackout", "decentral_share_all_noreward",
                "decentral_share_all_tokens_only")
# (column, header, format) of the summary table
COLS = (("frac_found", "found", "{:.3f}"), ("finds_auc", "AUC", "{:.3f}"),
        ("time_to_first", "t_first", "{:.0f}"), ("time_to_half", "t_half", "{:.0f}"),
        ("redundancy_frac", "redundant", "{:.2f}"),
        ("intentional_revisits", "revisits", "{:.1f}"),
        ("coverage_end", "coverage", "{:.3f}"), ("dist_per_find", "dist/find", "{:.0f}"),
        ("link_frac", "link", "{:.2f}"), ("reward", "reward", "{:+.2f}"))
CONTAINERS = ("open", "vehicle", "building", "rubble")
OWN = "own"                      # eval key: the variant's own training comms setting
CURVE_PANELS = (("reward", "episode reward"), ("frac_found", "fraction found"),
                ("finds_auc", "finds AUC"), ("coverage_end", "coverage"))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list(ALL_VARIANTS))
    ap.add_argument("--scenes", nargs="+", default=["synthetic:0-200"])
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--robots", default="3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-episodes", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-seed", type=int, default=10_000)
    ap.add_argument("--eval-ranges", nargs="+", default=["100", "200", "inf"],
                    help="link ranges (metres, 'inf' allowed) the final policies are scored at")
    ap.add_argument("--own-eval", action=argparse.BooleanOptionalAction, default=True,
                    help="also score every variant at its *own* training comms setting (blackout "
                         "at range 0, central_full on one shared belief): the common ranges give "
                         "a variant a radio it never had")
    ap.add_argument("--out", default=None, help="run directory (default runs/sweep_<stamp>)")
    ap.add_argument("--skip-train", action="store_true",
                    help="only (re)build the summary from an existing --out directory")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--train-arg", action="append", default=[],
                    help="extra argument forwarded to train.py, repeatable (e.g. --train-arg=--lr "
                         "--train-arg=1e-4)")
    return ap


# ---- training ----------------------------------------------------------------------------------
def train_one(a, run: Path, variant: str) -> int:
    try:
        name = str(run.relative_to("runs"))       # train.py writes to runs/<name>
    except ValueError:
        name = str(run.resolve())                 # ... an absolute --out lands where it says
    cmd = [sys.executable, "scripts/train.py", "--variant", variant,
           "--name", name, "--scenes", *a.scenes,
           "--updates", str(a.updates), "--envs", str(a.envs), "--rollout", str(a.rollout),
           "--robots", str(a.robots), "--seed", str(a.seed), "--device", a.device,
           "--eval-every", str(a.eval_every), "--eval-episodes", str(a.eval_episodes)]
    if a.workers:
        cmd += ["--workers", str(a.workers)]
    if a.no_baselines:
        cmd += ["--no-baselines"]
    cmd += list(a.train_arg)
    print(f"[sweep] {variant}: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    rc = subprocess.run(cmd).returncode
    print(f"[sweep] {variant}: {'ok' if rc == 0 else f'FAILED rc={rc}'} "
          f"in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return rc


# ---- evaluation --------------------------------------------------------------------------------
def variant_cfg(variant: str, comms_range: float | None, common_reward: bool = True):
    from rlplanner.sim.config import EnvConfig, RewardConfig
    path = Path(variant) if Path(variant).exists() else VARIANT_DIR / f"{variant}.yaml"
    d = dict(yaml.safe_load(path.read_text()) or {})
    d.pop("train", None)
    d.pop("notes", None)
    cfg = EnvConfig.from_dict(d)
    if common_reward:
        # one yardstick for the `reward` column: a variant that trained without the redundancy and
        # revisit terms is still *scored* with them, or `..._noreward` tops the table for not being
        # charged. The ablation is what the policy learned from, not what it is judged by.
        cfg.reward = RewardConfig()
    if comms_range is not None:
        # every variant is scored under decentralised execution at a fixed range, `central_full`
        # included: that row is the out-of-distribution transfer of a centrally trained policy
        cfg.comms.mode = "range"
        cfg.comms.range_m = float(comms_range)
        cfg.comms.randomize_range = False
    return cfg


def eval_one(ck: Path, variant: str, a, comms_range: float | None) -> dict:
    """`comms_range=None` keeps the variant's own comms block (the `own` row)."""
    from rlplanner.train.evaluate import TorchActor, evaluate_policy, load_checkpoint
    from rlplanner.train.scenes import SceneBank, parse_robots
    cfg = variant_cfg(variant, comms_range)
    policy, _ = load_checkpoint(ck, "cpu")
    if not policy.use_local:
        cfg.tokens.local_size = 0
    if not policy.use_robot_bev:
        cfg.tokens.robot_bev_size = 0
    bank = SceneBank(" ".join(a.scenes))
    lo, _ = parse_robots(a.robots)
    cfg.robot.n_robots = bank.robot_bounds((lo, lo), "heldout")[0]
    res, rows = evaluate_policy(TorchActor(policy, "cpu", True, variant), bank, cfg,
                                a.eval_episodes, lo, seed=a.eval_seed, split="heldout",
                                workers=a.workers, return_rows=True)
    return {"cols": {k: _mean_ci(v) for k, v in res.items()},
            "by_container": _containers(rows), "n": len(rows),
            "setting": comms_setting(cfg),
            "sequential_decode": bool(policy.sequential_decode)}


def comms_setting(cfg) -> str:
    """One-line description of the comms an evaluation ran under."""
    c = cfg.comms
    if c.mode == "full":
        return "full (one shared belief)"
    if c.randomize_range and c.range_choices:
        return "range " + "/".join(_rng(x) for x in c.range_choices) + " (random per episode)"
    if c.range_m <= 0:
        return "range 0 (blackout: spawn exchange only)"
    return f"range {_rng(c.range_m)}"


def decode_flag(run: Path, variant: str) -> bool | None:
    """`policy.sequential_decode` the run actually trained with (args.json), else the variant's."""
    import json as _json
    ap = run / "args.json"
    if ap.exists():
        d = _json.loads(ap.read_text())
        v = (d.get("policy") or {}).get("sequential_decode", d.get("sequential_decode"))
        if v is not None:
            return bool(v)
    path = Path(variant) if Path(variant).exists() else VARIANT_DIR / f"{variant}.yaml"
    if not path.exists():
        return None
    train = dict((yaml.safe_load(path.read_text()) or {}).get("train") or {})
    v = train.get("sequential_decode", train.get("sequential-decode"))
    return None if v is None else bool(v)


def _mean_ci(v) -> tuple[float, float]:
    v = np.asarray(v, np.float64)
    n = int(np.isfinite(v).sum())
    if n == 0:
        return (float("nan"), float("nan"))
    sd = float(np.nanstd(v, ddof=1)) if n > 1 else 0.0
    return (float(np.nanmean(v)), 1.96 * sd / max(1.0, np.sqrt(n)) if n > 1 else 0.0)


def _containers(rows) -> dict[str, tuple[int, int]]:
    """{container: (found, total)} summed over the evaluation episodes."""
    out = {c: [0, 0] for c in CONTAINERS}
    for r in rows:
        for c in CONTAINERS:
            out[c][0] += int((r.get("found_by_container") or {}).get(c, 0))
            out[c][1] += int((r.get("n_by_container") or {}).get(c, 0))
    return {c: (v[0], v[1]) for c, v in out.items()}


# ---- reporting ---------------------------------------------------------------------------------
def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def summary_md(out: Path, a, results: dict, ranges) -> str:
    decode = {v: decode_flag(out / v, v) for v in a.variants}
    lines = [f"# Variant sweep — {out.name}", "",
             f"- scenes `{' '.join(a.scenes)}`, robots `{a.robots}`, seed {a.seed}, "
             f"{a.updates} updates x {a.envs} envs x {a.rollout} decisions",
             f"- held-out evaluation: {a.eval_episodes} episodes per variant per range, "
             f"seed {a.eval_seed}, deterministic actions",
             "- every variant is evaluated under **range comms** at each of the common ranges "
             "below, so `central_full` is scored out of distribution (it trained on one shared "
             "map) and the blackout is handed a radio it never had — the **own** section scores "
             "each variant at the comms setting it trained on instead.",
             "- each variant is always evaluated with **its own gossip payload flags**, at every "
             "range: `decentral_share_rays` never receives coverage, `..._pos_cov` never receives "
             "rays or visited records, the blackout receives nothing after spawn.",
             "- every variant is scored with the **default reward**, `..._noreward` included: the "
             "ablation is what the policy trained on, not the yardstick it is judged by.",
             "- `policy.sequential_decode` (the same-decision claim mask, a centralised-execution "
             "device): " + ", ".join(f"`{v}`={_flag(decode[v])}" for v in a.variants) + ".",
             "- cells are mean +- 95% CI half-width over the evaluation episodes.", ""]
    for rng in ranges:
        own = rng == OWN
        lines += [("## eval at each variant's own training comms setting" if own
                   else f"## eval comms range = {_rng(rng)}"), ""]
        extra = ["setting"] if own else []
        head = "| variant | " + " | ".join(extra + [h for _, h, _ in COLS]) + " |"
        lines += [head, "|" + "---|" * (len(COLS) + len(extra) + 1)]
        for v in a.variants:
            res = results.get((v, rng))
            pre = [res.get("setting", "-") if res else "-"] if own else []
            if not res:
                lines.append(f"| {v} | " + " | ".join(pre + ["-" for _ in COLS]) + " |")
                continue
            cells = []
            for key, _, fmt in COLS:
                m, ci = res["cols"].get(key, (float("nan"), float("nan")))
                cells.append(f"{fmt.format(m)} ± {fmt.format(ci).lstrip('+')}"
                             if np.isfinite(m) else "-")
            lines.append(f"| {v} | " + " | ".join(pre + cells) + " |")
        lines += ["", "### finds by container (found / total, summed over the episodes)", "",
                  "| variant | " + " | ".join(CONTAINERS) + " |",
                  "|" + "---|" * (len(CONTAINERS) + 1)]
        for v in a.variants:
            res = results.get((v, rng))
            if not res:
                lines.append(f"| {v} | " + " | ".join("-" for _ in CONTAINERS) + " |")
                continue
            bc = res["by_container"]
            lines.append(f"| {v} | " + " | ".join(f"{bc[c][0]}/{bc[c][1]}" for c in CONTAINERS)
                         + " |")
        lines.append("")
    lines += ["## training curves", "", "![curves](curves.png)", "",
              "## columns", "",
              "- `found` = fraction of casualties found, `AUC` = finds-AUC over the horizon",
              "- `t_first` / `t_half` = seconds to the first / to half the casualties (t_max if "
              "never)",
              "- `redundant` = fraction of the cells a robot covered per decision that a peer "
              "had already covered (the redundancy term charges `redundancy_cost` x this)",
              "- `revisits` = intentional revisits per episode, summed over robots (gross: a "
              "revisit that turns up a casualty is refunded and costs nothing)",
              "- `reward` = undiscounted episode return under the default reward config",
              "- `link` = mean fraction of robot pairs in contact per decision", ""]
    return "\n".join(lines)


def _flag(v) -> str:
    return "?" if v is None else str(bool(v))


def _rng(r) -> str:
    if r == OWN:
        return OWN
    return "inf" if not np.isfinite(r) else f"{r:.0f} m"


def curves(out: Path, variants) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (key, title) in zip(axes.ravel(), CURVE_PANELS):
        for v in variants:
            rows = read_rows(out / v / "eval.csv")
            xy = [(r["decisions"], r[key]) for r in rows
                  if key in r and np.isfinite(r.get(key, np.nan))]
            if xy:
                ax.plot(*zip(*xy), "o-", ms=3, lw=1.4, label=v)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("decisions")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle(f"variant sweep — {out.name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    for v in a.variants:
        if not (Path(v).exists() or (VARIANT_DIR / f"{v}.yaml").exists()):
            raise SystemExit(f"[sweep] unknown variant {v!r}; known: {sorted(ALL_VARIANTS)}")
    out = Path(a.out) if a.out else Path("runs") / f"sweep_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep_args.json").write_text(json.dumps(vars(a), indent=2, default=str))
    ranges = [float(r) for r in a.eval_ranges] + ([OWN] if a.own_eval else [])

    failed: list[str] = []
    if not a.skip_train:
        for v in a.variants:
            if train_one(a, out / v, v) != 0:
                failed.append(v)

    results: dict[tuple[str, float | str], dict] = {}
    for v in a.variants:
        ck = out / v / "latest.pt"
        if not ck.exists():
            print(f"[sweep] {v}: no checkpoint at {ck}, skipping evaluation")
            continue
        for rng in ranges:
            t0 = time.perf_counter()
            res = results[(v, rng)] = eval_one(ck, v, a, None if rng == OWN else rng)
            print(f"[sweep] eval {v} @ {_rng(rng)} [{res.get('setting', '-')}]: "
                  f"{time.perf_counter() - t0:.1f}s "
                  f"found={res['cols']['frac_found'][0]:.3f}", flush=True)
    curves(out, a.variants)
    (out / "summary.md").write_text(summary_md(out, a, results, ranges))
    with (out / "summary.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["variant", "eval", "comms_range_m", "eval_setting", "sequential_decode",
                     "n_episodes"]
                    + [c for k, _, _ in COLS for c in (k, f"{k}_ci")]
                    + [f"found_{c}" for c in CONTAINERS] + [f"n_{c}" for c in CONTAINERS])
        for (v, rng), res in results.items():
            row = [v, _rng(rng), "" if rng == OWN else rng, res.get("setting", ""),
                   res.get("sequential_decode", ""), res["n"]]
            for key, _, _ in COLS:
                m, ci = res["cols"].get(key, (float("nan"), float("nan")))
                row += [m, ci]
            row += [res["by_container"][c][0] for c in CONTAINERS]
            row += [res["by_container"][c][1] for c in CONTAINERS]
            wr.writerow(row)
    print(f"[sweep] wrote {out}/summary.md, summary.csv and curves.png")
    if failed:
        print(f"[sweep] training FAILED for: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

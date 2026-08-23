#!/usr/bin/env python3
"""Export Scene JSON (schema v0.1) for preset x seed, without Isaac or Nucleus.

    uv run python scripts/export_scenes.py --preset earthquake --seeds 0:20 --out data/scenes
    uv run python scripts/export_scenes.py --preset tornado explosion --seeds 0,3,7 --region 800 600
    uv run python scripts/export_scenes.py --pipeline v2 --locale downtown \\
        --disaster earthquake tornado explosion --severity-range 0.5 1.0 \\
        --seeds 0:60 --region-range 500 1500 --size-jitter 0.25 \\
        --casualties auto --bystanders auto --out data/scenes_v2
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from rlplanner.scene.casualties import AUTO, HumanConfig
from rlplanner.scene.export import (V2_LOCALE_PRESET, available_presets,
                                    export_scene, format_summary_table,
                                    sample_region, sample_severity,
                                    scene_summary)


def parse_seeds(text: str) -> list[int]:
    """`0:20` (half-open range), `0,3,7` (explicit), `5` (single)."""
    text = text.strip()

    def num(tok: str, what: str) -> int:
        try:
            return int(tok)
        except ValueError:
            raise ValueError(f"--seeds {text!r}: {what} {tok.strip()!r} is not an "
                             "integer (use 0:20, 0,3,7 or 5)") from None

    if ":" in text:
        lo, _, hi = text.partition(":")
        a, b = num(lo, "range start"), num(hi, "range end")
        if b <= a:
            raise ValueError(f"--seeds {text!r}: empty range, end must exceed start")
        seeds = list(range(a, b))
    else:
        seeds = [num(s, "seed") for s in text.split(",") if s.strip()]
    if not seeds:
        raise ValueError(f"--seeds {text!r}: no seeds given")
    if any(s < 0 for s in seeds):
        raise ValueError(f"--seeds {text!r}: seeds must be non-negative")
    return seeds


def parse_count(text: str) -> int | str:
    """A non-negative integer, or "auto" (scale with the region area)."""
    if str(text).strip().lower() == AUTO:
        return AUTO
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not an integer or {AUTO!r}") from None
    if v < 0:
        raise argparse.ArgumentTypeError(f"{text!r} must be non-negative")
    return v


def _out_dir(path: str) -> Path:
    d = Path(path)
    if d.exists() and not d.is_dir():
        raise ValueError(f"--out {path!r} exists and is not a directory")
    d.mkdir(parents=True, exist_ok=True)
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", nargs="+", default=["earthquake"],
                    help=f"preset name(s) or config path(s); known: {', '.join(available_presets())}")
    ap.add_argument("--pipeline", default="v1", choices=("v1", "v2"),
                    help="v1 = build_city only; v2 = the detailed city "
                         "(anisotropic blocks, districts, parks)")
    ap.add_argument("--locale", default="downtown",
                    choices=sorted(V2_LOCALE_PRESET),
                    help="v2: which pristine locale preset to compose onto")
    ap.add_argument("--disaster", nargs="+", default=None,
                    help="disaster type(s); required for --pipeline v2, and an "
                         "override of the preset's own type for v1")
    ap.add_argument("--seeds", default="0:20", help="0:20 | 0,3,7 | 5")
    ap.add_argument("--out", default="data/scenes", help="output directory")
    ap.add_argument("--region", nargs=2, type=float, metavar=("W", "H"),
                    help="override the city extent in metres")
    ap.add_argument("--region-range", nargs=2, type=float, metavar=("LO", "HI"),
                    help="sample W and H independently and uniformly from "
                         "[LO, HI] m (per seed, deterministic, multiples of 10)")
    ap.add_argument("--cell-m", type=float, default=2.0, help="damage-grid spacing")
    ap.add_argument("--severity", type=float, default=None, help="override preset severity")
    ap.add_argument("--severity-range", nargs=2, type=float, metavar=("LO", "HI"),
                    help="sample severity per (seed, disaster) from [LO, HI]")
    ap.add_argument("--casualties", type=parse_count,
                    default=HumanConfig.n_casualties,
                    help="count, or 'auto' for 15 per 400x400 m of region, "
                         "clipped to [10, 80]")
    ap.add_argument("--bystanders", type=parse_count,
                    default=HumanConfig.n_bystanders,
                    help="count, or 'auto' for half the casualty count")
    ap.add_argument("--spawn-corner", default="ll", choices=("ll", "lr", "ul", "ur"))
    ap.add_argument("--size-jitter", type=float, default=0.0,
                    help="spread the per-category fallback footprints (0.2 = +-20%%), "
                         "so a pool of building models stops packing as one size")
    ap.add_argument("--indent", type=int, default=None, help="pretty-print the JSON")
    ap.add_argument("--verbose", action="store_true", help="let the generator narrate")
    args = ap.parse_args(argv)

    try:
        seeds = parse_seeds(args.seeds)
        out_dir = _out_dir(args.out)
    except ValueError as e:
        ap.error(str(e))
    if args.pipeline == "v2" and not args.disaster:
        ap.error("--pipeline v2 needs --disaster (e.g. --disaster earthquake "
                 "tornado explosion)")
    if args.region and args.region_range:
        ap.error("--region and --region-range are mutually exclusive")
    if args.severity is not None and args.severity_range:
        ap.error("--severity and --severity-range are mutually exclusive")
    hcfg = HumanConfig(n_casualties=args.casualties, n_bystanders=args.bystanders)
    fixed_region = tuple(args.region) if args.region else None
    # v2 iterates locale x disaster; v1 keeps iterating presets, with the
    # disaster left to the preset unless one is named.
    sources = [args.locale] if args.pipeline == "v2" else args.preset
    disasters = args.disaster or [None]

    # Bad ranges, extents and severities are rejected by the library; a CLI
    # should say so on one line instead of unrolling its own stack.
    def fail(seed, source, disaster, e: Exception):
        ap.error(f"{source}/{disaster or 'preset'} seed {seed}: {e}")

    rows, total = [], 0.0
    for source in sources:
        for disaster in disasters:
            for seed in seeds:
                region = fixed_region
                severity = args.severity
                try:
                    if region is None and args.region_range:
                        region = sample_region(seed, *args.region_range)
                    if severity is None and args.severity_range:
                        severity = sample_severity(seed, disaster or source,
                                                   *args.severity_range)
                except ValueError as e:
                    fail(seed, source, disaster, e)
                t0 = time.perf_counter()
                try:
                    scene = export_scene(source, seed, region_m=region,
                                         cell_m=args.cell_m, human_cfg=hcfg,
                                         severity=severity,
                                         spawn_corner=args.spawn_corner,
                                         size_jitter=args.size_jitter,
                                         pipeline=args.pipeline, disaster=disaster,
                                         verbose=args.verbose)
                except (ValueError, KeyError, FileNotFoundError) as e:
                    fail(seed, source, disaster, e)
                dt = time.perf_counter() - t0
                stem = (f"{scene.meta.locale}_{scene.meta.disaster_type}"
                        if args.pipeline == "v2" else scene.meta.preset)
                path = out_dir / f"{stem}_{seed}.json"
                scene.to_json(path, indent=args.indent)
                row = scene_summary(scene)
                row["secs"] = dt
                rows.append(row)
                total += dt

    print(format_summary_table(rows))
    print(f"\n{len(rows)} scenes -> {out_dir}  ({total:.1f}s total, "
          f"{total / max(len(rows), 1):.2f}s/scene)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

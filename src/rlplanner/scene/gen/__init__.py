"""Vendored, pxr-free copy of the disaster-dataset procedural city generator.

Source: disaster-dataset/scene_gen @ krrishj/disaster-dataset (2026-08-20).
Divergence from upstream, deliberately minimal so a re-vendor is a copy plus
these patches:
  * `scene_generator.py`: the top-level `from pxr import ...` is wrapped in
    try/except (None on failure). Only the USD-writing helpers touch pxr;
    `build_city`, `make_damage_field` and the config layer are pure python.
  * `compile_disaster.py`: sibling imports made package-relative.
  * `config/`: presets/ (disaster specs plus the v2 `downtown.yaml` locale),
    low_level/default.yaml and asset_sets/ copied verbatim;
    low_level/compiled/ dropped (high-level specs are compiled in memory).

The v2 (detailed city) pipeline — `generate_scene.py`, `layout/`, `detail/` —
is vendored the same way. Upstream runs as a flat set of modules on
`sys.path`, so every sibling import becomes a relative one:
  * `layout/city_layout.py`: `from scene_generator import _jitter_posf` ->
    `from ..scene_generator import ...`; `from layout import _rng_range,
    _weighted` -> `from . import ...`; the two lazy `from detail import
    districts` (in `subdivide`) -> `from ..detail import districts`; the two
    lazy `import scene_generator as sg` (in `patched`) -> `from .. import
    scene_generator as sg`.
  * `detail/districts.py`: `from scene_generator import ...` ->
    `from ..scene_generator import ...`; lazy `from layout import
    city_layout` -> `from ..layout import city_layout`; lazy `from detail
    import parks` -> `from . import parks`.
  * `detail/parks.py`: `from scene_generator import (...)` ->
    `from ..scene_generator import (...)`; lazy `from detail import districts`
    -> `from . import districts`.
  * `generate_scene.py`, four patches, all so the pipeline can run without a
    USD stage:
      1. imports made relative, and `city_detail` / `road_markings` are NOT
         vendored — they only decorate a stage (street furniture, road paint)
         and produce no 2.5D geometry. They stay as module attributes set to
         `None`, and `generate_scene_on_stage` skips each when it is None, so
         the call sequence still reads like upstream and a test can stub them.
      2. `prune_prims`, `stamp_asset_provenance` and `apply_surface_overrides`
         return 0 immediately when `stage is None` (they import pxr and walk
         prims; there is nothing to walk).
      3. `sg.apply_placements` / `sg.apply_ground_planes` are called only when
         `stage is not None`.
      4. `main()`'s `from compile_disaster import load_scene_config` made
         relative.
    So `generate_scene_on_stage(None, config)` runs the whole pure-python half
    and returns the placement list. `scene/export.py::build_v2` repeats those
    sub-steps directly (same order, same `random.Random(seed + 7717)`), and
    `tests/test_export_v2.py` asserts the two agree placement for placement.

Not patched, worth knowing:
  * `city_layout.PARK_RESERVES` is a module global ("one city per process"
    upstream). Every v2 run re-enters `city_layout.patched`, which rewrites it,
    so exporting many scenes in one process is safe; nothing else reads it.
  * `generate_scene.check_duplicate_yaml_keys()` re-reads and re-parses every
    config file on each call. `build_v2` does not call it — it validates the
    configs, it does not generate anything.
"""
from . import compile_disaster, compile_locale, scene_generator  # noqa: F401
from . import generate_scene  # noqa: F401

__all__ = ["scene_generator", "compile_disaster", "compile_locale",
           "generate_scene"]

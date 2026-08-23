"""Window-vs-file policy for the scripts.

A script opens a window unless it was told to write a file: `--out` (or `--record`) writes and
stays headless; otherwise a GUI backend is used when one is usable (a display exists and an
explicit `MPLBACKEND` is not a headless one); otherwise a default file is written and its path
printed. An explicit `MPLBACKEND` always wins, so `MPLBACKEND=Agg` keeps tests deterministic.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HEADLESS_BACKENDS = frozenset({"agg", "pdf", "ps", "svg", "cairo", "template", "pgf"})
GUI_BACKENDS = ("QtAgg", "TkAgg", "GTK4Agg", "GTK3Agg", "MacOSX")


def explicit_backend() -> str | None:
    """`MPLBACKEND` if the user set one (it wins over any auto-selection)."""
    b = (os.environ.get("MPLBACKEND") or "").strip()
    return b or None


def is_headless_backend(name: str | None) -> bool:
    return str(name or "agg").strip().lower() in HEADLESS_BACKENDS


def has_display() -> bool:
    if sys.platform.startswith(("win", "darwin")):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def gui_possible() -> bool:
    """True when a window could be opened: a display exists and MPLBACKEND is not headless."""
    b = explicit_backend()
    if b is not None and is_headless_backend(b):
        return False
    return has_display()


def select_backend(want_gui: bool) -> str:
    """Pick the backend *before* any figure exists; returns the active backend name.

    `matplotlib.use` only records the name — a missing Qt/Tk binding surfaces at the first
    `plt.figure()` — so each candidate is resolved with `switch_backend`, which imports it now.
    """
    import matplotlib

    if explicit_backend() is not None:
        return matplotlib.get_backend()
    matplotlib.use("Agg", force=True)             # a cheap, always-importable starting point
    if not (want_gui and has_display()):
        return matplotlib.get_backend()
    import matplotlib.pyplot as plt
    for cand in GUI_BACKENDS:
        try:
            plt.switch_backend(cand)
            return matplotlib.get_backend()
        except Exception:                         # noqa: BLE001 - binding missing: try the next
            continue
    plt.switch_backend("Agg")
    return matplotlib.get_backend()


def gui_active() -> bool:
    import matplotlib

    return not is_headless_backend(matplotlib.get_backend())


def save(fig, path: str | Path, tag: str = "viz", dpi: int | None = None,
         tight: bool = True) -> Path:
    """Write `fig`; `tight` keeps legends sitting outside the axes from being clipped."""
    p = Path(path)
    if p.parent != Path(""):
        p.parent.mkdir(parents=True, exist_ok=True)
    kw = {"bbox_inches": "tight"} if tight else {}
    if dpi is not None:
        kw["dpi"] = int(dpi)
    fig.savefig(p, **kw)
    print(f"[{tag}] wrote {p}")
    return p


def finish(fig, out: str | Path | None, default_out: str | Path, tag: str = "viz",
           dpi: int | None = None, tight: bool = True) -> Path | None:
    """`out` wins; else a window when one is usable; else `default_out`. None = a window was shown."""
    if out is not None:
        return save(fig, out, tag, dpi, tight)
    if gui_active():
        import matplotlib.pyplot as plt
        try:
            plt.show()
            return None
        except Exception as exc:                 # noqa: BLE001 - backend could not open a window
            print(f"[{tag}] {type(exc).__name__}: {exc}; writing {default_out} instead")
    return save(fig, default_out, tag, dpi, tight)


__all__ = ["explicit_backend", "is_headless_backend", "has_display", "gui_possible",
           "select_backend", "gui_active", "save", "finish", "GUI_BACKENDS", "HEADLESS_BACKENDS"]

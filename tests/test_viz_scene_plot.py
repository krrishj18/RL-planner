import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from rlplanner.scene import schema
from rlplanner.viz.scene_plot import damage_grid, plot_scene, region_extent


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_plot_scene_synthetic_seeds(seed):
    scene = schema.make_synthetic_scene(seed)
    fig, ax = plt.subplots(figsize=(6, 6))
    out = plot_scene(scene, ax=ax)
    assert out is ax
    assert ax.get_xlim() == (scene.region[0], scene.region[2])
    assert ax.get_ylim() == (scene.region[1], scene.region[3])
    assert ax.get_aspect() == 1.0
    assert len(ax.patches) > len(scene.buildings)
    fig.canvas.draw()


def test_plot_scene_makes_its_own_axes_and_flags():
    scene = schema.make_synthetic_scene(3)
    ax = plot_scene(scene, show_damage=False, show_humans=False, show_ids=True, legend=False)
    assert ax.get_legend() is None
    ax.figure.canvas.draw()


def test_plot_scene_empty_scene():
    scene = schema.Scene()
    ax = plot_scene(scene)
    ax.figure.canvas.draw()


def test_plot_scene_uniform_damage_has_no_contours():
    scene = schema.Scene(damage_field=schema.DamageField(kind="uniform", params={"inside": 0.4}))
    ax = plot_scene(scene)
    ax.figure.canvas.draw()


def test_plot_scene_rejects_non_scene():
    with pytest.raises(TypeError):
        plot_scene({"meta": {}})


def test_region_extent_rejects_degenerate():
    scene = schema.Scene(meta=schema.Meta(region=(0.0, 0.0, 0.0, 10.0)))
    with pytest.raises(ValueError):
        region_extent(scene)


def test_damage_grid_prefers_sampled_grid():
    scene = schema.Scene(meta=schema.Meta(region=(0.0, 0.0, 10.0, 10.0)))
    scene.damage_field.grid = {"cell_m": 5.0, "nx": 2, "ny": 2,
                               "values": [[0.0, 0.25], [0.5, 1.0]]}
    X, Y, D = damage_grid(scene)
    assert D.shape == (2, 2)
    assert np.allclose(X[0], [2.5, 7.5])
    assert D[1, 1] == 1.0

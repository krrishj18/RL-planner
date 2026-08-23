import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from rlplanner.scene import schema
from rlplanner.viz.raster_plot import LAYERS, class_rgb, plot_raster, raster_extent
from viz_mocks import make_fake_raster


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def fake():
    return make_fake_raster(schema.make_synthetic_scene(1), cell_m=2.0)


@pytest.mark.parametrize("layer", LAYERS)
def test_plot_raster_layers(fake, layer):
    fig, ax = plt.subplots(figsize=(5, 5))
    out = plot_raster(fake, ax=ax, layer=layer)
    assert out is ax
    assert ax.get_aspect() == 1.0
    assert len(ax.images) == 1
    fig.canvas.draw()


def test_raster_extent_matches_region(fake):
    x0, x1, y0, y1 = raster_extent(fake)
    assert (x0, y0) == tuple(fake.origin)
    assert x1 == fake.origin[0] + fake.nx * fake.cell_m


def test_class_rgb_shape_and_shading(fake):
    rgb = class_rgb(fake)
    assert rgb.shape == (fake.ny, fake.nx, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    assert not np.allclose(class_rgb(fake, shade=False), rgb)


def test_bad_layer_raises(fake):
    with pytest.raises(ValueError):
        plot_raster(fake, layer="nope")


def test_missing_fields_raise():
    class Bare:
        pass

    with pytest.raises(AttributeError):
        raster_extent(Bare())


def test_wrong_shape_raises(fake):
    import dataclasses
    bad = dataclasses.replace(fake, cls=np.zeros((3, 3), np.int8))
    with pytest.raises(ValueError):
        plot_raster(bad)


def test_out_of_range_class_raises(fake):
    import dataclasses
    bad = dataclasses.replace(fake, cls=np.full((fake.ny, fake.nx), 99, np.int8))
    with pytest.raises(ValueError):
        plot_raster(bad)


def test_negative_cell_raises(fake):
    import dataclasses
    with pytest.raises(ValueError):
        raster_extent(dataclasses.replace(fake, cell_m=0.0))


def test_plot_raster_on_real_raster_if_available():
    rasterize = pytest.importorskip("rlplanner.sim.raster").rasterize
    r = rasterize(schema.make_synthetic_scene(0), 2.0)
    ax = plot_raster(r, layer="cls")
    ax.figure.canvas.draw()

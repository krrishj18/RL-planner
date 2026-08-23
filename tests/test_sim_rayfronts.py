import math

import numpy as np
import pytest

from rlplanner.scene.schema import (Building, CLASS_ID, DamageField, Debris, Human, Meta, Scene,
                                    Vehicle, make_synthetic_scene)
from rlplanner.sim.config import EnvConfig
from rlplanner.sim.raster import rasterize
from rlplanner.sim.rayfronts_sim import RayFrontsSim, _range_factor
from rlplanner.sim.state import RobotState

W = 200.0
PERSON_Q = ("person lying on the ground", "person")

# Query similarity is a *view* now: the tests take it themselves, exactly like a viewer or a
# baseline does, and never expect the belief to carry a per-query column.
ALL_Q = ("person lying on the ground", "person", "collapsed building", "damaged building",
         "rubble", "overturned car", "car", "bus stop", "road", "tree", "house")


def _scene(humans=(), **kw):
    return Scene(meta=Meta(region=(-W / 2, -W / 2, W / 2, W / 2)),
                 damage_field=DamageField(kind="uniform", params={"inside": 0.0}),
                 humans=list(humans), **kw)


def _robot(x=0.0, y=0.0, yaw=0.0, alt=25.0, idx=0):
    return RobotState(idx=idx, pos=np.array([x, y], float), alt=alt, heading=yaw,
                      target_xy=None, target_token_type=0, target_id=-1)


def _sim(scene, cfg=None, seed=0):
    cfg = cfg or EnvConfig()
    ras = rasterize(scene, cfg.raster.cell_m)
    rng = np.random.default_rng(seed)
    return RayFrontsSim(ras, cfg, rng), rng, ras, cfg


def _run(rf, rng, robots, n, t0=0.0, dt=1.0):
    for k in range(n):
        rf.update(robots, t0 + k * dt, rng)
    rf.end_of_decision(t0 + n * dt, robots)


def _sim_of(rf, feat, names=PERSON_Q) -> float:
    """Best cosine of a stored feature against a set of query names — computed here, on demand."""
    q = rf.emb.embed_queries(tuple(names))
    f = np.asarray(feat, np.float32)
    f = f / max(float(np.linalg.norm(f)), 1e-12)
    return float(np.clip(f @ q.T, 0.0, 1.0).max())


def _person(rf, feat) -> float:
    return _sim_of(rf, feat, PERSON_Q)


def _argmax_query(rf, feat, names=ALL_Q) -> str:
    q = rf.emb.embed_queries(tuple(names))
    f = np.asarray(feat, np.float32)
    f = f / max(float(np.linalg.norm(f)), 1e-12)
    return names[int(np.argmax(f @ q.T))]


def _class_cos(rf, cls_name, query) -> float:
    return float(np.clip(rf.class_emb[CLASS_ID[cls_name]] @ rf.query_vec(query), 0.0, 1.0))


def _forward_target(rf, tol_deg=12.0):
    """The most person-like ray target within `tol_deg` of +x, or None."""
    cand = [T for T in rf.ray_targets if abs(math.degrees(T.az)) < tol_deg]
    return max(cand, key=lambda T: _person(rf, T.feat)) if cand else None


# ---- voxels ----------------------------------------------------------------------------------
def test_voxel_bookkeeping():
    rf, rng, ras, cfg = _sim(make_synthetic_scene(0, region_m=(W, W)))
    _run(rf, rng, [_robot(-40.0, -40.0, 0.5)], 8)
    assert np.array_equal(rf.observed, rf.vox_cnt > 0)
    assert rf.observed.any()
    assert (rf.last_seen_t[rf.observed] >= 0).all()
    assert (rf.last_seen_t[~rf.observed] == -1).all()
    feat = rf.vox_feat[rf.observed]                       # unit per-cell features
    assert np.allclose(np.linalg.norm(feat, axis=1), 1.0, atol=1e-5)
    assert (rf.vox_feat_sum[~rf.observed] == 0).all()
    assert not hasattr(rf, "vox_sim"), "the belief must not keep a per-query grid"
    # the query view is derived on demand and agrees with the stored features
    sim = rf.query_sim("road")
    assert (sim >= 0).all() and (sim <= 1).all() and (sim[~rf.observed] == 0).all()
    assert np.allclose(sim[rf.observed], np.clip(feat @ rf.query_vec("road"), 0, 1), atol=1e-5)


def test_voxel_memory_is_never_cleared():
    """RayFronts' map is persistent: a cell keeps its feature sum and hit count for ever."""
    rf, rng, _, _ = _sim(_scene(), seed=0)
    _run(rf, rng, [_robot(0.0, 0.0, 0.0)], 3)
    obs0 = rf.observed.copy()
    cnt0 = rf.vox_cnt.copy()
    _run(rf, rng, [_robot(60.0, 60.0, 2.0)], 3, t0=3.0)    # look somewhere else entirely
    assert (rf.observed | obs0 == rf.observed).all()
    assert (rf.vox_cnt[obs0] >= cnt0[obs0]).all()


def test_voxel_similarity_tracks_the_class_row():
    cfg = EnvConfig()
    cfg.rayfronts.p_confuse = 0.0
    cfg.rayfronts.vox_noise_std = 0.0
    rf, rng, ras, cfg = _sim(_scene(), cfg)
    _run(rf, rng, [_robot(0.0, 0.0, 0.0)], 3)
    obs = rf.observed
    assert np.allclose(rf.query_sim("road")[obs], _class_cos(rf, "ground", "road"), atol=1e-6)


def test_confusion_moves_some_cells_off_their_class_row():
    cfg = EnvConfig()
    cfg.rayfronts.vox_noise_std = 0.0
    cfg.rayfronts.p_confuse = 1.0
    rf, rng, _, cfg = _sim(_scene(), cfg)
    _run(rf, rng, [_robot(0.0, 0.0, 0.0)], 1)
    v = rf.query_sim("road")[rf.observed]
    assert (v != _class_cos(rf, "ground", "road")).mean() > 0.5


# ---- humans ----------------------------------------------------------------------------------
def _observe_rate(visibility, container, trials=150, d=20.0):
    hits = 0
    hum = Human(id="h", pos=(d, 0.0, 0.0), role="casualty", pose="prone",
                container=container, visibility=visibility)
    sc = _scene([hum])
    for s in range(trials):
        rf, rng, _, _ = _sim(sc, seed=1000 + s)
        rf.update([_robot(0.0, 0.0, 0.0)], 0.0, rng)
        hits += int(rf.human_hits[0] > 0)
    return hits / trials


def test_observation_probability_ordering_by_visibility():
    p_open = _observe_rate("open", "open")
    p_part = _observe_rate("partial", "vehicle")
    p_occ = _observe_rate("occluded", "rubble")
    assert p_open > p_part > p_occ
    assert p_open > 0.35 and p_occ < 0.2


def test_range_factor_shape():
    d = 35.0
    assert _range_factor(np.array([0.0, 10.0]), 0.5, d) == pytest.approx([1.0, 1.0])
    assert _range_factor(np.array([d]), 0.5, d) == pytest.approx([0.5])
    assert _range_factor(np.array([0.75 * d]), 0.5, d) == pytest.approx([0.75])
    assert _range_factor(np.array([d + 1]), 0.5, d) == pytest.approx([0.25])


def test_a_human_is_found_by_voxel_hit_count_alone():
    """`found_hits` observations of the human's own cell, and nothing else."""
    cfg = EnvConfig()
    cfg.rayfronts.p_fp_ray = 0.0
    hum = Human(id="h", pos=(20.0, 0.0, 0.0), role="casualty", pose="prone")
    rf, rng, ras, cfg = _sim(_scene([hum]), cfg, seed=5)
    rb = _robot(0.0, 0.0, 0.0)
    rf.update([rb], 0.0, rng)
    assert bool(rf.human_found[0]) == (rf.human_hits[0] >= cfg.rayfronts.found_hits)
    _run(rf, rng, [rb], 6, t0=1.0)
    assert rf.human_hits[0] >= cfg.rayfronts.found_hits and rf.human_found[0]
    i, j = ras.xy_to_ij(20.0, 0.0)
    assert _person(rf, rf.vox_feat_sum[i, j]) > 0.5    # the person is in the voxel map, too


def test_found_needs_the_full_hit_count():
    cfg = EnvConfig()
    cfg.rayfronts.found_hits = 6
    cfg.rayfronts.p_fp_ray = 0.0
    hum = Human(id="h", pos=(20.0, 0.0, 0.0), role="casualty", pose="prone")
    rf, rng, _, cfg = _sim(_scene([hum]), cfg, seed=5)
    rb = _robot(0.0, 0.0, 0.0)
    rf.update([rb], 0.0, rng)
    assert not rf.human_found[0]
    _run(rf, rng, [rb], 12, t0=1.0)
    assert rf.human_found[0] == (rf.human_hits[0] >= 6)
    assert rf.human_hits[0] >= 6


def test_a_bystander_is_found_the_same_way_and_carries_no_role():
    hum = Human(id="h", pos=(20.0, 0.0, 0.0), role="bystander", pose="standing")
    rf, rng, _, _ = _sim(_scene([hum]), seed=5)
    _run(rf, rng, [_robot(0.0, 0.0, 0.0)], 6)
    assert rf.human_found[0]                            # the map cannot tell the two apart
    ev = rf.update([_robot(0.0, 0.0, 0.0)], 7.0, rng)
    assert not [e for e in ev if e.kind == "found"]     # only the first crossing is an event


def test_a_human_behind_a_wall_is_never_observed():
    sc = _scene([Human(id="h", pos=(30.0, 0.0, 0.0), role="casualty", pose="prone",
                       container="building", visibility="partial")],
                buildings=[Building(id="b", center=(15.0, 0.0), size=(6.0, 40.0),
                                    category="highrise")])
    rf, rng, _, _ = _sim(sc, seed=7)
    _run(rf, rng, [_robot(0.0, 0.0, 0.0)], 20)
    assert rf.human_hits[0] == 0 and not rf.human_found[0]


# ---- rays: what is visible from afar ----------------------------------------------------------
def _far_case(sc, seed=3, n=12):
    cfg = EnvConfig()
    cfg.rayfronts.p_fp_ray = 0.0
    rf, rng, _, cfg = _sim(sc, cfg, seed=seed)
    _run(rf, rng, [_robot(0.0, 0.0, 0.0)], n)
    return rf, cfg


def test_a_far_open_human_owns_its_ray_and_no_voxel_hit():
    """A human beyond the depth limit is only ever a semantic ray, never a voxel hit."""
    rf, cfg = _far_case(_scene([Human(id="h", pos=(60.0, 0.0, 0.0), role="casualty",
                                      pose="prone")]))
    assert rf.human_hits[0] == 0 and not rf.human_found[0]
    T = _forward_target(rf)
    assert T is not None
    assert _argmax_query(rf, T.feat) in PERSON_Q
    assert _person(rf, T.feat) > 0.6


def test_a_far_human_in_a_car_reads_as_a_car():
    """`partial` visibility is invisible from afar: the ray carries the container the drone sees."""
    sc = _scene([Human(id="h", pos=(60.0, 0.0, 0.0), role="casualty", pose="prone",
                       container="vehicle", visibility="partial", container_id="v")],
                vehicles=[Vehicle(id="v", center=(60.0, 0.0), state="toppled")])
    rf, cfg = _far_case(sc)
    T = _forward_target(rf)
    assert T is not None
    assert _argmax_query(rf, T.feat) in ("overturned car", "car")
    assert _person(rf, T.feat) < 0.55
    assert rf.human_hits[0] == 0


def test_a_far_human_under_rubble_reads_as_rubble():
    sc = _scene([Human(id="h", pos=(60.0, 0.0, 0.0), role="casualty", pose="prone",
                       container="rubble", visibility="occluded", container_id="d")],
                debris=[Debris(id="d", center=(60.0, 0.0), radius_m=8.0)])
    rf, cfg = _far_case(sc)
    T = _forward_target(rf)
    assert T is not None
    assert _argmax_query(rf, T.feat) in ("rubble", "collapsed building")
    assert _person(rf, T.feat) < 0.55


def test_a_far_human_behind_buildings_raises_no_ray():
    sc = _scene([Human(id="h", pos=(60.0, 0.0, 0.0), role="casualty", pose="prone")],
                buildings=[Building(id=f"b{k}", center=(30.0, -20.0 + 10.0 * k), size=(8.0, 10.0),
                                    category="highrise") for k in range(5)])
    rf, cfg = _far_case(sc)
    assert rf.human_hits[0] == 0
    assert all(_person(rf, T.feat) < 0.55 for T in rf.ray_targets)


def test_a_human_in_a_car_is_invisible_at_60m_and_mapped_from_15m():
    """The two regimes together: the car's ray is all a distant drone gets, but flying to the car
    puts the human inside the depth limit, where the per-visibility probability applies."""
    sc = _scene([Human(id="h", pos=(60.0, 0.0, 0.0), role="casualty", pose="prone",
                       container="vehicle", visibility="partial", container_id="v")],
                vehicles=[Vehicle(id="v", center=(60.0, 0.0), state="toppled")])
    far, _ = _far_case(sc, seed=4, n=12)
    assert far.human_hits[0] == 0 and not far.human_found[0]

    cfg = EnvConfig()
    cfg.rayfronts.p_fp_ray = 0.0
    near, rng, _, cfg = _sim(sc, cfg, seed=4)
    _run(near, rng, [_robot(45.0, 0.0, 0.0)], 12)
    assert near.human_hits[0] >= cfg.rayfronts.found_hits and near.human_found[0]


@pytest.mark.parametrize("d", [40.0, 55.0, 60.0, 75.0])
def test_ray_target_range_matches_the_true_horizontal_distance(d):
    """`range = alt / tan(-el)` for a target on the ground, within 10%."""
    got = []
    for seed in range(6):
        rf, cfg = _far_case(_scene([Human(id="h", pos=(d, 0.0, 0.0), role="casualty",
                                          pose="prone")]), seed=seed)
        cand = [T for T in rf.ray_targets if _person(rf, T.feat) > 0.6]
        if cand:
            T = max(cand, key=lambda T: _person(rf, T.feat))
            got.append((T.range_m, math.hypot(T.xy[0], T.xy[1])))
    assert got, "no person-grade ray at all"
    for r, xy in got:
        assert abs(r - d) / d < 0.10
        assert abs(xy - d) / d < 0.10       # and the target point lands on the human


def test_the_ray_range_falls_back_when_the_elevation_is_unusable():
    rf, cfg = _far_case(_scene())
    assert rf.target_range(np.array([0.0, 0.5]))[0] == pytest.approx(cfg.rayfronts.ray_range_m)
    assert rf.target_range(np.array([-1.5])) == pytest.approx(cfg.sensor.depth_limit_m)
    assert rf.target_range(np.array([-0.01])) == pytest.approx(cfg.sensor.visual_range_m)


# ---- rays: store ------------------------------------------------------------------------------
def test_spurious_person_rays_are_emitted_at_the_configured_rate():
    cfg = EnvConfig()
    cfg.rayfronts.p_fp_ray = 1.0
    rf, rng, _, _ = _sim(_scene(), cfg, seed=0)
    _run(rf, rng, [_robot(0.0, 0.0, 0.0)], 6)
    assert rf.n_fp_rays == 6
    person = np.maximum(rf.ray_query_sim(PERSON_Q[0]), rf.ray_query_sim(PERSON_Q[1]))
    assert (person > 0.6).any()


def test_rays_resolve_once_their_cone_is_observed():
    rf, rng, _, cfg = _sim(_scene(), seed=0)
    _run(rf, rng, [_robot(0.0, 0.0, 0.0)], 3)
    st = rf.store()
    assert st.n > 0 and not st.resolved.all()
    rf.observed[:] = True
    rf.end_of_decision(10.0)
    assert rf.store().resolved.all()
    assert not rf.ray_targets


def test_rays_resolve_by_ttl():
    cfg = EnvConfig()
    cfg.rayfronts.ray_ttl_s = 5.0
    rf, rng, _, _ = _sim(_scene(), cfg, seed=0)
    _run(rf, rng, [_robot(0.0, 0.0, 0.0)], 2)
    assert not rf.store().resolved.all()
    rf.end_of_decision(100.0)
    assert rf.store().resolved.all()


def test_ray_store_grows_and_compacts_with_stable_ids():
    rf, rng, _, _ = _sim(make_synthetic_scene(1, region_m=(W, W)), seed=0)
    rb = _robot(-60.0, -60.0, 0.6)
    for k in range(40):
        rb.pos = rb.pos + np.array([2.0, 1.0])
        rb.heading += 0.05
        rf.update([rb], float(k), rng)
        if k % 5 == 4:
            rf.end_of_decision(float(k), [rb])
    st = rf.store()
    assert st.n > 20
    assert len(set(st.ids.tolist())) == st.n            # ids unique
    ids_live = set(st.ids[~st.resolved].tolist())
    rf.compact()
    st2 = rf.store()
    assert set(st2.ids.tolist()) == ids_live
    assert st2.n == len(ids_live)
    assert all(v < st2.n for v in rf._ray_key.values())


def test_ray_conf_is_capped_and_features_are_unit():
    rf, rng, _, cfg = _sim(_scene(), seed=0)
    _run(rf, rng, [_robot(0.0, 0.0, 0.0)], 40)
    st = rf.store()
    assert (st.conf <= cfg.rayfronts.ray_conf_cap + 1e-6).all()
    assert np.isfinite(st.feat).all() and (np.linalg.norm(st.feat, axis=1) <= 1.0 + 1e-5).all()
    assert np.allclose(np.linalg.norm(st.feat_peak, axis=1), 1.0, atol=1e-5)
    q = rf.ray_query_sim(PERSON_Q[0])
    assert q.shape == (st.n,) and (q >= 0).all() and (q <= 1).all()
    assert (st.n_obs > 0).all()
    assert (st.t_last >= st.t_first).all()
    assert (st.el <= 0.0).all()                          # everything a drone sees is below it


def test_rays_are_never_merged_across_origins():
    """Two robots looking at the same place from different cells keep two separate rays."""
    cfg = EnvConfig()
    cfg.rayfronts.p_fp_ray = 0.0
    sc = _scene([Human(id="h", pos=(0.0, 0.0, 0.0), role="casualty", pose="prone")])
    rf, rng, _, cfg = _sim(sc, cfg, seed=1)
    a = _robot(-60.0, 0.0, 0.0, idx=0)
    b = _robot(0.0, -60.0, math.pi / 2, idx=1)
    _run(rf, rng, [a, b], 10)
    hot = [T for T in rf.ray_targets if _person(rf, T.feat) > 0.6]
    if len(hot) >= 2:
        o = np.array([T.origin_xy for T in hot])
        assert len({tuple(np.round(x, 3)) for x in o}) >= 2
    assert len({T.id for T in rf.ray_targets}) == len(rf.ray_targets)


# ---- segments ---------------------------------------------------------------------------------
def test_segments_cover_only_observed_cells_and_carry_a_feature():
    cfg = EnvConfig()
    rf, rng, ras, cfg = _sim(make_synthetic_scene(0, region_m=(W, W)), cfg, seed=0)
    rb = _robot(-40.0, -40.0, 0.5)
    _run(rf, rng, [rb], 10)
    segs = rf.segments
    assert segs
    lab = rf.seg_labels
    assert (lab[~rf.observed] == -1).all(), "a segment must never contain an unobserved cell"
    assert (lab[rf.observed] >= 0).all()
    for sgm in segs:
        assert rf.observed[sgm.ij] and sgm.n_cells >= 1 and sgm.mean_hits >= 1.0
        assert lab[sgm.ij] >= 0
        assert abs(float(np.linalg.norm(sgm.feat)) - 1.0) < 1e-5
        assert sgm.ray_count >= 0


def test_a_person_cell_is_in_a_segment_the_query_can_score():
    cfg = EnvConfig()
    cfg.rayfronts.p_fp_ray = 0.0
    hum = Human(id="h", pos=(20.0, 0.0, 0.0), role="casualty", pose="prone")
    rf, rng, ras, cfg = _sim(_scene([hum]), cfg, seed=2)
    rb = _robot(0.0, 0.0, 0.0)
    for k in range(20):
        rf.update([rb], float(k), rng)
    rf.end_of_decision(20.0, [rb])
    i, j = ras.xy_to_ij(20.0, 0.0)
    assert _person(rf, rf.vox_feat_sum[i, j]) > 0.4
    assert rf.seg_labels[i, j] >= 0                # the person's cell belongs to some segment


def test_segment_ids_are_stable_while_the_map_does_not_move():
    rf, rng, _, _ = _sim(make_synthetic_scene(0, region_m=(W, W)), seed=0)
    rb = _robot(-40.0, -40.0, 0.5)
    _run(rf, rng, [rb], 8)
    before = {sgm.id: tuple(np.round(sgm.xy, 6)) for sgm in rf.segments}
    rf.end_of_decision(8.0, [rb])
    after = {sgm.id: tuple(np.round(sgm.xy, 6)) for sgm in rf.segments}
    assert before == after


def test_segmentation_is_not_rerun_every_decision():
    """The labels are refreshed once enough of the map is new; the statistics every decision."""
    rf, rng, _, _ = _sim(make_synthetic_scene(0, region_m=(W, W)), seed=0)
    rb = _robot(-40.0, -40.0, 0.5)
    _run(rf, rng, [rb], 8)
    at = rf._seg_dec_at
    lab = rf.seg_labels.copy()
    rf.end_of_decision(8.5, [rb])                  # nothing new observed
    assert rf._seg_dec_at == at and np.array_equal(rf.seg_labels, lab)


# ---- frontiers --------------------------------------------------------------------------------
def test_frontiers_exist_then_vanish_when_everything_is_observed():
    rf, rng, _, _ = _sim(_scene(), seed=0)
    _run(rf, rng, [_robot(0.0, 0.0, 0.0)], 4)
    assert rf.frontier_clusters and rf.frontier_mask.any()
    for c in rf.frontier_clusters:
        assert c.n_cells >= 1 and c.info_gain >= 0
        assert rf.frontier_mask[c.cell_ij[:, 0], c.cell_ij[:, 1]].all()
    rf.observed[:] = True
    rf.end_of_decision(9.0)
    assert not rf.frontier_clusters and not rf.frontier_mask.any()


def test_frontier_cells_are_observed_and_touch_the_unknown():
    rf, rng, ras, _ = _sim(_scene(), seed=0)
    _run(rf, rng, [_robot(0.0, 0.0, 0.7)], 3)
    f = rf.frontier_mask
    assert (f <= rf.observed).all()
    un = ~rf.observed
    nb = np.zeros_like(f)
    nb[:-1, :] |= un[1:, :]
    nb[1:, :] |= un[:-1, :]
    nb[:, :-1] |= un[:, 1:]
    nb[:, 1:] |= un[:, :-1]
    assert np.array_equal(f, rf.observed & nb)


def test_frontier_ids_are_stable_while_the_cluster_persists():
    rf, rng, _, _ = _sim(_scene(), seed=0)
    _run(rf, rng, [_robot(0.0, 0.0, 0.0)], 3)
    before = {c.id: tuple(c.centroid_xy) for c in rf.frontier_clusters}
    rf.end_of_decision(4.0)                       # nothing changed -> same ids
    after = {c.id: tuple(c.centroid_xy) for c in rf.frontier_clusters}
    assert before == after


# ---- determinism ------------------------------------------------------------------------------
def test_same_seed_same_belief():
    sc = make_synthetic_scene(2, region_m=(W, W))
    outs = []
    for _ in range(2):
        rf, rng, _, _ = _sim(sc, seed=11)
        rb = _robot(-50.0, -50.0, 0.4)
        for k in range(20):
            rb.pos = rb.pos + np.array([3.0, 2.0])
            rb.heading += 0.03
            rf.update([rb], float(k), rng)
            if k % 5 == 4:
                rf.end_of_decision(float(k), [rb])
        st = rf.store()
        outs.append((rf.vox_feat_sum.copy(), rf.vox_cnt.copy(), st.feat.copy(), st.ids.copy(),
                     rf.seg_labels.copy(), [c.id for c in rf.frontier_clusters],
                     [c.id for c in rf.segments]))
    for a, b in zip(outs[0], outs[1]):
        assert np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b


def test_no_humans_scene_is_fine():
    rf, rng, _, _ = _sim(_scene(), seed=0)
    _run(rf, rng, [_robot(0.0, 0.0, 0.0)], 3)
    assert rf.human_hits.shape == (0,) and rf.human_found.shape == (0,)


# ---- the open-set contract ---------------------------------------------------------------------
def test_no_query_is_evaluated_while_the_belief_updates():
    """The point of the change: a decision must never scan the map against the query list."""
    rf, rng, _, _ = _sim(make_synthetic_scene(0, region_m=(W, W)), seed=0)
    rb = _robot(-40.0, -40.0, 0.5)
    rf.query_sim("person")                       # a deliberate view does count
    assert rf.n_query_calls == 1
    n0 = rf.n_query_calls
    for k in range(15):
        rf.update([rb], float(k), rng)
        if k % 5 == 4:
            rf.end_of_decision(float(k), [rb])
    assert rf.n_query_calls == n0


def test_set_queries_moves_nothing_but_the_queries():
    rf, rng, _, _ = _sim(make_synthetic_scene(0, region_m=(W, W)), seed=0)
    _run(rf, rng, [_robot(-40.0, -40.0, 0.5)], 8)
    feat = rf.vox_feat_sum.copy()
    segs = [s.id for s in rf.segments]
    rays = [r.id for r in rf.ray_targets]
    rf.set_queries(("car", "tree"), weights=[1.0, 0.3])
    assert np.array_equal(feat, rf.vox_feat_sum)
    assert segs == [s.id for s in rf.segments] and rays == [r.id for r in rf.ray_targets]
    assert rf.queries == ("car", "tree")
    assert rf.query_w.tolist() == [1.0, pytest.approx(0.3)]
    assert rf.query_emb.shape == (2, rf.D)

import json
import math

import numpy as np
import pytest

from rlplanner.scene.schema import CLASS_ID, CLASS_NAMES, DamageField, Meta, Scene, N_CLASSES
from rlplanner.sim import embeddings as E
from rlplanner.sim.config import DEFAULT_QUERIES, EnvConfig, RayFrontsConfig
from rlplanner.sim.raster import rasterize
from rlplanner.sim.rayfronts_sim import RayFrontsSim
from rlplanner.sim.similarity_table import Q as TABLE_QUERIES
from rlplanner.sim.similarity_table import build_sim_table
from rlplanner.sim.state import RobotState

W = 140.0
# the hand-authored table is what the *embeddings* are factorized from; the mission query list
# (DEFAULT_QUERIES) is a separate thing entirely and never indexes the belief
TABLE = build_sim_table(TABLE_QUERIES)
PERSON_QUERY_IDX = (0, 1)


def _qsim(rf, name):
    """Query view taken on demand — the belief keeps no per-query grid."""
    return rf.query_sim(name)


def _ray_sim(rf, name, peak=True):
    return rf.ray_query_sim(name, peak)


def _scene(humans=()):
    return Scene(meta=Meta(region=(-W / 2, -W / 2, W / 2, W / 2)),
                 damage_field=DamageField(kind="uniform", params={"inside": 0.0}),
                 humans=list(humans))


def _robot(x=0.0, y=0.0, yaw=0.0, alt=25.0, idx=0):
    return RobotState(idx=idx, pos=np.array([x, y], float), alt=alt, heading=yaw,
                      target_xy=None, target_token_type=0, target_id=-1)


def _sim(cfg=None, scene=None, seed=0):
    cfg = cfg or EnvConfig()
    ras = rasterize(scene if scene is not None else _scene(), cfg.raster.cell_m)
    rng = np.random.default_rng(seed)
    return RayFrontsSim(ras, cfg, rng), rng, cfg


def _run(rf, rng, robots, n, t0=0.0):
    for k in range(n):
        rf.update(robots, t0 + k, rng)
    rf.end_of_decision(t0 + n, robots)


# ---- the table itself --------------------------------------------------------------------------
def test_fallback_reproduces_the_hand_authored_table():
    emb = E.get_embedding_table(TABLE_QUERIES, dim=24)
    assert emb.source == "factorized" and emb.D == 24
    cos = emb.class_emb @ emb.query_emb.T
    assert cos.shape == (N_CLASSES, len(TABLE_QUERIES))
    assert np.abs(cos - TABLE).max() < 0.05
    assert np.allclose(emb.sim_table(), np.clip(TABLE, 0, 1), atol=0.05)


def test_embeddings_are_unit_norm():
    emb = E.get_embedding_table(TABLE_QUERIES, dim=24)
    assert np.allclose(np.linalg.norm(emb.class_emb, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(emb.query_emb, axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("dim", [12, 16, 24, 32])
def test_factorization_tolerance_across_dims(dim):
    c, q = E.factorize_table(TABLE, dim=dim)
    assert np.abs(c @ q.T - TABLE).max() < E.FACTOR_TOL


def test_a_dim_too_small_to_represent_the_table_is_rejected():
    with pytest.raises(ValueError, match="cannot represent"):
        E.factorized_table(TABLE_QUERIES, dim=8)
    with pytest.raises(ValueError, match="dim must be >= 2"):
        E.factorize_table(TABLE, dim=1)


def test_semantic_invariants_survive_the_factorization():
    emb = E.get_embedding_table(TABLE_QUERIES, dim=24)
    cos = emb.class_emb @ emb.query_emb.T
    q_lying = TABLE_QUERIES.index("person lying on the ground")
    for c in [n for n in CLASS_NAMES if not n.startswith("human")]:
        for q in PERSON_QUERY_IDX:
            assert cos[CLASS_ID[c], q] <= 0.2, (c, TABLE_QUERIES[q])
    assert int(np.argmax(cos[:, q_lying])) == CLASS_ID["human_prone"]
    assert int(np.argmax(cos[:, TABLE_QUERIES.index("person")])) == CLASS_ID["human_standing"]
    assert cos[CLASS_ID["building_destroyed"], TABLE_QUERIES.index("collapsed building")] > 0.8


def test_similarity_is_cosine():
    emb = E.get_embedding_table(TABLE_QUERIES, dim=24)
    f = emb.class_emb[CLASS_ID["debris"]] * 3.7          # scale must not matter
    s = emb.similarity(f)
    assert s.shape == (len(TABLE_QUERIES),)
    assert np.allclose(s, emb.class_emb[CLASS_ID["debris"]] @ emb.query_emb.T, atol=1e-5)
    batch = emb.similarity(emb.class_emb)
    assert batch.shape == (N_CLASSES, len(TABLE_QUERIES))
    with pytest.raises(ValueError, match="feature dim"):
        emb.similarity(np.zeros((2, 5), np.float32))


def test_unknown_query_raises_with_a_clear_message():
    emb = E.get_embedding_table(TABLE_QUERIES, dim=24)
    with pytest.raises(ValueError, match="a fire hydrant"):
        emb.embed_queries(("a fire hydrant",))
    with pytest.raises(ValueError, match="build_text_embeddings"):
        emb.embed_queries(("a fire hydrant",))
    with pytest.raises(ValueError, match="empty query list"):
        emb.embed_queries(())


def test_load_embeddings_validates(tmp_path):
    p = tmp_path / "e.json"
    p.write_text(json.dumps({"dim": 4, "class_names": list(CLASS_NAMES), "queries": ["person"],
                             "class_emb": np.eye(N_CLASSES, 4).tolist(),
                             "query_emb": [[1.0, 0.0, 0.0, 0.0]]}))
    t = E.load_embeddings(p)
    assert t.D == 4 and t.names == ("person",) and t.source.startswith("cache:")
    p.write_text(json.dumps({"dim": 4, "class_names": ["nope"], "queries": ["person"],
                             "class_emb": [[1.0]], "query_emb": [[1.0]]}))
    with pytest.raises(ValueError, match="class_names"):
        E.load_embeddings(p)
    p.write_text(json.dumps({"dim": 4, "class_names": list(CLASS_NAMES)}))
    with pytest.raises(ValueError, match="missing key"):
        E.load_embeddings(p)
    with pytest.raises(FileNotFoundError):
        E.load_embeddings(tmp_path / "absent.json")


def test_shipped_siglip_cache_is_loadable_and_carries_the_pca_basis():
    p = E.DATA_PATH.parent / "text_embeddings_siglip_vitb16.json"
    if not p.exists():
        pytest.skip("SigLIP cache not shipped")
    t = E.load_embeddings(p, DEFAULT_QUERIES)
    assert t.D == 24 and t.names == tuple(DEFAULT_QUERIES)
    assert np.allclose(np.linalg.norm(t.class_emb, axis=1), 1.0, atol=1e-5)
    assert "pca" in t.meta and len(t.meta["pca"]["components"]) == 24
    cos = t.class_emb @ t.query_emb.T
    q = TABLE_QUERIES.index("person lying on the ground")
    assert int(np.argmax(cos[:, q])) == CLASS_ID["human_prone"]


# ---- the belief carries features ---------------------------------------------------------------
def test_vox_feat_is_unit_and_the_query_view_is_its_cosine():
    rf, rng, cfg = _sim()
    _run(rf, rng, [_robot(0.0, 0.0, 0.4)], 4)
    obs = rf.observed
    assert rf.vox_feat_sum.shape == (rf.raster.ny, rf.raster.nx, cfg.rayfronts.embedding_dim)
    feat = rf.vox_feat[obs]
    assert np.allclose(np.linalg.norm(feat, axis=1), 1.0, atol=1e-5)
    direct = np.clip(rf.emb.similarity(feat), 0.0, 1.0)
    view = np.stack([_qsim(rf, q) for q in rf.queries])
    assert np.allclose(view[:, obs].T, direct, atol=1e-5)
    assert (rf.vox_feat_sum[~obs] == 0).all()


def test_noise_free_voxel_feature_is_the_class_embedding():
    cfg = EnvConfig()
    cfg.rayfronts.feat_noise_std = 0.0
    cfg.rayfronts.p_confuse = 0.0
    rf, rng, _ = _sim(cfg)
    _run(rf, rng, [_robot(0.0, 0.0, 0.0)], 3)
    feat = rf.vox_feat[rf.observed]
    assert np.allclose(feat, rf.class_emb[CLASS_ID["ground"]][None, :], atol=1e-5)


def test_repeated_looks_average_the_feature_noise_away():
    """A single noisy look shrinks the cosine; the estimate converges to the class row."""
    cfg = EnvConfig()
    cfg.rayfronts.p_confuse = 0.0
    rf, rng, _ = _sim(cfg)
    truth = float(rf.class_emb[CLASS_ID["ground"]] @ rf.query_vec("road"))
    rf.update([_robot(0.0, 0.0, 0.0)], 0.0, rng)
    one = float(np.abs(_qsim(rf, "road")[rf.observed] - truth).mean())
    for k in range(1, 30):
        rf.update([_robot(0.0, 0.0, 0.0)], float(k), rng)
    many = float(np.abs(_qsim(rf, "road")[rf.observed] - truth).mean())
    assert many < 0.5 * one and many < 0.02


def test_rays_carry_features():
    rf, rng, cfg = _sim(seed=1)
    _run(rf, rng, [_robot(0.0, 0.0, 0.3)], 6)
    st = rf.store()
    assert st.n > 0
    assert st.feat is not None and st.feat.shape == (st.n, cfg.rayfronts.embedding_dim)
    assert st.feat_peak is not None and st.feat_peak.shape == st.feat.shape
    mean = _ray_sim(rf, "road", peak=False)
    both = _ray_sim(rf, "road", peak=True)
    assert np.allclose(mean, np.clip(rf.emb.similarity(st.feat, rf.query_vec("road")[None]), 0, 1).ravel(),
                       atol=1e-5)
    assert (both >= mean - 1e-6).all()


def test_a_person_owns_its_ray_bin_over_the_background():
    from rlplanner.scene.schema import Human
    cfg = EnvConfig()
    cfg.rayfronts.p_fp_ray = 0.0
    rf, rng, cfg = _sim(cfg, _scene([Human(id="h", pos=(60.0, 0.0, 0.0), role="casualty",
                                           pose="prone")]), seed=3)
    _run(rf, rng, [_robot(0.0, 0.0, 0.0)], 10)
    st = rf.store()
    person = np.maximum(_ray_sim(rf, "person lying on the ground"), _ray_sim(rf, "person"))
    hot = np.flatnonzero(person > 0.6)
    assert hot.size >= 1
    assert min(abs(math.degrees(st.az[k])) for k in hot) < cfg.rayfronts.ray_az_bin_deg


# ---- set_queries ------------------------------------------------------------------------------
def test_set_queries_only_moves_the_query_embeddings():
    rf, rng, cfg = _sim(seed=2)
    _run(rf, rng, [_robot(0.0, 0.0, 0.5)], 5)
    obs = rf.observed.copy()
    feat = rf.vox_feat_sum.copy()
    ray_feat = rf.store().feat.copy()
    new_q = ("rubble", "person")
    rf.set_queries(new_q)
    assert rf.queries == new_q and rf.nq == 2
    assert rf.query_emb.shape == (2, rf.D)
    assert np.array_equal(feat, rf.vox_feat_sum)          # the belief itself never moves
    assert np.array_equal(ray_feat, rf.store().feat)
    view = _qsim(rf, "rubble")
    unit = rf.vox_feat[obs]
    assert np.allclose(view[obs], np.clip(unit @ rf.query_vec("rubble"), 0, 1), atol=1e-5)
    assert (view[~obs] == 0).all()


def test_set_queries_keeps_stepping_the_belief():
    rf, rng, _ = _sim(seed=2)
    _run(rf, rng, [_robot(0.0, 0.0, 0.5)], 3)
    rf.set_queries(("rubble", "person", "road"))
    _run(rf, rng, [_robot(6.0, 0.0, 0.5)], 3, t0=4.0)
    assert rf.query_emb.shape[0] == 3
    v = _qsim(rf, "rubble")
    assert (v >= 0).all() and (v <= 1).all()


def test_set_queries_with_an_unknown_query_raises_on_the_fallback():
    rf, rng, _ = _sim()
    _run(rf, rng, [_robot(0.0, 0.0, 0.0)], 2)
    with pytest.raises(ValueError, match="a fire hydrant"):
        rf.set_queries(("a fire hydrant",))
    with pytest.raises(ValueError, match="empty query list"):
        rf.set_queries(())
    assert rf.queries == tuple(DEFAULT_QUERIES)          # unchanged after the failure


def test_set_queries_from_the_cached_table():
    p = E.DATA_PATH.parent / "text_embeddings_siglip_vitb16.json"
    if not p.exists():
        pytest.skip("SigLIP cache not shipped")
    cfg = EnvConfig()
    cfg.rayfronts.embeddings_path = str(p)
    rf, rng, _ = _sim(cfg, seed=4)
    assert rf.emb.source.startswith("cache:")
    _run(rf, rng, [_robot(0.0, 0.0, 0.4)], 3)
    rf.set_queries(("rubble", "tree"))          # in the cache, no encoder needed
    assert rf.query_emb.shape[0] == 2
    feat = rf.vox_feat[rf.observed]
    view = np.stack([_qsim(rf, q) for q in rf.queries])
    assert np.allclose(view[:, rf.observed].T, np.clip(rf.emb.similarity(feat), 0, 1), atol=1e-5)


# ---- config / plumbing --------------------------------------------------------------------------
def test_feat_noise_std_aliases_vox_noise_std():
    c = RayFrontsConfig()
    assert c.feat_noise_std == c.vox_noise_std == 0.08
    c.vox_noise_std = 0.5
    assert c.feat_noise_std == 0.5
    assert RayFrontsConfig(vox_noise_std=0.02).feat_noise_std == 0.02
    assert "vox_noise_std" not in EnvConfig().to_dict()["rayfronts"]
    legacy = EnvConfig.from_dict({"rayfronts": {"vox_noise_std": 0.2, "p_confuse": 0.0}})
    assert legacy.rayfronts.feat_noise_std == 0.2 and legacy.rayfronts.p_confuse == 0.0


def test_config_roundtrip_keeps_the_embedding_fields(tmp_path):
    c = EnvConfig()
    c.rayfronts.embedding_dim = 12
    c.rayfronts.feat_noise_std = 0.03
    p = tmp_path / "c.yaml"
    c.to_yaml(p)
    c2 = EnvConfig.from_yaml(p)
    assert c2.rayfronts.embedding_dim == 12 and c2.rayfronts.vox_noise_std == 0.03
    assert c2.to_dict() == c.to_dict()
    c.rayfronts.embedding_dim = 1
    assert any("embedding_dim" in e for e in c.validate())


def test_other_embedding_dims_run_end_to_end():
    cfg = EnvConfig()
    cfg.rayfronts.embedding_dim = 12
    rf, rng, _ = _sim(cfg, seed=0)
    _run(rf, rng, [_robot(0.0, 0.0, 0.2)], 3)
    assert rf.D == 12 and rf.vox_feat_sum.shape[2] == 12
    full = E.factorized_table(TABLE_QUERIES, dim=12)
    assert np.abs(full.sim_table() - TABLE).max() < 0.05


def test_env_state_exposes_features():
    from rlplanner.scene.schema import make_synthetic_scene
    from rlplanner.sim.env import DisasterEnv
    cfg = EnvConfig()
    cfg.robot.n_robots = 1
    cfg.t_max_s = 20.0
    env = DisasterEnv(make_synthetic_scene(0, region_m=(120.0, 120.0)), cfg, seed=0)
    st = env.state
    assert st.vox_feat_sum is rf_sum(env)
    assert st.emb is env.rf.emb and st.query_names() == tuple(cfg.rayfronts.queries)
    obs, _, _, _ = env.step(np.zeros(1, np.int64))
    st = env.state
    assert st.vox_feat.shape == (env.raster.ny, env.raster.nx, cfg.rayfronts.embedding_dim)
    assert st.rays.feat is not None and st.rays.feat.shape[1] == cfg.rayfronts.embedding_dim
    assert np.isfinite(obs.tokens).all()


def rf_sum(env):
    return env.rf.vox_feat_sum


def test_determinism_of_the_feature_belief():
    outs = []
    for _ in range(2):
        rf, rng, _ = _sim(seed=17)
        rb = _robot(-40.0, -40.0, 0.4)
        for k in range(12):
            rb.pos = rb.pos + np.array([3.0, 2.0])
            rb.heading += 0.05
            rf.update([rb], float(k), rng)
        rf.end_of_decision(12.0)
        st = rf.store()
        outs.append((rf.vox_feat_sum.copy(), rf.seg_labels.copy(), st.feat.copy(),
                     st.feat_peak.copy(), st.ids.copy()))
    for a, b in zip(*outs):
        assert np.array_equal(a, b)


def test_memory_footprint_at_750_cells():
    """The per-env belief at the largest shipped region (1500 m at 2 m cells)."""
    n = 750 * 750
    d = EnvConfig().rayfronts.embedding_dim
    feat = n * d * 4
    # vox_feat_sum + vox_cnt + last_seen_t + seg_labels + observed/observable/frontier_mask
    total = feat + n * 4 + n * 4 + n * 4 + 3 * n
    assert feat / 1e6 < 80.0, feat / 1e6
    assert total / 1e6 < 90.0, total / 1e6


# ---- QA pass: exactness, degradation, query switching -------------------------------------------
EXACT_DIM = 12          # rank of the joint Gram is 15 + 11; 12 already fits the shipped table


@pytest.mark.parametrize("dim", [12, 16, 24, 64])
def test_factorization_is_exact_and_keeps_the_requested_dim(dim):
    """D must be what `rayfronts.embedding_dim` asked for even above the table's rank (26)."""
    emb = E.factorized_table(TABLE_QUERIES, dim=dim)
    assert emb.D == dim and emb.class_emb.shape == (N_CLASSES, dim)
    cos = emb.class_emb.astype(np.float64) @ emb.query_emb.astype(np.float64).T
    assert np.abs(cos - TABLE).max() < 1e-6
    assert np.allclose(np.linalg.norm(emb.class_emb, axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(emb.query_emb, axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("dim", [8, 10, 11])
def test_dims_below_the_exact_fit_threshold_are_rejected(dim):
    c, q = E.factorize_table(TABLE, dim=dim)
    assert np.abs(c @ q.T - TABLE).max() > 1e-3         # genuinely cannot fit
    with pytest.raises(ValueError, match="cannot represent"):
        E.factorized_table(TABLE_QUERIES, dim=dim)


def test_every_argmax_matches_the_hand_table():
    emb = E.get_embedding_table(TABLE_QUERIES, dim=24)
    cos = emb.class_emb.astype(np.float64) @ emb.query_emb.astype(np.float64).T
    for i, name in enumerate(CLASS_NAMES):
        assert int(np.argmax(cos[i])) == int(np.argmax(TABLE[i])), name
    for j, q in enumerate(TABLE_QUERIES):
        assert int(np.argmax(cos[:, j])) == int(np.argmax(TABLE[:, j])), q
    bg = [CLASS_ID[c] for c in CLASS_NAMES if not c.startswith("human")]
    assert cos[np.ix_(bg, [TABLE_QUERIES.index(q) for q in DEFAULT_QUERIES])].max() <= 0.3


def test_a_cache_of_another_dim_is_refused_rather_than_used_silently():
    p = E.DATA_PATH.parent / "text_embeddings_siglip_vitb16.json"
    if not p.exists():
        pytest.skip("SigLIP cache not shipped")
    with pytest.raises(ValueError, match="embedding_dim"):
        E.get_embedding_table(TABLE_QUERIES, dim=16, path=str(p))
    assert E.get_embedding_table(TABLE_QUERIES, dim=24, path=str(p)).D == 24


@pytest.mark.parametrize("std", [0.0, 0.08, 0.5])
@pytest.mark.parametrize("p_confuse", [0.0, 1.0])
def test_noise_and_confusion_degrade_gracefully(std, p_confuse):
    cfg = EnvConfig()
    cfg.rayfronts.feat_noise_std = std
    cfg.rayfronts.p_confuse = p_confuse
    rf, rng, _ = _sim(cfg, seed=5)
    _run(rf, rng, [_robot(0.0, 0.0, 0.3)], 6)
    obs = rf.observed
    assert obs.any()
    assert np.isfinite(rf.vox_feat_sum).all()
    assert np.allclose(np.linalg.norm(rf.vox_feat[obs], axis=1), 1.0, atol=1e-5)
    v = _qsim(rf, "road")
    assert (v >= 0.0).all() and (v <= 1.0).all()
    st = rf.store()
    assert np.isfinite(st.feat).all() and np.isfinite(st.feat_peak).all()
    assert (_ray_sim(rf, "road") >= _ray_sim(rf, "road", peak=False) - 1e-6).all()


@pytest.mark.parametrize("std", [0.08, 0.5])
def test_heavy_noise_still_averages_toward_the_class_row(std):
    """Even at 0.5 per dimension the estimate converges toward the hand table with more looks."""
    cfg = EnvConfig()
    cfg.rayfronts.feat_noise_std = std
    cfg.rayfronts.p_confuse = 0.0
    rf, rng, _ = _sim(cfg, seed=1)
    truth = float(rf.class_emb[CLASS_ID["ground"]] @ rf.query_vec("road"))
    rb = _robot(0.0, 0.0, 0.0)
    rf.update([rb], 0.0, rng)
    one = float(np.abs(_qsim(rf, "road")[rf.observed] - truth).mean())
    for k in range(1, 60):
        rf.update([rb], float(k), rng)
    many = float(np.abs(_qsim(rf, "road")[rf.observed] - truth).mean())
    assert many < 0.6 * one and many < 0.10, (one, many)


def test_a_cell_seen_once_and_a_cell_seen_1000_times_are_both_unit():
    from rlplanner.sim.rayfronts_sim import _vox_scatter
    emb = E.get_embedding_table(TABLE_QUERIES, dim=24)
    rng = np.random.default_rng(0)
    fs = np.zeros((1, 2, emb.D), np.float32)
    cnt = np.zeros((1, 2), np.int32)
    seen = np.full((1, 2), -1.0, np.float32)
    ij = np.array([[0, 0]], np.int32)
    rows = np.array([CLASS_ID["debris"]], np.int64)
    for k in range(1000):
        noise = rng.standard_normal((1, emb.D), np.float32) * np.float32(0.5)
        _vox_scatter(ij, rows, emb.class_emb, noise, fs, cnt, seen, np.float32(k))
        if k == 0:
            once = np.linalg.norm(fs[0, 0] / np.linalg.norm(fs[0, 0]))
    assert int(cnt[0, 0]) == 1000 and once == pytest.approx(1.0, abs=1e-6)
    f = fs[0, 0] / np.linalg.norm(fs[0, 0])
    assert np.linalg.norm(f) == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(fs).all()
    # 1000 looks recover the class row (|n| ~ 2.4 at std 0.5, so the mean still scatters ~0.05)
    assert np.abs(np.clip(f @ emb.query_emb.T, 0, 1) - np.clip(TABLE[CLASS_ID["debris"]], 0, 1)).max() < 0.08
    assert (fs[0, 1] == 0).all()                                    # the untouched cell stays zero


def test_a_cancelled_feature_sum_gives_zeros_not_nan():
    """Nothing may divide by a zero-length accumulated feature."""
    from rlplanner.sim.state import EnvState
    cfg = EnvConfig()
    st = EnvState(t=0.0, decision_idx=0, scene=None, raster=None, cfg=cfg, robots=[],
                  observed=np.ones((2, 2), bool), vox_cnt=np.ones((2, 2), np.int32),
                  last_seen_t=np.zeros((2, 2), np.float32), rays=None,
                  ray_targets=[], frontier_mask=np.zeros((2, 2), bool), frontier_clusters=[],
                  segments=[], seg_labels=np.full((2, 2), -1, np.int32),
                  human_hits=np.zeros(0, np.int32),
                  human_found=np.zeros(0, bool), last_obs=None, last_actions=None, cum_reward=0.0,
                  events=[], metrics={}, vox_feat_sum=np.zeros((2, 2, 24), np.float32))
    f = st.vox_feat
    assert f.shape == (2, 2, 24) and np.isfinite(f).all() and (f == 0).all()
    st.vox_feat_sum = None
    with pytest.raises(AttributeError, match="vox_feat_sum"):
        st.vox_feat


# ---- switching queries on a live env ------------------------------------------------------------
def _env(n_robots=2, region=(160.0, 160.0), seed=0, decisions=3, cfg=None):
    from rlplanner.scene.schema import make_synthetic_scene
    from rlplanner.sim.env import DisasterEnv
    cfg = cfg or EnvConfig()
    cfg.robot.n_robots = n_robots
    cfg.t_max_s = 120.0
    env = DisasterEnv(make_synthetic_scene(0, region_m=region), cfg, seed=seed)
    for _ in range(decisions):
        env.step(np.zeros(n_robots, np.int64))
    return env


MANY = ("road", "tree", "house", "person", "person lying on the ground", "collapsed building",
        "damaged building", "rubble")


@pytest.mark.parametrize("names", [("rubble", "person"),                       # fewer
                                   ("person", "person", "road"),               # duplicated
                                   MANY,                                       # eight = Qmax
                                   DEFAULT_QUERIES + ("house",)])              # more
def test_env_set_queries_moves_only_the_query_block(names):
    from rlplanner.sim.state import TOKEN_FIXED
    from rlplanner.sim.tokens import BEV_CHANNELS
    env = _env()
    width = env.state.last_obs.tokens.shape[2]
    feat = env.rf.vox_feat_sum.copy()
    obs = env.set_queries(names)
    nq = len(names)
    assert obs.tokens.shape == (env.n_robots, env.k_tokens, TOKEN_FIXED + env.rf.D) 
    assert obs.tokens.shape[2] == width          # the width never depends on the mission
    assert int(obs.query_mask.sum()) == nq
    assert env.state.query_names() == tuple(names)
    assert np.array_equal(feat, env.rf.vox_feat_sum)
    # and stepping continues
    obs2, _, _, _ = env.step(np.zeros(env.n_robots, np.int64))
    assert obs2.tokens.shape == obs.tokens.shape and np.isfinite(obs2.tokens).all()
    assert obs2.bev.shape == (len(BEV_CHANNELS), env.cfg.tokens.bev_size, env.cfg.tokens.bev_size)
    assert env.rf.segments


def test_query_weights_reach_the_observation():
    """The weight column is where an LLM-proposed hint will carry its likelihood."""
    env = _env(decisions=1)
    obs = env.set_queries(("person", "rubble"), weights=[1.0, 0.25])
    assert obs.query_w[:2].tolist() == [1.0, pytest.approx(0.25)]
    assert obs.query_w[2:].max() == 0.0


def test_a_belief_query_switch_needs_no_token_rebuild():
    """`RayFrontsSim.set_queries` alone used to break the token builder; nothing indexes by query
    any more, so the belief and the builder cannot disagree."""
    env = _env(decisions=1)
    env.rf.set_queries(("rubble", "person"))
    obs, _, _, _ = env.step(np.zeros(env.n_robots, np.int64))
    assert np.isfinite(obs.tokens).all()


def test_a_query_set_without_a_person_query_is_allowed():
    """Nothing in the belief privileges the person queries: the candidate order is generic."""
    env = _env(decisions=1)
    env.set_queries(("road", "tree"))
    obs, _, _, _ = env.step(np.zeros(env.n_robots, np.int64))
    assert env.state.query_names() == ("road", "tree")
    assert np.isfinite(obs.tokens).all()


def test_a_config_without_a_person_query_runs():
    from rlplanner.scene.schema import make_synthetic_scene
    from rlplanner.sim.env import DisasterEnv
    cfg = EnvConfig()
    cfg.rayfronts.queries = ("road", "tree")
    cfg.robot.n_robots = 1
    env = DisasterEnv(make_synthetic_scene(0, region_m=(80.0, 80.0)), cfg, seed=0)
    obs, _, _, _ = env.step(np.zeros(1, np.int64))
    assert np.isfinite(obs.tokens).all() and np.isfinite(obs.bev).all()


def test_env_pickles_its_state_and_two_seeds_agree():
    import pickle
    a, b = _env(seed=3), _env(seed=3)
    for name in ("vox_feat_sum", "seg_labels"):
        assert np.array_equal(getattr(a.state, name), getattr(b.state, name)), name
    assert np.array_equal(a.state.rays.feat, b.state.rays.feat)
    assert np.array_equal(a.state.rays.feat_peak, b.state.rays.feat_peak)
    assert np.array_equal(a.state.last_obs.tokens, b.state.last_obs.tokens)
    st = pickle.loads(pickle.dumps(a.state))           # the train side ships obs across processes
    assert np.array_equal(st.vox_feat_sum, a.state.vox_feat_sum)
    assert st.query_names() == a.state.query_names()
    assert np.allclose(st.emb.class_emb, a.state.emb.class_emb)
    ob = pickle.loads(pickle.dumps(a.state.last_obs))
    assert np.array_equal(ob.tokens, a.state.last_obs.tokens)


def test_the_siglip_cache_runs_but_its_cosines_are_compressed():
    """Characterisation, not a target: the shipped SigLIP cosines top out around 0.56 on the person
    queries, so a threshold calibrated on the hand table does not transfer."""
    p = E.DATA_PATH.parent / "text_embeddings_siglip_vitb16.json"
    if not p.exists():
        pytest.skip("SigLIP cache not shipped")
    cfg = EnvConfig()
    cfg.rayfronts.embeddings_path = str(p)
    cfg.robot.n_robots = 1
    env = _env(n_robots=1, region=(120.0, 120.0), decisions=3, cfg=cfg)
    st = env.state
    assert env.rf.emb.source.startswith("cache:")
    assert np.isfinite(st.last_obs.tokens).all() and np.isfinite(st.last_obs.bev).all()
    cos = env.rf.emb.class_emb @ env.rf.emb.query_emb.T
    assert cos.max() < 0.6                                     # the person queries top out here
    assert cos.min() < 0.0                                     # negative cosines clip to 0 in a view


# ---- the fixed PCA basis the rasters project onto -----------------------------------------------
def test_pc_basis_is_fixed_orthonormal_and_shared():
    a = E.get_embedding_table(TABLE_QUERIES, dim=24)
    b = E.factorized_table(TABLE_QUERIES, dim=24)
    ma, ca = a.pc_basis(E.FEAT_PC_DIM)
    mb, cb = b.pc_basis(E.FEAT_PC_DIM)
    assert ca.shape == (E.FEAT_PC_DIM, a.D)
    assert np.allclose(ca @ ca.T, np.eye(E.FEAT_PC_DIM), atol=1e-4)
    assert np.allclose(ma, mb) and np.allclose(ca, cb)      # identical across envs by construction
    assert np.allclose(a.pc_basis(E.FEAT_PC_DIM)[1], ca)    # and stable across calls
    p = a.project(a.class_emb)
    assert p.shape == (N_CLASSES, E.FEAT_PC_DIM) and np.isfinite(p).all()
    assert np.abs(p).max() < 2.0

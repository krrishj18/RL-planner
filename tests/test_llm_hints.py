import json

import numpy as np
import pytest

from rlplanner.llm.digest import DIGEST_MAX_CHARS, DigestBuilder, build_digest, nearest_class
from rlplanner.llm.embed import QueryEmbedder
from rlplanner.llm.hint_agent import (BackendError, ClaudeBackend, HintAgent, HintController,
                                      QueryEdits, ScriptedBackend, extract_json, make_backend)
from rlplanner.llm.schedule import NOISE_TAG, QueryScheduleSampler, default_pool
from rlplanner.scene.schema import CLASS_NAMES, make_synthetic_scene
from rlplanner.sim.baselines import make_policy
from rlplanner.sim.config import EnvConfig
from rlplanner.sim.embeddings import CLASS_PROMPTS, get_embedding_table
from rlplanner.sim.env import DisasterEnv

OBS_UNCHANGED = ("tokens", "token_mask", "token_xy", "token_type", "token_id", "robot_feat", "bev")


def _same(a, b) -> bool:
    """`token_xy` pads with nan, so equality has to treat nan as equal."""
    a, b = np.asarray(a), np.asarray(b)
    return np.array_equal(a, b, equal_nan=np.issubdtype(a.dtype, np.floating))


def _env(robots=2, region=(140.0, 140.0), t_max=150.0, seed=0, **kw):
    cfg = EnvConfig()
    cfg.robot.n_robots = robots
    cfg.t_max_s = t_max
    for k, v in kw.items():
        setattr(cfg.queries_dynamic, k, v)
    return DisasterEnv(make_synthetic_scene(0, region_m=region), cfg, seed=seed)


def _table(queries=("person lying on the ground", "person")):
    return get_embedding_table(queries, dim=24)


def _warm(env, n=12):
    pol = make_policy("ray_follower")
    obs = env.state.last_obs
    for _ in range(n):
        obs, _, done, _ = env.step(pol.act(obs, env.state))
        if done:
            break
    return obs


# ---- embedding ---------------------------------------------------------------------------------
def test_projected_class_name_lands_on_its_own_class():
    """The fitted SigLIP -> sim map must put every known class prompt nearest its own class."""
    qe = QueryEmbedder.build(_table())
    if qe.W is None:
        pytest.skip("no SigLIP cache in this checkout")
    assert qe.class_roundtrip() == {c: c for c in CLASS_NAMES}


def test_lexicon_entries_decode_to_themselves():
    """Every lexicon vector is a legal query vector, and a class prompt decodes to its class."""
    emb = _table()
    qe = QueryEmbedder.build(emb)
    for i, c in enumerate(CLASS_NAMES):
        v, how = qe.embed(CLASS_PROMPTS[c])
        assert how.startswith("exact")
        assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-5)
        assert int(np.argmax(v @ emb.class_emb.T)) == i


def test_a_word_outside_the_vocabulary_is_rejected_not_guessed():
    qe = QueryEmbedder.build(_table())
    v, how = qe.embed("a fire hydrant")
    assert v is None and how == "unmatched"
    assert qe.register("a fire hydrant") == (None, "unmatched")


def test_a_synonym_resolves_and_becomes_a_legal_query():
    """A hint in the LLM's own wording is registered in the bank, which is what set_queries reads."""
    env = _env()
    qe = QueryEmbedder.for_env(env)
    name, how = qe.register("crushed vehicle")
    assert name == "crushed vehicle" and how.startswith("fuzzy")
    obs = env.set_queries(("person", "crushed vehicle"))
    assert env.state.query_names() == ("person", "crushed vehicle")
    assert int(obs.query_mask.sum()) == 2
    car = env.rf.emb.class_emb[CLASS_NAMES.index("vehicle_toppled")]
    assert float(obs.query_emb[1] @ car) > 0.8


# ---- digest ------------------------------------------------------------------------------------
def test_digest_stays_under_its_cap_on_a_busy_scene():
    env = _env(robots=3, region=(300.0, 300.0), t_max=400.0)
    _warm(env, 40)
    assert env.state.segments and env.state.ray_targets      # the scene really is busy
    for cap in (DIGEST_MAX_CHARS, 400, 120):
        d = build_digest(env.state, since_t=0.0, max_chars=cap)
        assert len(d) <= cap, (cap, len(d))
    d = build_digest(env.state, since_t=0.0)
    assert "queries:" in d and "casualties found:" in d and "new semantic rays" in d


def test_digest_reports_only_the_new_interval():
    env = _env()
    db = DigestBuilder()
    _warm(env, 6)
    first = db.build(env)
    assert "since t=0" in first
    _warm(env, 4)
    second = db.build(env)
    assert f"since t={env.cfg.decision_dt * 6:.0f}" in second


def test_nearest_class_decode_is_a_view_not_an_input():
    """The decode is text for the LLM; nothing in it touches the observation."""
    env = _env()
    obs = _warm(env, 6)
    before = {k: getattr(obs, k).copy() for k in OBS_UNCHANGED}
    build_digest(env.state, since_t=0.0)
    for k, v in before.items():
        assert _same(v, getattr(env.state.last_obs, k)), k
    k, cos = nearest_class(env.rf.emb.class_emb, env.rf.emb)
    assert k.tolist() == list(range(len(CLASS_NAMES)))
    assert cos.min() > 0.99


# ---- backends ----------------------------------------------------------------------------------
def test_scripted_backend_is_deterministic():
    a, b = ScriptedBackend(), ScriptedBackend()
    digests = ["nothing here", "vehicle_toppled x3", "more", "and more"]
    assert [a.update(d) for d in digests] == [b.update(d) for d in digests]


def test_claude_backend_parses_the_cli_envelope():
    envelope = json.dumps({"is_error": False,
                           "result": '```json\n{"add": [{"text": "rubble", "weight": 0.4}], '
                                     '"remove": [], "reweight": {}}\n```'})
    b = ClaudeBackend(runner=lambda argv, t: envelope)
    d = b.update("digest", 8, ())
    assert QueryEdits.parse(d).add == [("rubble", 0.4)]
    argv = b.argv("hello")
    assert argv[:3] == ["claude", "-p", "hello"] and "--output-format" in argv
    assert argv[argv.index("--model") + 1] == "opus"


def test_claude_backend_retries_once_on_malformed_output():
    replies = ["I think we should look for rubble.", '{"add": [], "remove": ["person"]}']
    calls = []

    def runner(argv, timeout):
        calls.append(argv)
        return replies[len(calls) - 1] if len(calls) <= len(replies) else replies[-1]

    b = ClaudeBackend(runner=runner, output_format="text")
    assert b.update("digest", 8, ())["remove"] == ["person"]
    assert len(calls) == 2
    assert "not usable" in calls[1][2]


def test_claude_backend_gives_up_after_the_retry():
    b = ClaudeBackend(runner=lambda argv, t: "no json here at all", output_format="text")
    with pytest.raises(BackendError):
        b.update("digest", 8, ())


def test_cli_error_envelope_becomes_a_backend_error():
    env = json.dumps({"is_error": True, "result": "credit balance too low"})
    b = ClaudeBackend(runner=lambda argv, t: env)
    with pytest.raises(BackendError):
        b.update("digest", 8, ())


def test_make_backend_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown backend"):
        make_backend("gpt")


@pytest.mark.parametrize("text,expect", [
    ('{"add": []}', {"add": []}),
    ('```json\n{"add": []}\n```', {"add": []}),
    ('Sure!\n{"add": [], "remove": ["x"]}\nHope that helps', {"add": [], "remove": ["x"]}),
])
def test_extract_json_survives_the_usual_wrappers(text, expect):
    assert extract_json(text) == expect


# ---- agent -------------------------------------------------------------------------------------
class _Broken:
    """Every method fails, the way a missing CLI or a timeout does."""
    name = "broken"

    def initial(self, context, cap, vocab):
        raise BackendError("boom")

    def update(self, digest, cap, vocab):
        raise BackendError("boom")


def test_backend_failure_gives_noop_edits_and_the_mission_defaults():
    ag = HintAgent(_Broken(), embedder=QueryEmbedder.build(_table()))
    ag.reset()
    assert ag.initial_queries("ctx") == [("person lying on the ground", 1.0), ("person", 1.0)]
    e = ag.update("digest")
    assert e.is_noop and e.note == "backend failure"
    assert any("boom" in w for w in ag.warnings)


def test_a_failing_backend_never_stops_the_episode():
    env = _env()
    qe = QueryEmbedder.for_env(env)
    ctl = HintController(HintAgent(_Broken(), embedder=qe), qe, every=2, condition="broken")
    ctl.agent.reset()
    ctl.start(env)
    pol = make_policy("ray_follower")
    obs = env.state.last_obs
    for _ in range(10):
        obs, _, done, info = env.step(pol.act(obs, env.state))
        obs = ctl.after_step(env, info) or obs
        if done:
            break
    assert np.isfinite(obs.tokens).all()
    assert all(rec["add"] == [] and rec["remove"] == [] for rec in ctl.log[1:])
    assert env.state.query_names() == ("person lying on the ground", "person")


def test_the_agent_never_exceeds_the_query_token_capacity():
    ag = HintAgent(ScriptedBackend(initial=[(q, 1.0) for q in
                                            ("person", "rubble", "car", "tree", "road")]),
                   embedder=QueryEmbedder.build(_table()), max_queries=3)
    ag.reset()
    assert len(ag.initial_queries("ctx")) == 3
    ag.apply(QueryEdits(add=[("house", 0.9), ("bus stop", 0.8)]))
    assert len(ag.active) == 3 and len(ag.warnings) >= 1


def test_edits_against_a_query_that_is_not_active_are_dropped():
    ag = HintAgent("scripted", embedder=QueryEmbedder.build(_table()))
    ag.reset([("person", 1.0)])
    e = ag._validate(QueryEdits(add=[("person", 0.5)], remove=["tree"], reweight={"road": 0.2}))
    assert e.is_noop


# ---- closed loop -------------------------------------------------------------------------------
def test_scripted_hint_loop_runs_a_full_episode_and_finalises():
    env = _env(robots=3, region=(200.0, 200.0), t_max=200.0)
    qe = QueryEmbedder.for_env(env)
    ag = HintAgent("scripted", embedder=qe, max_queries=env.cfg.tokens.max_queries)
    ag.reset()
    ctl = HintController(ag, qe, every=4, condition="scripted")
    ctl.start(env)
    pol = make_policy("ray_follower")
    obs, done, info, n = env.state.last_obs, False, {}, 0
    while not done:
        obs, _, done, info = env.step(pol.act(obs, env.state))
        obs = ctl.after_step(env, info) or obs
        n += 1
        assert np.isfinite(obs.tokens).all() and np.isfinite(obs.bev).all()
    m = info["metrics"]
    assert m["n_decisions"] == n and m["time_to_first"] > 0
    assert 0.0 <= m["frac_found"] <= 1.0 and m["coverage_end"] > 0.0
    changed = [r for r in ctl.log if r["add"] or r["remove"] or r["reweight"]]
    assert len(changed) >= 2, [r["kind"] for r in ctl.log]      # the add and the removal both fired
    assert "crushed vehicle" in env.state.query_names()
    assert "person" not in env.state.query_names()              # dropped mid-episode


def test_the_controller_logs_every_turn_with_the_resulting_list():
    env = _env()
    qe = QueryEmbedder.for_env(env)
    ctl = HintController(HintAgent("scripted", embedder=qe), qe, every=2, condition="scripted")
    ctl.agent.reset()
    ctl.start(env)
    obs = env.state.last_obs
    for _ in range(6):
        obs, _, done, info = env.step(np.zeros(env.n_robots, np.int64))
        ctl.after_step(env, info)
    assert ctl.log[0]["kind"] == "initial"
    for rec in ctl.log:
        assert rec["queries"] == list(env.state.query_names()) or rec is not ctl.log[-1]
        assert set(rec) >= {"condition", "decision", "t", "backend", "add", "remove", "reweight"}
        json.dumps(rec)                                        # the log must be JSONL-serialisable


# ---- the training-side sampler ------------------------------------------------------------------
def test_sampler_is_deterministic_per_seed():
    emb = _table()
    def run(seed):
        s = QueryScheduleSampler(pool=default_pool(emb), emb=emb, every=2, p_edit=1.0,
                                 n_init=(1, 3), noise_std=0.05)
        rng = np.random.default_rng(seed)
        names, w = s.initial(rng)
        out = [(list(names), list(w))]
        for d in range(1, 12):
            n2, w2 = s.edit(names, w, d, rng)
            if n2 is not None:
                names, w = n2, w2
            out.append((list(names), list(w)))
        return out
    assert run(7) == run(7)
    assert run(7) != run(8)


def test_a_noised_draw_is_a_new_name_with_a_nearby_vector():
    emb = _table()
    s = QueryScheduleSampler(pool=("person",), emb=emb, noise_std=0.1)
    n = s._draw(np.random.default_rng(0), set())
    assert n.startswith("person" + NOISE_TAG) and n in emb.bank
    assert float(emb.bank[n] @ emb.bank["person"]) > 0.8
    assert float(np.linalg.norm(emb.bank[n])) == pytest.approx(1.0, abs=1e-5)


def test_default_pool_excludes_earlier_noised_draws():
    emb = _table()
    s = QueryScheduleSampler(pool=("person",), emb=emb, noise_std=0.1)
    s._draw(np.random.default_rng(1), set())
    assert not any(NOISE_TAG in q for q in default_pool(emb))


def test_sampler_stress_leaves_no_stale_token_or_bev_state():
    """Many add/remove cycles through the env's own set_queries: only the query block may move."""
    env = _env(robots=3)
    obs0 = _warm(env, 10)
    base = {k: getattr(obs0, k).copy() for k in OBS_UNCHANGED}
    feat0 = env.rf.vox_feat_sum.copy()
    emb = env.rf.emb
    s = QueryScheduleSampler(pool=default_pool(emb), emb=emb, every=1, p_edit=1.0, n_init=(1, 4),
                             noise_std=0.05, max_queries=env.cfg.tokens.max_queries)
    rng = np.random.default_rng(3)
    names, w = s.initial(rng)
    obs = env.set_queries(names, w)
    for d in range(1, 60):
        n2, w2 = s.edit(names, w, d, rng)
        if n2 is None:
            continue
        names, w = n2, w2
        obs = env.set_queries(names, w)
        assert int(obs.query_mask.sum()) == len(names)
        assert obs.query_w[: len(names)] == pytest.approx(np.asarray(w, np.float32), abs=1e-6)
        assert obs.query_mask[len(names):].sum() == 0
        assert obs.query_emb[len(names):].max(initial=0.0) == 0.0
        for k, v in base.items():
            assert _same(v, getattr(obs, k)), k
    assert np.array_equal(feat0, env.rf.vox_feat_sum)
    obs, _, _, _ = env.step(np.zeros(env.n_robots, np.int64))   # and it still steps
    assert np.isfinite(obs.tokens).all()


# ---- env wiring --------------------------------------------------------------------------------
def test_queries_dynamic_is_off_by_default_and_round_trips(tmp_path):
    c = EnvConfig()
    assert c.queries_dynamic.enabled is False and c.validate() == []
    p = tmp_path / "c.yaml"
    c.to_yaml(p)
    assert EnvConfig.from_yaml(p).to_dict() == c.to_dict()
    assert isinstance(EnvConfig.from_dict(c.to_dict()).queries_dynamic.pool, tuple)


def test_the_default_config_reproduces_exactly_with_the_block_present():
    a = _env(seed=5)
    b = _env(seed=5)
    for _ in range(8):
        oa, ra, _, _ = a.step(np.zeros(a.n_robots, np.int64))
        ob, rb, _, _ = b.step(np.zeros(b.n_robots, np.int64))
    assert ra == rb and np.array_equal(oa.tokens, ob.tokens)
    assert a.state.query_names() == tuple(EnvConfig().rayfronts.queries)


def test_the_env_applies_the_schedule_at_decision_boundaries():
    env = _env(enabled=True, every=3, p_edit=1.0, n_init_min=2, n_init_max=2, seed=2)
    assert len(env.state.query_names()) == 2
    seen = {env.state.query_names()}
    for _ in range(10):
        obs, _, done, _ = env.step(np.zeros(env.n_robots, np.int64))
        seen.add(env.state.query_names())
        assert int(obs.query_mask.sum()) == len(env.state.query_names())
        assert np.isfinite(obs.tokens).all()
        if done:
            break
    assert len(seen) > 1


def test_the_schedule_is_reproducible_for_one_seed():
    kw = dict(enabled=True, every=2, p_edit=1.0, noise_std=0.05)
    a, b = _env(seed=11, **kw), _env(seed=11, **kw)
    for _ in range(8):
        a.step(np.zeros(a.n_robots, np.int64))
        b.step(np.zeros(b.n_robots, np.int64))
        assert a.state.query_names() == b.state.query_names()
        assert np.allclose(a.rf.query_w, b.rf.query_w)


def test_an_invalid_schedule_block_is_rejected():
    c = EnvConfig()
    c.queries_dynamic.enabled = True
    c.queries_dynamic.n_init_max = c.tokens.max_queries + 1
    c.queries_dynamic.w_min = 0.0
    errs = c.validate()
    assert any("n_init_max" in e for e in errs) and any("w_min" in e for e in errs)

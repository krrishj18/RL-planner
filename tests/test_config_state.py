import numpy as np
import pytest

from rlplanner.sim.config import EnvConfig, DEFAULT_QUERIES
from rlplanner.sim import state as ST


def test_default_config_valid_and_roundtrip(tmp_path):
    c = EnvConfig()
    assert c.validate() == []
    assert c.k_tokens == 1 + 32 + 32 + 32 + 32       # hold + frontier/ray/segment/visited
    assert c.substeps_per_decision == 5
    p = tmp_path / "c.yaml"
    c.to_yaml(p)
    c2 = EnvConfig.from_yaml(p)
    assert c2.to_dict() == c.to_dict()
    assert isinstance(c2.rayfronts.queries, tuple)


def test_config_validation_errors():
    c = EnvConfig()
    c.sensor.depth_limit_m = 100.0
    c.sensor.visual_range_m = 50.0
    c.robot.n_robots = 11
    c.decision_dt = 0.5
    c.sensor.mode = "sphere"
    c.comms = "mesh"        # a string is coerced to CommsConfig(mode=...); "range" is legal now
    errs = c.validate()
    for key in ("depth_limit", "n_robots", "decision_dt", "sensor.mode", "comms"):
        assert any(key in e for e in errs), (key, errs)


def test_too_many_mission_queries_is_rejected():
    c = EnvConfig()
    c.rayfronts.queries = tuple(f"q{i}" for i in range(c.tokens.max_queries + 1))
    assert any("max_queries" in e for e in c.validate())
    c.rayfronts.queries = ()
    assert any("empty" in e for e in c.validate())


def test_partial_yaml_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("robot:\n  n_robots: 5\nrayfronts:\n  queries: [person, car]\n")
    c = EnvConfig.from_yaml(p)
    assert c.robot.n_robots == 5 and c.rayfronts.queries == ("person", "car")
    from rlplanner.sim.config import SensorConfig
    assert c.sensor.depth_limit_m == SensorConfig().depth_limit_m  # untouched defaults survive


def test_default_queries_are_the_mission_not_a_scan_list():
    """The default query list is what the episode is looking for, nothing else: it is an input to
    the policy, so it must not read as a taxonomy of the map."""
    assert DEFAULT_QUERIES == ("person lying on the ground", "person")
    assert len(DEFAULT_QUERIES) <= EnvConfig().tokens.max_queries


def test_token_feature_names_length():
    names = ST.token_feature_names(24)
    assert len(names) == ST.TOKEN_FIXED + 24
    assert names[:5] == ["type_hold", "type_frontier", "type_ray", "type_segment", "type_visited"]
    assert names[ST.F_FEAT0] == "feat0" and names[-1] == "feat23"
    assert "reachable" in names and not any(n.startswith("sim:") for n in names)
    assert len(set(names)) == len(names)


def test_raystore_live_mask():
    n = 4
    rs = ST.RayStore(origin_xy=np.zeros((n, 2)), az=np.zeros(n), el=np.zeros(n),
                     conf=np.ones(n, np.float32), n_obs=np.ones(n, np.int32), t_first=np.zeros(n),
                     t_last=np.zeros(n), ids=np.arange(n, dtype=np.int32),
                     resolved=np.array([False, True, False, True]))
    assert rs.n == 4 and rs.live().sum() == 2
    assert not hasattr(rs, "sims"), "the ray store must not carry a per-query column any more"

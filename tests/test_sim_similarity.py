import json

import numpy as np
import pytest

from rlplanner.scene.schema import CLASS_ID, CLASS_NAMES, N_CLASSES
from rlplanner.sim.similarity_table import (BASE_TABLE, Q, build_sim_table, load_sim_table,
                                            query_index)

# The hand-authored table is the *source of the embeddings*, not something the simulator scans;
# these tests check the table itself, so they use its own full query list.
DEFAULT_QUERIES = Q
PERSON_QUERY_IDX = (0, 1)

T = build_sim_table(DEFAULT_QUERIES)


def test_shape_and_range():
    assert T.shape == (N_CLASSES, len(DEFAULT_QUERIES)) and T.dtype == np.float32
    assert (T >= 0).all() and (T <= 1).all()
    assert set(BASE_TABLE) == set(CLASS_NAMES)
    assert all(set(row) == set(Q) for row in BASE_TABLE.values())


def test_background_classes_score_low_on_person_queries():
    bg = [c for c in CLASS_NAMES if not c.startswith("human")]
    for c in bg:
        for q in PERSON_QUERY_IDX:
            assert T[CLASS_ID[c], q] <= 0.2, (c, DEFAULT_QUERIES[q])


def test_contract_confusions():
    q = {name: i for i, name in enumerate(DEFAULT_QUERIES)}
    assert T[CLASS_ID["debris"], q["collapsed building"]] == pytest.approx(0.5)
    assert T[CLASS_ID["human_standing"], q["person lying on the ground"]] == pytest.approx(0.4)
    assert T[CLASS_ID["vehicle_toppled"], q["car"]] == pytest.approx(0.7)
    assert T[CLASS_ID["human_prone"], q["person lying on the ground"]] > 0.8
    assert T[CLASS_ID["building_destroyed"], q["collapsed building"]] > \
        T[CLASS_ID["building_damaged"], q["collapsed building"]]


def test_query_order_is_respected():
    t = build_sim_table(("person", "road"))
    assert t.shape == (N_CLASSES, 2)
    assert np.allclose(t[:, 0], T[:, DEFAULT_QUERIES.index("person")])
    assert np.allclose(t[:, 1], T[:, DEFAULT_QUERIES.index("road")])


def test_unknown_query_raises():
    with pytest.raises(ValueError, match="unknown queries"):
        build_sim_table(("person", "flying saucer"))
    with pytest.raises(ValueError):
        build_sim_table(())


def test_helpers():
    assert query_index(DEFAULT_QUERIES, "rubble") == 4
    assert query_index(DEFAULT_QUERIES, "nope") == -1


def test_no_person_query_index_helper_survives():
    """Nothing may look up "the person column" any more: the mission queries are policy input."""
    import rlplanner.sim.similarity_table as st
    import rlplanner.sim.config as cfgmod
    assert not hasattr(st, "person_query_indices") and not hasattr(st, "PERSON_QUERY_NAMES")
    assert not hasattr(cfgmod, "PERSON_QUERY_IDX")


def test_load_sim_table_values_form(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"queries": list(DEFAULT_QUERIES), "values": T.tolist()}))
    assert np.allclose(load_sim_table(p), T)
    assert np.allclose(load_sim_table(p, ("road", "person")),
                       T[:, [DEFAULT_QUERIES.index("road"), DEFAULT_QUERIES.index("person")]])
    with pytest.raises(ValueError):
        load_sim_table(p, ("nope",))


def test_load_sim_table_dict_form_and_validation(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"queries": ["person"], "table": {c: {"person": 0.5} for c in CLASS_NAMES}}))
    assert np.allclose(load_sim_table(p), 0.5)
    p.write_text(json.dumps({"queries": ["person"], "table": {"road": {"person": 0.5}}}))
    with pytest.raises(ValueError, match="missing class row"):
        load_sim_table(p)
    p.write_text(json.dumps({"queries": list(DEFAULT_QUERIES), "values": (T * 3).tolist()}))
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        load_sim_table(p)
    p.write_text(json.dumps({"queries": ["person"], "values": T.tolist()}))
    with pytest.raises(ValueError, match="values shape"):
        load_sim_table(p)

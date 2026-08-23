"""Training: token policy, MAPPO/PPO, scene sampling, evaluation.

Attributes are resolved lazily so `rlplanner.train.par_env` (imported by every env worker
process) does not drag torch into the workers.
"""
from typing import Any

_LAZY = {
    "ObsBatch": "obs", "TokenPolicy": "policy", "PPO": "ppo", "PPOConfig": "ppo",
    "Collector": "rollout", "EnvPool": "rollout", "RolloutBatch": "rollout", "gae": "rollout",
    "SceneBank": "scenes", "SceneKey": "scenes", "parse_scenes": "scenes",
    "parse_robots": "scenes", "parse_t_max": "scenes", "parse_scene_mix": "scenes",
    "auto_robots": "scenes", "auto_t_max": "scenes", "AUTO": "scenes", "BUCKETS": "scenes",
    "SubprocVecEnv": "par_env", "SerialVecEnv": "par_env",
    "make_vec_env": "par_env", "default_workers": "par_env",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(f".{mod}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)

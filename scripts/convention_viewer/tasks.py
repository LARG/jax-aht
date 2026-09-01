"""Task registry shared by the convention viewer scripts."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    slug: str            # short id used in file/dir names and the site URLs
    label: str           # human-readable name shown in the viewer
    env: str             # 'lbf' or 'overcooked'
    variant: str         # lbf/overcooked layout variant
    yaml_key: str        # key into heldout_set / best_response_set

    @property
    def env_name(self) -> str:
        return "lbf" if self.env == "lbf" else "overcooked-v1"

    @property
    def max_steps(self) -> int:
        return 128 if self.env == "lbf" else 400


TASKS = {
    t.slug: t
    for t in [
        TaskSpec("lbf_7x7", "Level-Based Foraging (7x7)", "lbf", "lbf_7x7_nolevels", "lbf/lbf_7x7_nolevels"),
        TaskSpec("cramped_room", "Overcooked: Cramped Room", "overcooked", "cramped_room", "overcooked-v1/cramped_room"),
        TaskSpec("asymm_advantages", "Overcooked: Asymmetric Advantages", "overcooked", "asymm_advantages", "overcooked-v1/asymm_advantages"),
        TaskSpec("counter_circuit", "Overcooked: Counter Circuit", "overcooked", "counter_circuit", "overcooked-v1/counter_circuit"),
        TaskSpec("coord_ring", "Overcooked: Coordination Ring", "overcooked", "coord_ring", "overcooked-v1/coord_ring"),
        TaskSpec("forced_coord", "Overcooked: Forced Coordination", "overcooked", "forced_coord", "overcooked-v1/forced_coord"),
    ]
}


def sanitize(name: str) -> str:
    """Normalize an agent label into a filesystem/config-safe key.

    'lbrdiv-conf ([1, 0])' -> 'lbrdiv_conf_1_0', matching the naming used by
    evaluation/configs/global_heldout_br.yaml.
    """
    out = []
    for ch in name:
        out.append(ch if (ch.isalnum() or ch == "_") else "_")
    key = "".join(out)
    while "__" in key:
        key = key.replace("__", "_")
    return key.strip("_")

"""Explicit supported games. Legacy construction remains in the existing facade."""

from functools import partial

from ..game_registry import GameRegistry, GameSpec, SlotSpec
from ..objects import BreakoutRelations, DEMON_SLOTS, MultiGameRelations
from .pong import PONG_ACTIONS, PONG_OCATARI

REGISTRY = GameRegistry()
REGISTRY.register(PONG_OCATARI)
for game, names, actions in (
    ("Breakout", ("player", "ball"), PONG_ACTIONS[:4]),
    ("Pong", ("enemy", "player", "ball"), PONG_ACTIONS),
    ("DemonAttack", DEMON_SLOTS, PONG_ACTIONS),
):
    REGISTRY.register(GameSpec(
        game=game, backend_id=f"legacy_visual_{game.lower()}_v1",
        env_id=f"{game}NoFrameskip-v4", actions=actions,
        # Legacy categories are semantic labels, not RAM slot categories.
        slots=tuple(SlotSpec(i, name, name) for i, name in enumerate(names)),
        relation_schema=f"legacy_{game.lower()}_v1",
        representations=("comparative", "logical"), incomplete_states=(False, True),
        reset_policies=("legacy_fire",),
        reward_transforms=("sign_plus_0.1" if game == "Pong" else "sign",),
        snapshot_version="atari_relational_v1", exact_resume=True,
        relation_factory=(BreakoutRelations if game == "Breakout"
                          else partial(MultiGameRelations, game)),
    ))

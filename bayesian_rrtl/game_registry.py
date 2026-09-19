"""Small explicit capability registry; importing it never imports OCAtari."""

from dataclasses import dataclass
from typing import Callable

from .objects import ObjectFrame


@dataclass(frozen=True)
class SlotSpec:
    index: int
    identity: str
    category: str


@dataclass(frozen=True)
class GameSpec:
    game: str
    backend_id: str
    env_id: str
    actions: tuple[str, ...]
    slots: tuple[SlotSpec, ...]
    relation_schema: str
    representations: tuple[str, ...]
    incomplete_states: tuple[bool, ...]
    reset_policies: tuple[str, ...]
    reward_transforms: tuple[str, ...]
    snapshot_version: str | None
    exact_resume: bool
    object_adapter: Callable[[object], ObjectFrame] | None = None
    relation_factory: Callable | None = None
    backend_factory: Callable | None = None


class GameRegistry:
    def __init__(self):
        self._specs = {}

    def register(self, spec: GameSpec) -> None:
        key = (spec.game, spec.backend_id)
        if key in self._specs:
            raise ValueError(f"Game/backend already registered: {key}")
        if len({slot.identity for slot in spec.slots}) != len(spec.slots):
            raise ValueError("Duplicate slot identity")
        if len({slot.index for slot in spec.slots}) != len(spec.slots):
            raise ValueError("Duplicate slot index")
        self._specs[key] = spec

    def get(self, game: str, backend_id: str) -> GameSpec:
        try:
            return self._specs[(game, backend_id)]
        except KeyError as exc:
            raise ValueError(f"Unsupported game/backend: {game}/{backend_id}") from exc

    def keys(self):
        return tuple(self._specs)


def get_game_spec(game: str, backend_id: str) -> GameSpec:
    from .games import REGISTRY
    return REGISTRY.get(game, backend_id)

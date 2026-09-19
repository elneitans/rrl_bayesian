"""Object-based emulator boundary, independent of any optional Atari package."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .objects import ObjectFrame


@dataclass(frozen=True)
class ObjectStep:
    frame: ObjectFrame
    raw_reward: float
    terminated: bool
    truncated: bool


@runtime_checkable
class ObjectBackend(Protocol):
    backend_id: str
    actions: tuple[str, ...]

    def reset(self, *, seed: int) -> ObjectFrame: ...
    def step(self, action: int) -> ObjectStep: ...
    def snapshot(self) -> dict: ...
    def restore(self, snapshot: dict) -> None: ...
    def close(self) -> None: ...

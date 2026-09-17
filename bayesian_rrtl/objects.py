"""Object boundary for visual extraction now and a future OCAtari adapter.

Coordinates are pixels in the original RGB image, origin top-left. Identities
are semantic slots (player, ball), stable across frames; absence is explicit.
"""

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np

from .encoding import RelationKey, RelationalEncoder


@dataclass(frozen=True)
class DetectedObject:
    identity: str
    present: bool
    bbox: tuple[float, float, float, float] | None = None

    def __post_init__(self):
        if not self.identity or type(self.present) is not bool:
            raise ValueError('An object needs an identity and explicit presence')
        if self.present:
            if self.bbox is None or len(self.bbox) != 4 or not all(map(math.isfinite, self.bbox)):
                raise ValueError('Present objects need finite x,y,width,height')
            if min(self.bbox[2:]) <= 0:
                raise ValueError('Object width and height must be positive')
            object.__setattr__(self, 'bbox', tuple(self.bbox))
        elif self.bbox is not None:
            raise ValueError('Absent objects must not carry stale coordinates')

    @property
    def center(self):
        if not self.present:
            return None
        x, y, width, height = self.bbox
        return x + width / 2, y + height / 2


@dataclass(frozen=True)
class ObjectFrame:
    objects: tuple[DetectedObject, ...]
    width: int = 160
    height: int = 210

    def __post_init__(self):
        object.__setattr__(self, 'objects', tuple(sorted(self.objects, key=lambda o: o.identity)))
        if len({o.identity for o in self.objects}) != len(self.objects):
            raise ValueError('Duplicate object identity')
        if self.width < 1 or self.height < 1:
            raise ValueError('Invalid frame dimensions')


class ObjectExtractor(Protocol):
    """Future backends must supply this contract; no OCAtari dependency here."""
    backend_id: str

    def extract(self, observation: np.ndarray) -> ObjectFrame: ...
    def reset(self) -> None: ...
    def snapshot(self) -> dict: ...
    def restore(self, state: dict) -> None: ...


class LegacyBreakoutExtractor:
    backend_id = 'legacy_visual_breakout_v1'

    def __init__(self):
        from preprocessing import BreakoutPreprocessor
        self.preprocessor = BreakoutPreprocessor()

    def extract(self, observation):
        if np.shape(observation) != (210, 160, 3):
            raise ValueError('Breakout extractor expects original 210x160 RGB frames')
        self.preprocessor.read_from_array(observation)
        boxes = self.preprocessor.get_boxes()
        objects = []
        for identity, box in (('ball', boxes[:4]), ('player', boxes[4:])):
            present = bool(box[2] > 0 and box[3] > 0)
            objects.append(DetectedObject(identity, present, tuple(map(float, box)) if present else None))
        return ObjectFrame(tuple(objects))

    def reset(self):
        # get_boxes derives everything from the current frame; no temporal tracker.
        pass

    def snapshot(self):
        return {'backend_id': self.backend_id}

    def restore(self, state):
        if state != self.snapshot():
            raise ValueError('Extractor snapshot mismatch')


def make_extractor(backend):
    if backend == LegacyBreakoutExtractor.backend_id:
        return LegacyBreakoutExtractor()
    for game in ('Pong', 'DemonAttack'):
        if backend == f'legacy_visual_{game.lower()}_v1':
            return LegacyMultiGameExtractor(game)
    if backend.startswith('ocatari'):
        raise NotImplementedError('OCAtari adapter is deferred; use legacy_visual_breakout_v1')
    raise ValueError(f'Unknown extractor backend: {backend}')


class BreakoutRelations:
    """Preserve historical Fact semantics, with explicit temporal object frames.

    The legacy relation function fixes paddle y and can discard incomplete
    states. Those choices remain identical across the paired H6 variants.
    """
    def __init__(self, representation='comparative', include_incomplete_states=False):
        if representation not in ('comparative', 'logical'):
            raise ValueError('Unknown relation representation')
        from preprocessing import get_state_comparative_breakout, get_state_logical_breakout
        self.builder = (get_state_comparative_breakout if representation == 'comparative'
                        else get_state_logical_breakout)
        self.representation = representation
        self.include_incomplete_states = include_incomplete_states
        self.previous = None

    @staticmethod
    def legacy_info(frame):
        objects = {obj.identity: obj for obj in frame.objects}
        if set(objects) != {'player', 'ball'} or (frame.width, frame.height) != (160, 210):
            raise ValueError('Breakout relations require player/ball slots and 160x210 coordinates')
        labels = {}
        for name, obj in objects.items():
            x, y = obj.center if obj.present else (0.0, 0.0)
            labels[f'{name}_x'], labels[f'{name}_y'] = x, y
        return {'labels': labels}

    def reset(self):
        self.previous = None

    def transform(self, frame):
        current = self.legacy_info(frame)
        state = self.builder(current, self.previous, self.include_incomplete_states)
        self.previous = current
        return frozenset(state)

    def encoder(self):
        pairs = (('player_t', 'ball_t'), ('ball_t', 'ball_t-1'), ('player_t', 'player_t-1'))
        dimensions = ('x', 'y') if self.representation == 'comparative' else (
            'same-x', 'more-x', 'less-x', 'same-y', 'more-y', 'less-y')
        catalog = [RelationKey(self.representation, name, *pair) for name in dimensions for pair in pairs]
        catalog += [RelationKey('logical', 'present', obj) for obj in ('player_t', 'ball_t', 'player_t-1', 'ball_t-1')]
        catalog.append(RelationKey('logical', 'incontact', 'player_t', 'ball_t'))
        return RelationalEncoder(catalog)


DEMON_SLOTS = ('player', 'player_missile', 'enemy_missile', *(f'enemy_big_{i}' for i in range(3)),
               *(f'enemy_small_{i}' for i in range(6)))


class LegacyMultiGameExtractor(LegacyBreakoutExtractor):
    """Historical Pong/DemonAttack boxes; enemy indices retain spatial-rank semantics."""
    def __init__(self, game):
        from preprocessing import PongPreprocessor, DemonAttackPreprocessor
        if game not in ('Pong', 'DemonAttack'):
            raise ValueError('Unsupported multigame extractor')
        self.game = game
        self.backend_id = f'legacy_visual_{game.lower()}_v1'
        self.preprocessor = PongPreprocessor() if game == 'Pong' else DemonAttackPreprocessor()

    def extract(self, observation):
        if np.shape(observation) != (210, 160, 3):
            raise ValueError('Expected original RGB frame')
        objects = []
        if self.game == 'Pong':
            self.preprocessor.read_from_array(observation)
            for identity, box in zip(('enemy', 'player', 'ball'), self.preprocessor.get_boxes()):
                present = bool(box[2] > 0 and box[3] > 0)
                objects.append(DetectedObject(identity, present, tuple(map(float, box)) if present else None))
        else:
            info = self.preprocessor.get_info(observation)
            for identity in DEMON_SLOTS:
                if identity in info:
                    left, right = info[identity]
                    box = (float(left.x), float(left.y), float(right.x-left.x), float(right.y-left.y))
                    objects.append(DetectedObject(identity, True, box))
                else:
                    objects.append(DetectedObject(identity, False))
        return ObjectFrame(tuple(objects))


class MultiGameRelations(BreakoutRelations):
    def __init__(self, game, representation='comparative', include_incomplete_states=False):
        import preprocessing
        self.game = game
        self.representation = representation
        self.include_incomplete_states = include_incomplete_states
        suffix = 'pong' if game == 'Pong' else 'demon_attack'
        self.builder = getattr(preprocessing, f'get_state_{representation}_{suffix}')
        self.previous = None

    def legacy_info(self, frame):
        if self.game == 'Pong':
            labels = {}
            for obj in frame.objects:
                x, y = obj.center if obj.present else (0., 0.)
                labels[f'{obj.identity}_x'], labels[f'{obj.identity}_y'] = x, y
            if {o.identity for o in frame.objects} != {'player', 'enemy', 'ball'}:
                raise ValueError('Missing Pong slots')
            return {'labels': labels}
        from preprocessing import Point
        if {o.identity for o in frame.objects} != set(DEMON_SLOTS):
            raise ValueError('Missing DemonAttack slots')
        info = {}
        for obj in frame.objects:
            if obj.present:
                x, y, w, h = obj.bbox
                info[obj.identity] = (Point(x, y), Point(x+w, y+h))
        return info

    def encoder(self):
        from itertools import combinations
        slots = ('player', 'ball', 'enemy') if self.game == 'Pong' else DEMON_SLOTS
        pairs = [(a+'_t', b+'_t') for a, b in combinations(slots, 2)]
        pairs += [(a+'_t', a+'_t-1') for a in slots]
        dimensions = ('x', 'y') if self.representation == 'comparative' else (
            'same-x', 'more-x', 'less-x', 'same-y', 'more-y', 'less-y')
        catalog = [RelationKey(self.representation, name, *pair) for name in dimensions for pair in pairs]
        catalog += [RelationKey('logical', 'present', a+t) for a in slots for t in ('_t', '_t-1')]
        catalog += [RelationKey('logical', 'incontact', *pair) for pair in pairs]
        return RelationalEncoder(catalog)

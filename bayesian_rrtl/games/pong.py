"""Pong RAM slots verified against OCAtari 2.2.1 (hud=False)."""

from dataclasses import asdict
import math

from ..encoding import RelationKey, RelationalEncoder
from ..game_registry import GameSpec, SlotSpec
from ..objects import DetectedObject, ObjectFrame

PONG_ACTIONS = ("NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE")
PONG_SLOTS = (SlotSpec(0, "player", "Player"), SlotSpec(1, "ball", "Ball"),
              SlotSpec(2, "enemy", "Enemy"))
SPATIAL_PAIRS = (("player", "ball"), ("player", "enemy"), ("ball", "enemy"))
CONTACT_PAIRS = (("player", "ball"), ("ball", "enemy"))
PONG_CATALOG = tuple(
    [RelationKey("comparative", axis, a + "_t", b + "_t")
     for a, b in SPATIAL_PAIRS for axis in ("x", "y")]
    + [RelationKey("comparative", axis, slot.identity + "_t", slot.identity + "_t-1")
       for slot in PONG_SLOTS for axis in ("x", "y")]
    + [RelationKey("logical", "present", slot.identity + time)
       for slot in PONG_SLOTS for time in ("_t", "_t-1")]
    + [RelationKey("logical", "incontact", a + "_t", b + "_t") for a, b in CONTACT_PAIRS]
)


def pong_object_frame(objects) -> ObjectFrame:
    if len(objects) != len(PONG_SLOTS):
        raise ValueError(f"Pong RAM requires exactly three slots; got {len(objects)}")
    copied = []
    for slot in PONG_SLOTS:
        obj = objects[slot.index]
        if not obj:
            copied.append(DetectedObject(slot.identity, False))
            continue
        if getattr(obj, "category", None) != slot.category:
            raise ValueError(f"Pong slot {slot.index} requires category {slot.category}")
        try:
            bbox = tuple(float(value) for value in obj.xywh)
            copied.append(DetectedObject(slot.identity, True, bbox))
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid bbox for Pong slot {slot.identity}") from exc
    return ObjectFrame(tuple(copied), width=160, height=210)


class PongRelationsV1:
    """Fixed comparative relations in screen coordinates, with one-frame memory.

    Displacements are between consecutive decisions. Bbox contact does not
    imply detection of collisions that occur between frames skipped by ALE.
    """

    relation_schema = "pong_relational_v1"

    def __init__(self, representation="comparative", include_incomplete_states=True):
        if representation != "comparative" or include_incomplete_states is not True:
            raise ValueError("pong_relational_v1 requires comparative representation and incomplete states")
        # Reuse the historical fact data contract, never its relation builders.
        from preprocessing import Fact
        self._fact = Fact
        self.previous = None

    def encoder(self):
        return RelationalEncoder(PONG_CATALOG)

    def manifest(self):
        return {
            "relation_schema": self.relation_schema,
            "config": {"representation": "comparative", "include_incomplete_states": True,
                       "coordinates": "pixel_centers_top_left_x_right_y_down",
                       "spatial_tolerance": 4, "temporal_tolerance": 0,
                       "contact": "bbox_overlap_or_touch_both_axes",
                       "history": "previous_decision_clear_before_nonzero_raw_reward"},
            "catalog_hash": self.encoder().catalog_hash,
        }

    @staticmethod
    def _objects(frame):
        if not isinstance(frame, ObjectFrame) or (frame.width, frame.height) != (160, 210):
            raise ValueError("Pong relations require a 160x210 ObjectFrame")
        objects = {obj.identity: obj for obj in frame.objects}
        if set(objects) != {slot.identity for slot in PONG_SLOTS}:
            raise ValueError("Pong relations require exactly player/ball/enemy slots")
        return objects

    @staticmethod
    def _compare(first, second, tolerance):
        difference = first - second
        return "same" if abs(difference) <= tolerance else "less" if difference < 0 else "more"

    @staticmethod
    def _contact(first, second):
        ax, ay, aw, ah = first.bbox
        bx, by, bw, bh = second.bbox
        return ax <= bx + bw and bx <= ax + aw and ay <= by + bh and by <= ay + ah

    def reset(self):
        self.previous = None

    def transform(self, frame, *, raw_reward=0.0):
        objects = self._objects(frame)
        if not math.isfinite(raw_reward):
            raise ValueError("Pong relations require a finite raw reward")
        previous = (self._objects(self.previous)
                    if self.previous is not None and raw_reward == 0 else {})
        facts = []
        for name, obj in objects.items():
            before = previous.get(name)
            facts.append(self._fact("logical", obj.present, "present", name + "_t"))
            facts.append(self._fact("logical", before is not None and before.present,
                                    "present", name + "_t-1"))
            if obj.present and before is not None and before.present:
                for axis, current, old in zip(("x", "y"), obj.center, before.center):
                    facts.append(self._fact("comparative", self._compare(current, old, 0),
                                            axis, name + "_t", name + "_t-1"))
        for a, b in SPATIAL_PAIRS:
            first, second = objects[a], objects[b]
            if first.present and second.present:
                for axis, left, right in zip(("x", "y"), first.center, second.center):
                    facts.append(self._fact("comparative", self._compare(left, right, 4),
                                            axis, a + "_t", b + "_t"))
        for a, b in CONTACT_PAIRS:
            if objects[a].present and objects[b].present:
                facts.append(self._fact("logical", self._contact(objects[a], objects[b]),
                                        "incontact", a + "_t", b + "_t"))
        # Even an absent ball replaces history, preventing comparison across gaps.
        self.previous = frame
        return frozenset(facts)

    def snapshot(self):
        """JSON-compatible relational memory only; not an emulator snapshot."""
        return {"kind": "pong_relations_memory_v1", "manifest": self.manifest(),
                "previous": None if self.previous is None else asdict(self.previous)}

    def restore(self, snapshot):
        if (not isinstance(snapshot, dict)
                or set(snapshot) != {"kind", "manifest", "previous"}
                or snapshot["kind"] != "pong_relations_memory_v1"
                or snapshot["manifest"] != self.manifest()):
            raise ValueError("Pong relational snapshot semantics/config mismatch")
        previous = snapshot["previous"]
        if previous is not None:
            try:
                previous = ObjectFrame(
                    tuple(DetectedObject(**obj) for obj in previous["objects"]),
                    width=previous["width"], height=previous["height"],
                )
                self._objects(previous)
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError("Invalid Pong relational snapshot frame") from exc
        self.previous = previous


def make_pong_backend(*, max_episode_steps):
    from ..ocatari_backend import OCAtariBackend
    return OCAtariBackend("Pong", max_episode_steps=max_episode_steps)


PONG_OCATARI = GameSpec(
    game="Pong", backend_id="ocatari_ram_v1", env_id="ALE/Pong-v5",
    actions=PONG_ACTIONS, slots=PONG_SLOTS,
    relation_schema="pong_relational_v1", representations=("comparative",),
    incomplete_states=(True,), reset_policies=("none",), reward_transforms=("sign",),
    snapshot_version="atari_relational_v2", exact_resume=True, object_adapter=pong_object_frame,
    relation_factory=PongRelationsV1, backend_factory=make_pong_backend,
)

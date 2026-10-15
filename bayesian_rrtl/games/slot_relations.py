"""C13/C14 fixed slot catalogues, certified absence and conservative temporal memory."""
from dataclasses import asdict
import math

from bayesian_rrtl.encoding import RelationKey, RelationalEncoder
from bayesian_rrtl.objects import DetectedObject, ObjectFrame
from experiments.pong_analysis_runtime import digest


def convert_slots(objects, slots, *, primary=None):
    if len(objects) != len(slots):
        raise ValueError(f'Detector requires {len(slots)} slots; got {len(objects)}')
    copied = []
    for slot in slots:
        obj = objects[slot.index]
        category = getattr(obj, 'category', None)
        if category not in (slot.category, 'NoObject'):
            raise ValueError(f'Slot {slot.index}/{slot.identity}: unexpected category {category}; expected {slot.category}/NoObject')
        # False-y unknown objects are never accepted as certified absence.
        if category == 'NoObject':
            copied.append(DetectedObject(slot.identity, False))
        elif not bool(obj):
            raise ValueError(f'Invisible {category} is not certified NoObject in slot {slot.identity}')
        else:
            try:
                copied.append(DetectedObject(slot.identity, True, tuple(float(x) for x in obj.xywh)))
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'Invalid bbox for slot {slot.identity}') from exc
    if primary is not None:
        copied = [o for o in copied if o.identity in primary]
    return ObjectFrame(tuple(copied))


class SlotRelations:
    def __init__(self, names, pairs, contacts, *, schema, movement, spatial_tolerance=4.,
                 temporal_tolerance=0.):
        self.names, self.pairs, self.contacts = tuple(names), tuple(pairs), tuple(contacts)
        self.movement = tuple(movement)
        self.relation_schema = schema
        self.spatial_tolerance, self.temporal_tolerance = spatial_tolerance, temporal_tolerance
        if any(not math.isfinite(v) or v < 0 for v in (spatial_tolerance, temporal_tolerance)):
            raise ValueError('Tolerances must be finite and nonnegative')
        self.catalog = tuple(
            [RelationKey('comparative', axis, a+'_t', b+'_t') for a,b in self.pairs for axis in ('x','y')]
            + [RelationKey('comparative', axis, name+'_t', name+'_t-1') for name in self.movement for axis in ('x','y')]
            + [RelationKey('logical', 'present', name+suffix) for name in self.names for suffix in ('_t','_t-1')]
            + [RelationKey('logical', 'incontact', a+'_t', b+'_t') for a,b in self.contacts])
        self.previous = None

    def encoder(self):
        return RelationalEncoder(self.catalog)

    def manifest(self):
        return {'relation_schema': self.relation_schema, 'catalog_hash': self.encoder().catalog_hash,
                'objects': list(self.names), 'relations': len(self.catalog), 'movement': list(self.movement),
                'coordinates': 'pixel_centers_top_left_x_right_y_down',
                'spatial_tolerance': self.spatial_tolerance, 'temporal_tolerance': self.temporal_tolerance,
                'contact': 'bbox_overlap_or_touch_both_axes_not_emulator_collision',
                'history': 'previous_decision_no_reward_reset; unknown_previous_presence_unobserved',
                'identity': 'detector_slots; no physical enemy/projectile velocity claim'}

    def reset(self):
        self.previous = None

    def objects(self, frame):
        if not isinstance(frame, ObjectFrame) or (frame.width, frame.height) != (160,210):
            raise ValueError('Expected 160x210 ObjectFrame')
        objects = {o.identity:o for o in frame.objects}
        if set(objects) != set(self.names):
            raise ValueError('Object frame differs from fixed slot catalogue')
        return objects

    @staticmethod
    def compare(a,b,tolerance):
        d = a-b
        return 'same' if abs(d) <= tolerance else 'less' if d < 0 else 'more'

    @staticmethod
    def contact(a,b):
        ax,ay,aw,ah = a.bbox
        bx,by,bw,bh = b.bbox
        return ax <= bx+bw and bx <= ax+aw and ay <= by+bh and by <= ay+ah

    def transform(self, frame, *, raw_reward=0., invalidated=()):
        from preprocessing import Fact
        if not math.isfinite(raw_reward):
            raise ValueError('Finite raw reward required')
        objects = self.objects(frame)
        previous = self.objects(self.previous) if self.previous is not None else {}
        if set(invalidated)-set(self.names):
            raise ValueError('Unknown invalidated slot')
        facts=[]
        for name in self.names:
            obj, old = objects[name], previous.get(name)
            facts.append(Fact('logical',obj.present,'present',name+'_t'))
            if old is not None and name not in invalidated:
                facts.append(Fact('logical',old.present,'present',name+'_t-1'))
            if (name in self.movement and name not in invalidated and obj.present
                    and old is not None and old.present and obj.bbox[2:] == old.bbox[2:]):
                for axis,a,b in zip(('x','y'),obj.center,old.center):
                    facts.append(Fact('comparative',self.compare(a,b,self.temporal_tolerance),axis,name+'_t',name+'_t-1'))
        for a,b in self.pairs:
            if objects[a].present and objects[b].present:
                for axis,x,y in zip(('x','y'),objects[a].center,objects[b].center):
                    facts.append(Fact('comparative',self.compare(x,y,self.spatial_tolerance),axis,a+'_t',b+'_t'))
        for a,b in self.contacts:
            if objects[a].present and objects[b].present:
                facts.append(Fact('logical',self.contact(objects[a],objects[b]),'incontact',a+'_t',b+'_t'))
        self.previous=frame
        return frozenset(facts)

    def snapshot(self):
        return {'kind':'slot_relations_memory_v1','manifest':self.manifest(),
                'previous':None if self.previous is None else asdict(self.previous)}

    def restore(self, saved):
        if saved['kind'] != 'slot_relations_memory_v1' or digest(saved['manifest']) != digest(self.manifest()):
            raise ValueError('Relational history contract mismatch')
        previous = saved['previous']
        if previous is not None:
            previous = ObjectFrame(tuple(DetectedObject(**o) for o in previous['objects']),previous['width'],previous['height'])
            self.objects(previous)
        self.previous=previous

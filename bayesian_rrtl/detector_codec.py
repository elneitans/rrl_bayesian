"""Explicit per-game OCAtari 2.2.1 RAM object codec, including unobserved slots."""
import copy
import importlib

BASE_FIELDS = ('_rgb','_xy','wh','_prev_xy','_orientation','hud','_visible')


class DetectorCodec:
    version='detector_fields_221_v1'

    def __init__(self, game, module, slots):
        self.game,self.module,self.slots=game,module,slots

    def fields(self, category):
        return BASE_FIELDS+(('nslen',) if category == 'NoObject' else ())

    def encode(self, objects):
        if len(objects) != len(self.slots):
            raise ValueError(f'{self.game}: unexpected detector slot count')
        result=[]
        for slot,obj in zip(self.slots,objects):
            category=getattr(obj,'category',None)
            if category not in (slot.category,'NoObject'):
                raise ValueError(f'{self.game} slot {slot.index}: unknown category {category}')
            if set(vars(obj)) != set(self.fields(category)):
                raise ValueError(f'{self.game}/{category}: unknown mutable detector fields {set(vars(obj))^set(self.fields(category))}')
            result.append({'category':category,'fields':copy.deepcopy(vars(obj))})
        return result

    def decode(self, saved):
        module=importlib.import_module(self.module)
        from ocatari.ram.game_objects import NoObject
        if len(saved) != len(self.slots):
            raise ValueError('Detector snapshot slot count mismatch')
        objects=[]
        for slot,item in zip(self.slots,saved):
            category=item['category']
            if category not in (slot.category,'NoObject') or set(item) != {'category','fields'}:
                raise ValueError('Unknown detector snapshot category/fields')
            if set(item['fields']) != set(self.fields(category)):
                raise ValueError('Unknown mutable detector snapshot fields')
            cls=NoObject if category == 'NoObject' else getattr(module,category)
            obj=cls()
            if set(vars(obj)) != set(self.fields(category)):
                raise ValueError('Detector constructor changed')
            vars(obj).update(copy.deepcopy(item['fields']))
            objects.append(obj)
        self.encode(objects)
        return objects

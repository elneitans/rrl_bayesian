"""Breakout OCAtari 2.2.1: audit all 110 detector slots; publish only paddle/ball."""
from bayesian_rrtl.game_registry import GameSpec, SlotSpec
from .slot_relations import SlotRelations, convert_slots

BREAKOUT_ACTIONS = ('NOOP','FIRE','RIGHT','LEFT')  # verified against ALE minimal action meanings
BREAKOUT_SLOTS = (SlotSpec(0,'player','Player'),SlotSpec(1,'ball','Ball')) + tuple(
    SlotSpec(2+i,f'block_{i:03d}','Block') for i in range(108))


def breakout_object_frame(objects):
    return convert_slots(objects,BREAKOUT_SLOTS,primary=('player','ball'))


class BreakoutRelationsV1(SlotRelations):
    def __init__(self, *, spatial_tolerance=4., temporal_tolerance=0.):
        super().__init__(('player','ball'),(('player','ball'),),(('player','ball'),),
            schema='breakout_primary_v1',movement=('player','ball'),
            spatial_tolerance=spatial_tolerance,temporal_tolerance=temporal_tolerance)


def make_backend(*, max_episode_steps):
    from bayesian_rrtl.ocatari_backend import OCAtariBackendV2
    return OCAtariBackendV2('Breakout',max_episode_steps=max_episode_steps)


def codec():
    from bayesian_rrtl.detector_codec import DetectorCodec
    return DetectorCodec('Breakout','ocatari.ram.breakout',BREAKOUT_SLOTS)


BREAKOUT_OCATARI = GameSpec(game='Breakout',backend_id='ocatari_ram_v2',env_id='ALE/Breakout-v5',
    actions=BREAKOUT_ACTIONS,slots=BREAKOUT_SLOTS,relation_schema='breakout_primary_v1',
    representations=('comparative',),incomplete_states=(True,),reset_policies=('none',),
    reward_transforms=('sign',),snapshot_version='ocatari_ram_state_v2',exact_resume=True,
    object_adapter=breakout_object_frame,relation_factory=BreakoutRelationsV1,backend_factory=make_backend,
    detector_codec_factory=codec,detector_module='ocatari.ram.breakout')

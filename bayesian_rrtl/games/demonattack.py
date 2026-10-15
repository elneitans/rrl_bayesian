"""Demon Attack RAM slots from OCAtari 2.2.1, not position-sorted identities."""
from bayesian_rrtl.game_registry import GameSpec,SlotSpec
from .slot_relations import SlotRelations,convert_slots

DEMON_ACTIONS=('NOOP','FIRE','RIGHT','LEFT','RIGHTFIRE','LEFTFIRE')
DEMON_SLOTS=(SlotSpec(0,'player','Player'),SlotSpec(1,'player_missile','PlayerMissile'))+tuple(
    SlotSpec(2+i,f'enemy_{i}','Enemy') for i in range(8))+tuple(
    SlotSpec(10+i,f'enemy_part_{i}','EnemyPart') for i in range(6))+tuple(
    SlotSpec(16+i,f'enemy_missile_{i:02d}','EnemyMissile') for i in range(14))
DEMON_TARGETS=tuple(s.identity for s in DEMON_SLOTS if s.category in ('Enemy','EnemyPart'))


def demonattack_object_frame(objects):
    return convert_slots(objects,DEMON_SLOTS)


class DemonAttackRelationsV1(SlotRelations):
    def __init__(self, *, spatial_tolerance=4.,temporal_tolerance=0.):
        names=tuple(s.identity for s in DEMON_SLOTS)
        # The detector cannot certify physical continuity of enemy/projectile occupants.
        # Do not publish their velocity, even if the same index is occupied twice.
        super().__init__(names,tuple(('player',n) for n in names if n != 'player'),
            tuple(('player',s.identity) for s in DEMON_SLOTS if s.category == 'EnemyMissile'),
            schema='demonattack_slots_v1',movement=('player',),
            spatial_tolerance=spatial_tolerance,temporal_tolerance=temporal_tolerance)


def make_backend(*, max_episode_steps):
    from bayesian_rrtl.ocatari_backend import OCAtariBackendV2
    return OCAtariBackendV2('DemonAttack',max_episode_steps=max_episode_steps)


def codec():
    from bayesian_rrtl.detector_codec import DetectorCodec
    return DetectorCodec('DemonAttack','ocatari.ram.demonattack',DEMON_SLOTS)


DEMONATTACK_OCATARI=GameSpec(game='DemonAttack',backend_id='ocatari_ram_v2',env_id='ALE/DemonAttack-v5',
    actions=DEMON_ACTIONS,slots=DEMON_SLOTS,relation_schema='demonattack_slots_v1',
    representations=('comparative',),incomplete_states=(True,),reset_policies=('none',),
    reward_transforms=('sign',),snapshot_version='ocatari_ram_state_v2',exact_resume=True,
    object_adapter=demonattack_object_frame,relation_factory=DemonAttackRelationsV1,backend_factory=make_backend,
    detector_codec_factory=codec,detector_module='ocatari.ram.demonattack')

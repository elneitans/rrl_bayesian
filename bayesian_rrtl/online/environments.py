"""Common A/B environment factory extended to audited per-game RAM contracts."""
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path
import time

import numpy as np

from bayesian_rrtl.game_registry import get_game_spec
from experiments.pong_analysis_runtime import digest


@dataclass(frozen=True)
class GameEnvironmentConfig:
    game: str
    pipeline: str
    rom_sha256: str
    horizon: int
    spatial_tolerance: float = 4.
    temporal_tolerance: float = 0.
    contract: str = 'atari_common_v3'

    def __post_init__(self):
        if self.game not in ('Pong','Breakout','DemonAttack') or self.contract != 'atari_common_v3':
            raise ValueError('Unknown common Atari game/contract')
        if self.pipeline not in (('visual','ocatari') if self.game == 'Pong' else ('ocatari',)):
            raise ValueError('Only Pong has a complementary visual pipeline')
        if self.game == 'Pong' and (self.spatial_tolerance,self.temporal_tolerance) != (4.,0.):
            raise ValueError('Pong retains its A3 relation contract')
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError('Positive episode horizon required')
        if len(self.rom_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.rom_sha256):
            raise ValueError('Explicit verified ROM SHA256 required')
        self.relations()

    def spec(self):
        return get_game_spec(self.game,'ocatari_ram_v1' if self.game == 'Pong' else 'ocatari_ram_v2')

    def relations(self):
        if self.game == 'Pong':
            from bayesian_rrtl.games.pong import PongRelationsV1
            return PongRelationsV1()
        return self.spec().relation_factory(spatial_tolerance=self.spatial_tolerance,
                                            temporal_tolerance=self.temporal_tolerance)

    def manifest(self):
        return {**asdict(self),'env_id':self.spec().env_id,'actions':list(self.spec().actions),
                'frameskip':4,'repeat_action_probability':0.,'reset_policy':'none',
                'learning_reward':'sign(raw_reward)','evaluation_reward':'raw_reward',
                'relations':self.relations().manifest()}


class CommonGameEnvironment:
    def __init__(self, config, *, backend=None):
        self.config=config
        self.actions=config.spec().actions
        self.relations=config.relations()
        self.backend=backend or config.spec().backend_factory(max_episode_steps=config.horizon)
        if tuple(self.backend.actions) != self.actions:
            self.backend.close()
            raise ValueError('Environment action order mismatch')
        self.done=True
        self.steps=0
        self.last_lives=None
        self.last_auxiliary=None
        if backend is None and self.backend.manifest()['rom_sha256'] != config.rom_sha256:
            self.backend.close()
            raise ValueError('ROM checksum mismatch')
        self.provenance='verified_atari' if backend is None else 'injected_driver'

    def encoder(self):
        return self.relations.encoder()

    def lives(self):
        if hasattr(self.backend,'lives'):
            return self.backend.lives()
        return int(self.backend.env._env.unwrapped.ale.lives())

    def reset(self, *, seed):
        self.relations.reset()
        self.steps=0
        frame=self.backend.reset(seed=seed)
        self.last_lives=self.lives()
        self.done=False
        return self._facts(frame,0.,False)

    def _facts(self, frame, raw, life_lost):
        start=time.perf_counter()
        if self.config.game == 'Pong':
            facts=self.relations.transform(frame,raw_reward=raw)
        else:
            # Lives can change without an observed absent decision between frameskips.
            # Only the ball history is invalidated; rewards never clear motion.
            invalid=('ball',) if life_lost and self.config.game == 'Breakout' else ()
            facts=self.relations.transform(frame,raw_reward=raw,invalidated=invalid)
        self.last_auxiliary={'lives':self.last_lives,'life_lost':life_lost,
                             'objects':asdict(frame),'learning_reward':'sign(raw_reward)'}
        self.last_relation_seconds=time.perf_counter()-start
        return facts

    def step(self, action):
        if self.done or type(action) is not int or not 0 <= action < len(self.actions):
            raise ValueError('Invalid action or completed episode')
        result=self.backend.step(action)
        self.steps+=1
        lives=self.lives()
        lost=self.last_lives is not None and lives < self.last_lives
        self.last_lives=lives
        terminal=bool(result.terminated)
        truncated=bool(result.truncated or (self.steps >= self.config.horizon and not terminal))
        self.done=terminal or truncated
        return self._facts(result.frame,result.raw_reward,lost),float(np.sign(result.raw_reward)),float(result.raw_reward),terminal,truncated

    def runtime_manifest(self):
        return {'contract':self.config.manifest(),'driver_provenance':self.provenance,
                'backend':self.backend.manifest(),
                'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def snapshot(self):
        return {'kind':'common_atari_state_v3','manifest':self.runtime_manifest(),
                'backend':self.backend.snapshot(),'relations':self.relations.snapshot(),
                'steps':self.steps,'done':self.done,'last_lives':self.last_lives,
                'auxiliary':copy.deepcopy(self.last_auxiliary)}

    def restore(self, saved):
        if saved['kind'] != 'common_atari_state_v3' or digest(saved['manifest']) != digest(self.runtime_manifest()):
            raise ValueError('Common Atari snapshot contract/runtime mismatch')
        # Validate relational state before mutating the emulator.
        memory=self.config.relations()
        memory.restore(saved['relations'])
        self.backend.restore(saved['backend'])
        self.relations=memory
        self.steps,self.done,self.last_lives=saved['steps'],saved['done'],saved['last_lives']
        self.last_auxiliary=copy.deepcopy(saved['auxiliary'])

    def close(self):
        self.backend.close()


@dataclass(frozen=True)
class CommonGameFactory:
    config: GameEnvironmentConfig

    def __call__(self):
        if self.config.game == 'Pong':
            # A3 behaviour and reset/reward semantics remain unchanged.
            from experiments.pong_common_contract import CommonPongConfig,CommonPongFactory
            return CommonPongFactory(CommonPongConfig(self.config.pipeline,self.config.rom_sha256,self.config.horizon))()
        return CommonGameEnvironment(self.config)

    def with_horizon(self, horizon):
        return type(self)(replace(self.config,horizon=horizon))

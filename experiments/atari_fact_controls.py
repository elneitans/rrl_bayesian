"""Diagnostic controls consume published Facts only; no detector/geometry side channel."""
from bayesian_rrtl.games.demonattack import DEMON_TARGETS


class FactController:
    def __init__(self, game, policy='heuristic'):
        if game not in ('Breakout','DemonAttack') or policy not in ('heuristic','random','noop'):
            raise ValueError('Unknown diagnostic game/policy')
        self.game,self.policy=game,policy
        self.actions=4 if game == 'Breakout' else 6
        self.log=[]
        self.incomplete_states=0

    def action(self, facts, env, rng):
        values={(f.name,f.obj1,f.obj2):str(f.value) for f in facts}
        def present(n):
            return values.get(('present',n+'_t',None))
        if self.policy == 'noop':
            action,reason=0,'declared_noop'
        elif self.policy == 'random':
            action,reason=int(rng.integers(self.actions)),'declared_uniform_random'
        elif self.game == 'Breakout':
            x=values.get(('x','player_t','ball_t'))
            if present('ball') == 'False':
                action,reason=1,'explicit_launch_absent_ball'
            else:
                action={'less':2,'more':3,'same':0}.get(x,0)
                reason='align_fact' if x else 'insufficient_facts_noop'
        else:
            target=next((n for n in DEMON_TARGETS if present(n) == 'True'),None)
            x=values.get(('x','player_t',target+'_t')) if target else None
            action={'less':4,'more':5,'same':1}.get(x,1)
            reason='align_first_present_slot_and_fire' if x else 'no_target_explicit_fire'
        batch=env.encoder().transform([facts])
        self.incomplete_states+=int(not batch.observed.all())
        self.log.append({'action':action,'reason':reason,'incomplete':not bool(batch.observed.all())})
        return action

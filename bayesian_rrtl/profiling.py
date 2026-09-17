"""H6 stage profiles; cumulative times overlap and must not be added."""

import pstats
import resource
import sys


def profile_summary(profile):
    stats = pstats.Stats(profile)
    definitions = {
        'encoding': ('encoding.py', 'transform'),
        'targets': ('targets.py', 'frozen_q_targets'),
        'proposals': ('proposals.py', 'propose'),
        'sweeps': ('ensemble.py', 'sweep'),
        'prediction': ('ensemble.py', 'predict_mean'),
        'action_selection': ('agent.py', 'select_action'),
        'fits': ('agent.py', 'fit_if_due'),
    }
    result = {}
    for stage, (filename, function) in definitions.items():
        entries = [value for (path, _, name), value in stats.stats.items()
                   if path.endswith('/' + filename) and name == function]
        result[stage] = {'calls': sum(v[1] for v in entries), 'self_seconds': sum(v[2] for v in entries),
                         'cumulative_seconds': sum(v[3] for v in entries)}
    result['note'] = 'Instrumented training only; cumulative times are nested, not additive.'
    # macOS reports bytes, Linux KiB. Each pilot variant runs in its own process.
    result['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == 'darwin' else 1024)
    return result

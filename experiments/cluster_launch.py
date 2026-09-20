"""Single coordinator per campaign on a shared Linux filesystem; no test stage.

Slurm may terminate the job at its wall-time limit. Resume uses the last
published checkpoint; this wrapper does not promise a final save on SIGKILL.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def campaign_lock(output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock inode: unlinking it can allow simultaneous coordinators.
    with output.with_name(output.name + '.cluster.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another cluster coordinator owns this campaign') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def command_for(args):
    command = [sys.executable, '-m', 'experiments.compare_pong',
               '--output', str(Path(args.output).resolve()), '--stage', args.stage]
    extension = args.training_budget_seconds is not None or args.evaluation_budget_seconds is not None
    if args.stage == 'train':
        if args.kind is None or extension or args.budget_reason:
            raise ValueError('train requires --kind and does not accept budget extensions')
        command += ['--protocol', str(ROOT / 'configs' / f'comparison_pong_{args.kind}.json')]
    else:
        if args.kind is not None:
            raise ValueError('resume uses only the frozen protocol; omit --kind')
        if extension != bool(args.budget_reason and args.budget_reason.strip()):
            raise ValueError('Budget extensions require a nonempty reason')
        for name in ('training_budget_seconds', 'evaluation_budget_seconds', 'budget_reason'):
            value = getattr(args, name)
            if value is not None:
                command += ['--' + name.replace('_', '-'), str(value)]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('train', 'resume'), required=True)
    parser.add_argument('--kind', choices=('smoke', 'pilot'))
    parser.add_argument('--training-budget-seconds', type=float)
    parser.add_argument('--evaluation-budget-seconds', type=float)
    parser.add_argument('--budget-reason')
    args = parser.parse_args()
    command = command_for(args)
    with campaign_lock(args.output):
        subprocess.run(command, cwd=ROOT, check=True)
        status = json.loads((args.output / 'campaign_status.json').read_text())
        if not status['states'] or any(v != 'completed' for v in status['states'].values()):
            print('Campaign is partial: inspect status and resume explicitly. No reserved test was run.',
                  file=sys.stderr)
            return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())

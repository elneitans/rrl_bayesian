"""Preserve verified historical sources and export tools, without copying results."""
import argparse
import shutil
from pathlib import Path

from experiments.pong_analysis_common import ROOT, sha, write_new


def preserve(campaign, output):
    from experiments.compare_pong import load_frozen
    manifest = load_frozen(campaign)  # original exact source-set and protocol checks
    expected = manifest['source_hashes']
    for relative, checksum in expected.items():
        if sha(ROOT / relative) != checksum:
            raise ValueError(f'Historical source changed: {relative}')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('bayesian_rrtl', 'experiments', 'configs', 'slurm'):
        shutil.copytree(ROOT / name, output / name, ignore=shutil.ignore_patterns('__pycache__'))
    for path in ROOT.iterdir():
        if path.is_file() and path.suffix in ('.py', '.txt', '.yml', '.toml'):
            shutil.copy2(path, output / path.name)
    for relative, checksum in expected.items():
        if sha(output / relative) != checksum:
            raise ValueError(f'Historical copy verification failed: {relative}')
    write_new(output / 'historical-preservation.json', {
        'protocol_sha256': sha(Path(campaign) / 'protocol.json'), 'source_hashes': expected,
        'source_campaign': str(Path(campaign).resolve()),
        'note': 'Preserve original Linux dependency environment and ROM separately.'})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(preserve(args.campaign, args.output))


if __name__ == '__main__':
    main()

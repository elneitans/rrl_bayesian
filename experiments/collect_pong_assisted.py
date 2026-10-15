"""Assisted collection under the new common contract and validated full-control gate."""
import argparse
from pathlib import Path
from experiments.collect_pong_dataset import collect
from experiments.pong_analysis_runtime import extension_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--pipeline', choices=('visual', 'ocatari'), default='ocatari')
    parser.add_argument('--smoke', action='store_true')
    extension_arguments(parser)
    args = parser.parse_args()
    print(collect(args.analysis, args.pipeline, assisted=True, smoke=args.smoke,
                  extend_seconds=args.extend_seconds, extension_id=args.extension_id))


if __name__ == '__main__':
    main()

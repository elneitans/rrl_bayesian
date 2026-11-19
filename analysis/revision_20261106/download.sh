#!/usr/bin/env bash
# Capture frontiers first, then their immutable dependencies. No rsync --delete.
set -euo pipefail
remote=${RRL_REVISION_REMOTE:-natan.brugueras@kraken.ing.puc.cl:archive/rrl_revision_20261106_02/campaign/}
destination=${RRL_REVISION_LOCAL:-results_cluster/rrtl_revision_20261106_02/campaign/}
mkdir -p "$destination"
rsync -a --include='*/' --include='status.json' --exclude='*' "$remote" "$destination"
rsync -a --exclude='status.json' --exclude='*.pkl' --exclude='.lock' --exclude='*.tmp' \
    --exclude='holdouts/' "$remote" "$destination"

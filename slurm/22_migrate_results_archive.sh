#!/usr/bin/env bash
# Login-node file operations only; no Python, no experiment submission.
set -euo pipefail
root="${RRL_ROOT:-$HOME/projects/rrl_bayesian}"
destination="$HOME/archive/rrl_bayesian_results"
cd "$root"
[[ -d results && ! -L results ]] || { echo 'Expected real results directory'; exit 2; }
while read -r job name; do
  if [[ "$name" == rrl-* && " ${RRL_EXTERNAL_WORKER_IDS:-} " != *" $job "* ]]; then
    echo "RRL job $job may access source results; refuse migration"; exit 2
  fi
done < <(squeue -h -r -u "$USER" -t RUNNING,COMPLETING,SUSPENDED -o '%i %j')
mkdir -p "$destination"
[[ "$(realpath results)" != "$(realpath "$destination")" ]] || exit 2
exec 9>"$destination/.migration.lock"
flock -n 9 || exit 2
rsync -a results/ "$destination/"
verification=$(rsync -acn --itemize-changes results/ "$destination/")
[[ -z "$verification" ]] || { printf '%s\n' "$verification"; exit 2; }
echo 'Checksum comparison passed; removing only source files copied to archive.'
rsync -ac --remove-source-files results/ "$destination/"
find results -depth -type d -empty -delete
[[ ! -e results ]] || { echo 'Source contains remaining entries; inspect before symlinking'; exit 2; }
ln -s "$destination" results
echo "Migration complete: $root/results -> $destination"

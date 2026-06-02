#!/usr/bin/env bash
# Re-run every topic notebook headless as a runnability gate.
# Executes in place so committed notebooks keep their embedded graphics.
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
failed=()
count=0
while IFS= read -r -d '' nb; do
  count=$((count + 1))
  echo ">> executing: $nb"
  if ! uv run jupyter nbconvert --to notebook --execute --inplace "$nb"; then
    failed+=("$nb")
  fi
done < <(find "$root/topics" -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' -print0)

if [ "${#failed[@]}" -gt 0 ]; then
  printf '\nFAILED (%d):\n' "${#failed[@]}"
  printf '  %s\n' "${failed[@]}"
  exit 1
fi
echo ""
echo "All ${count} notebooks executed cleanly."

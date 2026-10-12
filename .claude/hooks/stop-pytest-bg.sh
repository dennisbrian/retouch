#!/bin/bash
# Async Stop hook (asyncRewake): run the fast pytest subset in the background
# when uncommitted .py changes exist. Each distinct change set is tested once,
# runs never overlap, and a failure exits 2 so Claude is woken with the output.
# State lives in .git/claude-stop-pytest/ so nothing shows up in the working tree.

cat >/dev/null  # drain hook stdin

cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

py_changes=$(git status --porcelain -- '*.py' 2>/dev/null | grep -v '__pycache__/')
[ -z "$py_changes" ] && exit 0

state_dir="$(git rev-parse --absolute-git-dir)/claude-stop-pytest"
mkdir -p "$state_dir" || exit 0

fingerprint=$( {
  git diff HEAD -- '*.py'
  git ls-files --others --exclude-standard -- '*.py' | while read -r f; do
    echo "== $f"; cat "$f"
  done
} 2>/dev/null | shasum | cut -d' ' -f1 )

[ "$fingerprint" = "$(cat "$state_dir/last_fingerprint" 2>/dev/null)" ] && exit 0

lock="$state_dir/lock"
if ! mkdir "$lock" 2>/dev/null; then
  pid=$(cat "$lock/pid" 2>/dev/null)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    exit 0  # a run is already in progress; the next stop will pick up newer changes
  fi
  rm -rf "$lock"
  mkdir "$lock" 2>/dev/null || exit 0
fi
echo $$ > "$lock/pid"
trap 'rm -rf "$lock"' EXIT

echo "$fingerprint" > "$state_dir/last_fingerprint"
log="$state_dir/last_run.log"

py=.venv/bin/python
[ -x "$py" ] || py=python3

"$py" -m pytest -x -q --no-header -p no:cacheprovider --tb=short -rfE \
  -m "not slow and not native" > "$log" 2>&1
rc=$?

# 0 = passed, 5 = no tests collected; anything else is worth reporting
if [ "$rc" -ne 0 ] && [ "$rc" -ne 5 ]; then
  {
    echo "Background pytest (fast subset) FAILED (exit $rc) for uncommitted .py changes:"
    echo "$py_changes" | head -10
    echo "---"
    tail -40 "$log"
    echo "--- full log: $log"
  } >&2
  exit 2
fi
exit 0

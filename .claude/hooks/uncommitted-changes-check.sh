#!/bin/bash
# Stop hook: surface uncommitted changes when Claude finishes a turn,
# so a fix left uncommitted (or another session's WIP) is visible before commit.

cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

changes=$(git status --porcelain 2>/dev/null \
  | grep -vE '^.. "?(.*__pycache__/|test_output/|\.claude/worktrees/)' \
  | head -15)

[ -z "$changes" ] && exit 0

count=$(printf '%s\n' "$changes" | wc -l | tr -d ' ')
branch=$(git branch --show-current 2>/dev/null)
msg="Uncommitted changes on ${branch} (${count} shown):
${changes}"

jq -n --arg m "$msg" '{systemMessage: $m}'
exit 0

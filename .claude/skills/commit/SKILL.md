---
name: commit
description: Create a git commit safely, avoiding heredoc quoting failures. Use whenever the user asks to commit changes.
---

# Commit

1. Run `git status` and `git diff` (staged + unstaged) to see what will be committed. Show the user what's changing.
2. Draft a commit message following this repo's format (see CLAUDE.md § Git Workflow): `<type>(<scope>): <subject>`, optional body/footer.
3. Write the message to `/tmp/commit-msg.txt` with the Write tool. Never use heredocs, nested backticks, or inline `-m` strings with quotes/apostrophes.
4. Stage the relevant files by name (not `-A` or `.`).
5. Run `git commit -F /tmp/commit-msg.txt`.
6. Run `git status` to confirm the commit succeeded.

Only commit when the user has explicitly asked for it.

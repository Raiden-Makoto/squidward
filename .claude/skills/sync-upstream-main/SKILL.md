---
name: sync-upstream-main
description: Merge upstream sgl-project/sglang changes into our fork's local main branch and push. Use when the user asks to sync main with upstream, update main from upstream, pull in upstream changes, or merge upstream/main.
---

# Sync upstream into local main

Brings new commits from `upstream` (sgl-project/sglang) into our fork's `main`,
preserving the small fork-only tooling commits (`.cursor/rules/`, `.claude/skills/`)
that sit on top of upstream.

## Remotes (verify with `git remote -v`)

- `upstream` -> sgl-project/sglang (read-only source of truth)
- `origin`   -> Raiden-Makoto/squidward (our fork)

## Worktrees

- `squidward` -> worktree for main (origin)
- `squidward-glm51` -> worktree for RM/glm51

## Workflow

1. Ensure the working tree is clean (`git status`); commit or stash first.
2. `git fetch upstream --prune`
3. `git checkout main`
4. `git merge upstream/main` (merge, NOT rebase -- keeps our tooling commits on top
  and does not rewrite pushed history). Accept the default merge message.
5. If there are conflicts:
  - Our fork only adds files under `.cursor/rules/` and `.claude/skills/`, so conflicts
   are rare and usually trivial. Resolve them, `git add -A`, then `git commit`.
6. `git push origin main`
7. Report the new `main` HEAD and how many commits were merged in
  (`git rev-list --count main@{1}..main`).
8. Attempt to merge `origin/main` into the working branch `RM/glm51`. If there are conflicts, attempt to resolve the simple ones and escalate other conflicts to the user

## Notes

- Do NOT rebase `main` onto upstream -- that rewrites our pushed history.
- If `git fetch` fails with a TLS cert error, prefix the command with `GIT_SSL_NO_VERIFY=1`.
- HEAD-ordering invariant (see `remote-box-rules.mdc`): only advance a remote box to a
commit local already has -- never let the box get ahead of local.


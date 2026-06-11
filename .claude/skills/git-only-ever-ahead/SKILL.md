---
name: git-only-ever-ahead
description: Enforce the branch ordering invariant — origin/main never behind upstream/main, and feature branches never behind origin/main (only ever ahead). Reconciles via merge, fixes the recurring dual-environment stale-ref / phantom-revert issues, and pushes safely. Use when syncing branches, before/after pushing, when a branch looks behind, or when local git state disagrees with the remote.
can_invoke: user_and_claude
---

# Git: Only Ever Ahead, Never Behind

## Invariant (must always hold)
- `origin/main` MUST contain `upstream/main` — never behind upstream.
- Feature branches MUST contain `origin/main` (and thus `upstream/main`) — never behind either. A branch = `origin/main` + our work.
- Restated: a branch may be AHEAD of or EQUAL to its base, NEVER behind.

## Step 1 — Check
```bash
git fetch origin upstream --prune --quiet
git rev-list --left-right --count upstream/main...origin/main   # left = origin/main behind upstream; MUST be 0
git rev-list --left-right --count origin/main...HEAD            # left = branch behind origin/main; MUST be 0
git rev-list --left-right --count upstream/main...HEAD          # left = branch behind upstream; MUST be 0
```
Left number = commits the base has that we lack. Any nonzero left = VIOLATION → fix it.

## Step 2 — Fix when behind
Merge (do NOT rebase — rebasing rewrites already-pushed hashes and causes divergence).
```bash
# origin/main behind upstream:
git checkout -B main origin/main && git merge upstream/main --no-edit && git push origin main
# feature branch behind base:
git checkout <feature> && git merge main --no-edit
```
Known conflict: `utilities/run_dsv4.sh` `DP_MODE` — keep HEAD's `DP_MODE="${DP_MODE:-off}"` (DP off, opt-in via `DP_MODE=tp8dp8`).

## Step 3 — Push safely
```bash
git push -u origin <feature>          # normal case
```
Force-push ONLY a feature branch, ONLY with `--force-with-lease`, and ONLY after confirming the remote has no unique commits:
```bash
comm -23 <(git log origin/<branch> --format='%s'|sort -u) <(git log HEAD --format='%s'|sort -u)  # empty = safe
git push --force-with-lease origin <feature>
```
NEVER force-push `main`/`master`.

## Step 4 — Verify (all left numbers must be 0)
Re-run Step 1, plus `git rev-list --left-right --count origin/<feature>...HEAD` (expect `0   0`).

## Dual-environment gotchas (this workspace flips mounts; local state lies — trust the remote)
- **Stale tracking ref:** `git push` says "Everything up-to-date" or counts look wrong after a push. The push usually LANDED. Get the TRUE remote SHA with `git ls-remote origin <branch>`. If the local tracking ref is stuck, fix it directly: `git update-ref refs/remotes/origin/<branch> <true_sha>`.
- **Phantom-reverted working tree:** files on disk look reverted but HEAD has the commit → disk is stale. Restore: `git restore --source=HEAD --worktree -- <path>`.
- **Rewritten hashes:** if `branch vs remote` shows large divergence with identical commit *subjects*, an earlier rebase rewrote hashes; the `comm` check above confirms no unique remote work, then force-with-lease.

## Box rule
Commit/push from LOCAL first; the box only ever pulls. The box must NEVER be ahead of local.

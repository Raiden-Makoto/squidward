---
name: sglang-pr-workflow
description: Open and maintain a clean upstream PR to sgl-project/sglang from the Raiden-Makoto/squidward fork — branch hygiene, single-commit extraction, pre-commit lint, the real PR title/body conventions, draft-CI behavior, and the force-push/squash gotchas. Use whenever creating, updating, squashing, or reviewing a PR to upstream sglang.
---

# SGLang upstream PR workflow

Conventions + gotchas for contributing from this fork. Remotes:
`origin` = Raiden-Makoto/squidward (fork), `upstream` = sgl-project/sglang.

## 1. Branch off `upstream/main` — NOT fork `main`

Fork `main` carries fork-only tooling (`.cursor/rules/`, `.claude/skills/`, run
scripts) that must NOT appear in the PR diff. Always:
```
git fetch upstream main
git checkout -b RM/<feature> upstream/main
```
Then bring in ONLY the changed files (e.g. `git checkout <workbranch> -- path/...`)
and verify `git diff --stat upstream/main` shows just those files.

## 2. One clean commit

Squash the work into a single commit. The commit *message* is a normal
conventional-commit-style body (`perf(dsa): ...`) — that's fine for the commit,
but it is NOT the PR title (see §4).

## 3. Lint with pre-commit before pushing

```
pre-commit run --files <changed files>
```
`black-jupyter` will reformat; re-stage and fold into the commit. NOTE: the
`reject CI-registered tests inside the sglang package` hook may "fail" on a
gitignored local litter file (`python/sglang/jit_kernel/tests/test_triton_store_cache_local.py`)
— that's a local-only false positive, not part of any PR. Ignore it.

## 4. PR title = `[Tag] [Tag] plain description` (NOT a commit string)

Real sglang titles use bracketed tags + a plain sentence:
`[AMD] [GLM5] skip redundant -inf pre-fill of HIP indexer MQA-logits`,
`[misc] Make NaN-logit sanitization opt-in (default off)`.
Do NOT use `perf(dsa): ...` as the PR title. Tags: `[AMD]`, `[GLM5]`, `[Fix]`,
`[Perf]`, `[CI]`, `[Spec]`, `[misc]`, etc.

## 5. PR body = the sglang template, numbers only

Sections: `## Motivation`, `## Modifications`, `## Accuracy Tests`,
`## Speed Benchmarks`, `## Checklist`, `## Review and Merge Process`.
- Lead with numbers; no prose verdicts/interpretation.
- Speed = **e2e `sglang.bench_serving`** tables (baseline vs PR, per concurrency,
  median TTFT/ITL/E2EL/throughput). Do NOT include isolated-kernel microbench
  tables — they don't belong in the PR body.
- Accuracy = GSM8K before/after. An `Invalid` of ~0.001-0.002 is parse noise;
  drop the column rather than invite questions if accuracy is the headline.
- Do NOT invent requirements (e.g. don't claim a registered test is needed
  unless a reviewer asks).

## 6. Open as a draft

```
gh pr create --repo sgl-project/sglang --draft --base main \
  --head Raiden-Makoto:RM/<feature> --title "[AMD] [GLM5] ..." \
  --body-file <body.md>
```
On a DRAFT (or unapproved fork PR), the `pr-gate` / `call-gate` and `*-finish`
jobs fire and **fail fast (3-7s)** while every real test matrix shows `skipping`
— that's expected gate plumbing, NOT real CI burning compute. Real CI only runs
after marking ready + a maintainer trigger.

## 7. CI triggers + reading failures

- Rerun bot comments (need author/maintainer perms): `/rerun-failed-ci`,
  `/tag-and-rerun-ci`, `/tag-run-ci-label`. Reruns increment the run's *attempt*
  (same run id, same `createdAt`) — `gh run list` won't show a new row.
- Log-zip download (`gh run view --log-failed`) often fails here with a TLS
  cert error. Fallback: `curl -ksL -H "Authorization: Bearer $(gh auth token)"
  https://api.github.com/repos/sgl-project/sglang/actions/jobs/<job_id>/logs`.
- Triage before blaming the PR: AMD/HIP changes can't break NPU/CUDA-only jobs;
  disaggregation RDMA timeouts and NPU throughput-threshold failures are infra
  flakes. Check whether the same job passes on recent `main`.

## 8. Force-push / squash gotchas

- `--force-with-lease` (bare) fails with "stale info" because `git fetch <branch>`
  only writes `FETCH_HEAD`, not the `origin/<branch>` tracking ref. Get the TRUE
  remote SHA via `git ls-remote origin refs/heads/<branch>` and pin it:
  `git push --force-with-lease=<branch>:<full-sha> origin <branch>`.
- To squash a pushed branch: `git reset --soft <upstream-parent> && git commit`
  (re-write message via heredoc), then the pinned force-push. Do NOT `--amend` a
  commit that's already pushed unless intentionally force-pushing.
- Only force-push your own feature branch, never `main`/`master`.

## 9. Don't over-act

Create the branch / run pre-commit / etc. ONLY what was asked. Do not open a PR,
push, or mark ready unless explicitly told. "Create the branch" ≠ "open a PR".

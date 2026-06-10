#!/usr/bin/env bash

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <branch-name>"
  exit 1
fi

BRANCH_NAME="$1"

# Validate branch name: must not contain whitespace (spaces, tabs, etc.)
if [[ "$BRANCH_NAME" =~ [[:space:]] ]]; then
  echo "Error: Branch name \"$BRANCH_NAME\" must not contain whitespace."
  exit 1
fi

# Load github credentials from .env if needed
if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi

# Check if branch exists locally
if git rev-parse --verify "$BRANCH_NAME" >/dev/null 2>&1; then
  echo "Deleting local branch: $BRANCH_NAME"
  git branch -d "$BRANCH_NAME" || {
    echo "Failed to delete local branch: $BRANCH_NAME. It may not be fully merged."
    exit 1
  }
else
  echo "Local branch $BRANCH_NAME does not exist. Skipping local delete."
fi

echo "Deleting remote branch: $BRANCH_NAME"
git push origin --delete "$BRANCH_NAME"

echo "Branch $BRANCH_NAME deleted locally and from remote (origin)."
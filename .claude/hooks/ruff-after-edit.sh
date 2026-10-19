#!/usr/bin/env bash
file_path=$(jq -r '.tool_input.file_path // empty')
[[ "$file_path" == *.py && -f "$file_path" ]] || exit 0

if ! command -v uv >/dev/null 2>&1; then
  echo "ruff hook: uv not found on PATH, skipping lint/format" >&2
  exit 1  # non-blocking: surfaced to the user, Claude carries on
fi

# Use the project's pinned ruff (uv.lock), not uvx's latest.
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 1

# Fix first, then format (ruff's recommended order); keep what couldn't be fixed.
lint_out=$(uv run ruff check --fix --quiet "$file_path" 2>&1)
lint_status=$?
uv run ruff format --quiet "$file_path" >&2

if [[ $lint_status -ne 0 ]]; then
  echo "ruff: unfixable issues in $file_path:" >&2
  echo "$lint_out" >&2
  exit 0  # 2 -> stderr is fed back to Claude so it can fix them,
          # 0 still allows the hook to pass for prototyping
fi
exit 0

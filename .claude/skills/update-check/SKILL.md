---
name: update-check
description: After updating a core function and writing tests for it, run the following checks to ensure that the code is still valid and passes all tests.
---

Run these checks in order and report the results:

1. `uv run ruff check .` (lint)
2. `uv run ruff format --check .` (format verification)
3. `uv run pytest` (tests)

If any check fails, fix the issue and rerun that check before continuing. Report a summary: which checks passed, which failed, and the current version.
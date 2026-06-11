# AGENTS.md example

## Skills

For reviews of pull requests, diffs, commits, or changed files, use the `$code-review` skill.

## Repository-specific review rules

- Treat authentication, authorization, data-loss, and PHI/PII exposure as blocking.
- Prefer targeted tests before running the entire suite.
- Do not modify code during a review unless explicitly asked.
- Follow repository commands documented in `README.md` and `pyproject.toml`.

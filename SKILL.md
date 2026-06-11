---
name: code-review
description: Review code changes, pull requests, diffs, or selected files for correctness, security, reliability, maintainability, performance, and test coverage. Use when asked to review code, inspect a PR or diff, find bugs, assess implementation risk, or provide structured engineering feedback. Do not use for purely stylistic rewriting or when the user only wants new code written.
---

# Code Review Skill

## Objective

Perform a high-signal engineering review that identifies defects and meaningful risks without flooding the user with low-value style comments.

Prioritize:

1. Correctness and regressions
2. Security and privacy
3. Data integrity
4. Reliability and failure handling
5. Concurrency and async behavior
6. API and backward compatibility
7. Performance and scalability
8. Test coverage
9. Maintainability
10. Style only when it affects clarity or correctness

## Review scope

Before reviewing:

1. Read repository-level and directory-level `AGENTS.md` files.
2. Inspect the requested diff, pull request, commit range, or files.
3. Read nearby code needed to understand contracts and call paths.
4. Inspect relevant tests, schemas, configuration, and dependency changes.
5. Identify the intended behavior from the task, issue, PR description, or code.
6. Do not modify files unless the user explicitly asks for fixes.

If the review target is ambiguous, infer the narrowest reasonable scope from the current changes and state the assumption.

## Review workflow

### 1. Understand the change

Determine:

- What behavior is being added or changed?
- Which public interfaces, data models, or persistence paths are affected?
- What assumptions does the implementation make?
- What could fail at runtime?
- What existing behavior might regress?

### 2. Inspect the diff first

Start with changed lines, then trace into surrounding code only where needed.

Do not review the entire repository unless the user asks for a broad audit.

### 3. Validate behavior

Check:

- Boundary conditions
- Null, empty, malformed, and unexpected inputs
- Error paths and exception handling
- Cleanup and resource lifecycle
- Transaction boundaries and partial failure
- Retry behavior and idempotency
- Timeouts and cancellation
- Async blocking and race conditions
- Serialization and type conversions
- Database query correctness
- API status codes and response contracts
- Configuration defaults and environment differences
- Logging without leaking secrets or sensitive data

### 4. Check security

Look for:

- Hardcoded secrets, tokens, credentials, or private keys
- Injection risks: SQL, shell, template, prompt, path, or code injection
- Broken authentication or authorization
- Insecure direct object references
- Unsafe deserialization
- Path traversal
- SSRF
- Sensitive data exposure
- Missing input validation
- Overly broad permissions
- Dependency or supply-chain risk
- PHI, PII, or confidential data appearing in logs, prompts, traces, or errors

Do not reproduce real secrets in review output. Refer to the file and masked value.

### 5. Check tests

Determine whether tests cover:

- The primary success path
- Important edge cases
- Failure behavior
- Regression-prone behavior
- Security-sensitive behavior
- Backward compatibility

Do not demand tests for trivial mechanical changes. Explain the specific untested failure the proposed test would catch.

### 6. Run verification when safe

Use the repository's existing commands where available:

- Unit tests
- Targeted integration tests
- Type checking
- Linting
- Static analysis
- Build or compile checks

Prefer targeted commands before full-suite commands.

Do not install new tools, modify dependencies, access external systems, or run destructive commands without explicit user approval.

If verification cannot be run, state why.

## Finding criteria

Report a finding only when all are true:

- It is caused by or materially exposed by the reviewed change.
- It has a plausible failure scenario.
- It is actionable.
- The confidence is sufficient to defend the claim.

Avoid:

- Pure preferences
- Hypothetical issues with no realistic trigger
- Repeating the same root cause multiple times
- Compliments disguised as findings
- Large rewrites when a focused fix is sufficient

## Severity levels

Use these levels:

- **Critical**: Exploitable security issue, irreversible data loss, major outage risk, or severe compliance exposure.
- **High**: Likely production failure, authorization bypass, corruption, or major regression.
- **Medium**: Real bug or reliability problem with a narrower trigger or limited blast radius.
- **Low**: Minor defect or maintainability issue worth fixing but unlikely to cause substantial harm.

Do not inflate severity.

## Required output format

Start with a verdict:

- `APPROVE`
- `REQUEST CHANGES`
- `COMMENT`

Then provide findings ordered by severity.

For every finding use:

### [Severity] Short title

- **Location:** `path/to/file.py:line`
- **Problem:** What is wrong.
- **Failure scenario:** The concrete condition under which it fails.
- **Impact:** What happens to users, systems, or data.
- **Recommendation:** The smallest safe correction.
- **Confidence:** High, Medium, or Low.

After findings, include:

## Verification

- Commands run and results
- Commands not run and why

## Summary

- 2–5 sentences describing the change and overall risk
- If there are no findings, explicitly say: `No blocking issues found.`

## Inline-review mode

When the user asks for concise PR comments, output only actionable findings. Each comment must:

- Point to the narrowest relevant line range
- Explain the failure, not merely state a rule
- Avoid unnecessary background
- Suggest a concrete fix when practical

## Fix mode

Only when explicitly asked to fix the issues:

1. Make the smallest coherent change.
2. Add or update regression tests.
3. Run targeted verification.
4. Summarize changed files and residual risks.
5. Never silently change public contracts or data schemas.

## Language-specific reminders

### Python / FastAPI

Check:

- Blocking I/O inside `async def`
- Missing timeouts
- Incorrect dependency scopes
- Mutable defaults
- Broad exception handling
- Unawaited coroutines
- Session and transaction lifecycle
- Pydantic validation and serialization differences
- Response-model and status-code mismatches

### SQL / data pipelines

Check:

- Join cardinality
- Duplicate amplification
- Null handling
- Incremental-load boundaries
- Time-zone conversions
- Non-deterministic ordering
- Unsafe dynamic SQL
- Merge/upsert correctness
- Schema evolution and backward compatibility

### LLM / RAG systems

Check:

- Prompt injection boundaries
- Untrusted retrieved content being treated as instructions
- Secrets or sensitive data sent to models
- Missing output validation
- Unsupported factual claims
- Retrieval filtering and tenant isolation
- Token and context limits
- Retry amplification and runaway cost
- Logging of prompts, responses, PHI, or PII

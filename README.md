# Code Review Skill Demo

## Install in one repository

Copy the `.agents/skills/code-review` directory into the root of your repository.

Expected structure:

```text
your-repo/
├── AGENTS.md
└── .agents/
    └── skills/
        └── code-review/
            ├── SKILL.md
            └── references/
                └── review-checklist.md
```

## Use

In Codex:

```text
$code-review Review the current git diff.
```

Other demo prompts:

```text
$code-review Review the latest commit. Focus on correctness, security, and missing regression tests.
```

```text
$code-review Review the changed FastAPI and SQL files. Do not modify anything.
```

## Demo flow

1. Open a repository containing a small intentional bug.
2. Ask Codex to review the diff without the skill and save the response.
3. Install this skill.
4. Run the same review using `$code-review`.
5. Compare finding quality, severity, evidence, and verification.
6. Fix one issue and rerun the review.

The included eval CSV contains positive and negative trigger examples.

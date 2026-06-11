# Code Review Reference Checklist

Use this as a secondary checklist, not as a requirement to comment on every category.

## Correctness
- Intended behavior matches implementation
- Edge cases and invalid states handled
- Return values and exceptions match callers' expectations
- No off-by-one, rounding, ordering, or time-zone errors

## Security
- Authentication and authorization enforced at the correct boundary
- Inputs validated before use
- No injection, traversal, SSRF, or unsafe deserialization
- No secrets or sensitive data exposed

## Reliability
- Timeouts, cancellation, retries, and idempotency are appropriate
- Partial failures do not leave inconsistent state
- Resources are closed
- Async code does not block the event loop

## Data
- Transactions are correctly scoped
- Joins do not multiply rows unexpectedly
- Nulls, duplicates, and schema changes are considered
- Migrations are safe and reversible where practical

## APIs
- Backward compatibility considered
- Status codes and error payloads are consistent
- Validation occurs at the boundary
- Pagination and limits are enforced

## Tests
- Tests demonstrate the bug would be caught
- Edge and failure cases are included where meaningful
- Tests are deterministic and isolated

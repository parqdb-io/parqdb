# ParqDB RFCs

RFCs describe substantial changes before implementation. They are reviewed as
standalone pull requests so that design feedback remains separate from code
review.

Use an RFC for a change that affects public behavior, persistent formats,
cross-module contracts, or a major execution path. Local refactoring, bug
fixes, and narrowly scoped performance improvements can use the normal pull
request process.

## Process

1. Add `YYYYMMDD-short-title.md` to this directory in a pull request containing
   no implementation changes.
2. Discuss and revise the proposal in that pull request.
3. Merge the RFC when its major tradeoffs and unresolved implementation risks
   are understood. A merged RFC is accepted, but not necessarily implemented.
4. Track implementation in a linked issue and one or more code pull requests.
5. Replace a materially outdated RFC with a new RFC instead of rewriting its
   design history.

Architecture decision records in [`../decisions`](../decisions/) document
short, accepted decisions. RFCs contain the motivation, alternatives, and
reference design needed to review a substantial change before accepting it.

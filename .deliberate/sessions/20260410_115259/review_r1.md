**REJECT**

There is no reviewable plan here. `Error: Reached max turns (6)` is a tooling failure, not an implementation proposal, so I cannot assess correctness, completeness, architecture, risks, or testing.

What is missing and must be provided before this is implementable:

1. Scope for Milestones 1 and 2.
The plan needs to say exactly which heuristics are being changed, removed, or preserved, and where they live in the system.

2. Success criteria.
“Heuristics are okay if they’re needed for performance” is too vague. The plan needs measurable goals:
- what behavior is currently wrong
- what “fixed” means
- what latency / throughput / cost constraints justify heuristics

3. Decision framework.
A good plan should separate:
- correctness-critical logic that should be deterministic
- performance optimizations where heuristics are acceptable
- fallback behavior when heuristics misfire

4. Architectural impact.
The plan should explain whether Milestones 1 and 2 are being patched locally or whether heuristic logic is being centralized behind a strategy/config/evaluator layer. Without that, there is high risk of duplicated logic and inconsistent behavior.

5. Risk analysis.
At minimum, the plan should cover:
- false positives / false negatives introduced by heuristics
- regressions at milestone boundaries
- data-dependent edge cases
- observability needed to validate heuristic decisions in production

6. Testing strategy.
This is the biggest omission. A credible plan needs:
- unit tests for each heuristic rule
- regression tests for known bad cases
- benchmark/perf tests proving the heuristic is actually needed
- feature-flag or shadow-mode validation if behavior is user-visible or high risk

What I would require in a revised plan:

1. Problem statement for each milestone:
- current behavior
- desired behavior
- why the current heuristic fails

2. Proposed change per heuristic:
- keep, replace, tighten, or remove
- rationale
- expected correctness/performance tradeoff

3. Implementation shape:
- files/modules affected
- whether heuristic evaluation is centralized
- config knobs / thresholds and where they live

4. Validation:
- exact test cases
- benchmark dataset or scenarios
- rollout/monitoring plan if applicable

If you can paste the actual Claude plan, I can do a proper architectural review. As written, this is not ready for implementation review.
**APPROVE**

The revised plan is ready for `/ce-work`. It now makes U7 a real KD8 gate: projected assembled cost is a hard GO precondition, includes second field, drag, lambda ping-pong, bed/plasticity/wetting bounds, substep multiplier, U8 cost if triggered, and a buffer ledger against 9/16 buffers ([plan](/Users/cxc/Github/coffee-sim/docs/plans/2026-06-20-001-feat-pbmpm-prototype-plan.md:214)).

Prior REVISE items are honestly addressed:
- Candidate D is correctly scoped as the two-field DensU density-target proxy, not compliant projection; repo matches `dbg.w` and the setters ([pressure.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/pressure.wgsl:497), [mod.rs](/Users/cxc/Github/coffee-sim/src/solvers/twofield/mod.rs:904)).
- The comparison config is pinned outside `web.rs::setup_for`, which matters because twofield really gets wider nozzle/lower cap ([web.rs](/Users/cxc/Github/coffee-sim/src/web.rs:465)).
- PB-MPM `D` is correctly per-particle float state, distinct from fixed-point grid lanes, with separate U6 probes.
- Transfer stencil is explicitly 3x3x3 / 27-node quadratic, matching twofield ([transfers.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/transfers.wgsl:118)).
- Emission is now a U2 dependency before constraint visual validation; the twofield water-pool/emission pattern exists as cited.
- XPBD reorder reuse is treated as a ported, measured subsystem, not free P2G magic.
- U7 makes bounce-vs-D and projected assembled cost pass/fail decision inputs, with cap-hit/no-popcorn/conservation guards.

The sequencing is sound: U1 scaffold, U2 emission, U3 transfers, U4 constraint, U5 BC/restitution, then U6/U7 quantified gates. The visual-first posture is acceptable because Phase B now contains falsifiable, non-gameable backstops for both bounce and assembled-system risk.

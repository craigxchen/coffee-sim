# MPM v2 calibration guide

MPM v2 exposes physical and constitutive controls so a calibration can be judged against a
specific coffee, grinder, filter, recipe, and visual reference. The shipped values are broad
starting priors. They are not claims that every V60 brew has the same permeability, retained-water
ratio, swelling, drawdown, TDS, or extraction yield.

## What public data can and cannot constrain

- A published V60 protocol reports a liquid-retention ratio of approximately **2.1 g water per
  gram dry coffee**. It is one protocol, not a universal material property, and its accounting
  includes a pre-wetted paper filter and procedure-specific dead volume. MPM v2 therefore treats
  2.1 g/g as a visible *whole-bed* prior and reports separately how much can be assigned to bound
  water, capillary pore water, and filter water. An unattainable request remains visible rather
  than being hidden by a clamp. [Protocol source](https://pmc.ncbi.nlm.nih.gov/articles/PMC10486461/)
- Hargarten, Kuhn, and Briesen measured roughly **15% final particle-diameter increase**, with
  substantial particle scatter; approximately **60–80% of the final increase was reached after
  30 s**, and swelling was effectively complete after about 4 min. Earlier methods reported
  materially smaller changes, partly because erosion and attached fines confounded particle-size
  measurements. The UI therefore spans 0–15% diameter increase, starts at a deliberately
  non-extreme 8%, and uses an independently adjustable uptake rate. No regression test asserts a
  chosen swelling magnitude. [Open-access paper](https://mediatum.ub.tum.de/doc/1580140/document.pdf)
- The UC Davis public drip-brew dataset deliberately spans target levels of **1.0, 1.25, and 1.5%
  TDS** and **16, 20, and 24% extraction yield** across 87, 90, and 93 °C brews. These are useful
  endpoint bands for diagnostics, not V60 transient data: the experiments used a commercial batch
  brewer and adjusted grind/flow to hit the factorial targets. [CC0 dataset](https://zenodo.org/records/7542610)
- The public TUM espresso repository contains time-dependent component extraction at varied flow,
  temperature, and particle size. It is useful for checking monotonic trends and curve shape, but
  espresso pressure and bed geometry make direct rate fitting to a gravity-driven V60 unjustified.
  [Kinetics repository](https://data.mendeley.com/datasets/y2tz67f6ry/1)

There is no public dataset here that jointly records a V60's time-resolved three-dimensional bed
shape, saturation, flow, temperature, retained inventories, and extraction. Visual calibration is
therefore an intended workflow, not a fallback hidden behind fixed constants.

For orientation only, a 15 g / 250 g recipe retaining 2.1 g/g leaves about 218.5 g of delivered
water outside the retained inventories. Extracting 16–24% of the dry dose then gives roughly
2.4–3.6 g solute, which is naturally in the vicinity of the UC Davis drip-strength band. This
mass-balance coincidence is a useful cross-check; it is not evidence that the independent studies
describe the same bean, brewer, or transient process.

## Inventory definitions

The retained-water slider is a recipe-level target. For dry dose `m_dry`, the requested liquid
mass is

```text
m_target = retention_ratio * m_dry
```

The solver accounts for it as distinct inventories:

```text
whole-bed target = intragranular bound water
                 + capillary/interstitial pore water
                 + filter-paper water
                 + explicitly reported unmet target
```

Particle swelling is not another name for retention. With bound-water saturation `s` and the
diameter control `a`, structural volume follows

```text
d_eff / d_dry = 1 + s*a
V_struct / V_dry = (1 + s*a)^3
```

Water beyond that geometric change remains water mass and pore/filter inventory. Numerical solid
sample count does not change recipe mass or represented bulk volume; quadrature weights compensate
when the representative workload uniformly thins material points.

## Recommended calibration order

1. Keep the recipe and geometry fixed. Confirm the UI reports the expected delivered water, dry
   dose, particle census, and no inventory/finite-state violations.
2. Calibrate the dry bed with compaction tangent, initial consolidation pressure, numerical shear
   compliance, friction, and full-compaction solid fraction. Judge settling, crater formation, and
   cessation of motion before adding wet effects.
3. Calibrate wetting with bound-water capacity, absorption rate, bloom delay, and swelling. Compare
   the *time course* and final diameter response separately.
4. Calibrate drawdown and retention with intrinsic permeability, unresolved-drag multiplier,
   suction head, reference porosity, Corey exponent, and mobility floor. Check the inventory split,
   not only the final animation.
5. Calibrate paper permeability/capacity against drainage and post-brew dripping. A zero paper
   capacity removes paper storage only; it does not remove paper resistance.
6. Calibrate temperature and the fast/slow extraction rates last. Use TDS and extraction bands as
   plausibility diagnostics while checking that changing temperature, flow, and contact time has
   the expected monotone effect.

Change one subsystem at a time and record the full profile with the recipe and reference video.
Until such a profile is accepted, automated tests protect conservation, boundedness, operator
consistency, monotonic response, and timestep convergence—not a temporary slider value.

## Performance contract

Calibration runs and the browser performance gate are separate. The binding throughput fixture is
a deterministic 15 g / 250 mL late-brew state with exactly 12,000 solid samples and 75,000–80,000
water samples. At 1280×720 DPR 1, one displayed frame advances two 1/60 s steps and renders once.
After warm-up, the reference-device gate is a median frame time at or below 33.3 ms and a real-time
factor of at least 1.0. Optimizations may change scheduling or active work, but not remove the
inventories or constitutive controls being calibrated.

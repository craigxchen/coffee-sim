# Coffee Extraction Model

The simulator uses a reduced two-pool extraction model designed for the
particle-first MPM solver.

## Basis

The model follows the structure used in coffee-extraction literature:

- fast release from fines, broken cells, and grain surfaces
- slower release from intact grain interiors
- advection-dominated transport of dissolved solids in the liquid phase

Primary references:

- Moroney et al., "Modelling of coffee extraction during brewing using
  multiscale methods", Chemical Engineering Science, 2015.
- Moroney et al., "Coffee extraction kinetics in a well mixed system",
  Mathematics in Industry, 2016.
- O'Connell, Moroney, Lee et al., "A multiscale model of coffee extraction",
  PLOS ONE, 2019.

## Solver State

Each bed particle stores:

- pore water mass
- porosity
- permeability
- compaction
- fast extractable solute
- dissolved pore solute
- slow extractable solute
- saturation

Each water particle stores carried solute mass in the spare
`affine.col1.w` lane. Concentration is computed as:

```text
water_concentration = water_solute_mass / water_mass
```

The scalar is conservative particle state, so solute transport follows the
same advection path as water.

## Source Terms

For each wet bed particle, the bed-side extraction pass applies:

```text
drive = max(0, 1 - pore_concentration / max_solute_concentration)
wet_contact = smoothstep(0.03, 0.55, saturation)

fast_flux = k_fast * fast_extractable * wet_contact * drive * dt
slow_flux = k_slow * slow_extractable * wet_contact * drive * dt
```

The fluxes move mass from the fast and slow extractable reservoirs into
dissolved pore solute. Mobile water particles then reserve dissolved pore
solute atomically and carry it downstream.

When water is absorbed into bed pores, its carried solute is transferred back
to the bed pore-solute reservoir with the absorbed water fraction, avoiding
spurious concentration jumps on the remaining particle.

## Current Limits

This is still a coarse-grained model:

- no explicit grind-size distribution yet
- no liquid-phase diffusion or dispersion
- no separate beverage outlet accumulator
- rates are physical defaults, not calibrated against a target brew curve

The browser currently reports active/cup TDS-like ratios from water particles
still in the simulation.

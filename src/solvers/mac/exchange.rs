//! Conservative cell-local transactions. Allocation is simultaneous within each pass:
//! each particle contributes a partition of its inventory, never its entire inventory
//! to every neighboring cell. Thus arbitrary overlap cannot overdraw a donor.

use glam::DVec3;

use crate::models::{extraction, thermal, wetting};

use super::core::MacSimulation;

type Bins = Vec<Vec<(usize, f64)>>;

impl MacSimulation {
    fn exchange_bins(&self) -> (Bins, Bins) {
        let mut water = vec![Vec::new(); self.grid.cells.len()];
        let mut grains = vec![Vec::new(); self.grid.cells.len()];
        for (i, p) in self.water.iter().enumerate() {
            for (c, w) in self.grid.stencil(p.position, None) {
                water[c].push((i, w));
            }
        }
        for (i, p) in self.grains.iter().enumerate() {
            for (c, w) in self.grid.stencil(p.position, None) {
                grains[c].push((i, w));
            }
        }
        (water, grains)
    }

    pub(super) fn exchange(&mut self, dt: f64) {
        if self.grains.is_empty() || self.water.is_empty() {
            return;
        }
        let (water, grains) = self.exchange_bins();
        if self.cfg.absorb_rate > 0.0 {
            self.absorb(dt, &water, &grains);
        }
        if self.cfg.extract_rate > 0.0 {
            self.dissolve(dt, &water, &grains);
        }
        if self.cfg.mac.thermal_exchange {
            self.exchange_heat(dt, &water, &grains);
        }
    }

    fn absorb(&mut self, dt: f64, water: &Bins, grains: &Bins) {
        let rho = self.rho();
        let demand: Vec<f64> = self
            .grains
            .iter()
            .map(|g| {
                wetting::absorb_demand(
                    g.absorbed as f32,
                    g.capacity as f32,
                    self.cfg.absorb_rate,
                    dt as f32,
                ) as f64
            })
            .enumerate()
            .map(|(i, v)| v.min((self.grains[i].capacity - self.grains[i].absorbed).max(0.0)))
            .collect();
        let mut removed = vec![0.0; self.water.len()];
        let mut gained = vec![(0.0, 0.0, 0.0, DVec3::ZERO); self.grains.len()];
        for (ws, gs) in water.iter().zip(grains) {
            let volume: f64 = ws.iter().map(|&(i, w)| w * self.water[i].volume).sum();
            let wanted: f64 = gs.iter().map(|&(i, w)| w * demand[i]).sum();
            if volume <= 0.0 || wanted <= 0.0 {
                continue;
            }
            let take = wanted.min(volume);
            let solute: f64 = ws.iter().map(|&(i, w)| w * self.water[i].solute).sum();
            let heat: f64 = ws.iter().map(|&(i, w)| w * self.water[i].heat).sum();
            let momentum: DVec3 = ws
                .iter()
                .map(|&(i, w)| w * self.water_mass(&self.water[i]) * self.water[i].velocity)
                .sum();
            for &(i, w) in ws {
                removed[i] += w * take / volume;
            }
            for &(i, w) in gs {
                let v = take * w * demand[i] / wanted;
                gained[i].0 += v;
                gained[i].1 += v / volume * solute;
                gained[i].2 += v / volume * heat;
                gained[i].3 += v / volume * momentum;
            }
        }
        for (p, fraction) in self.water.iter_mut().zip(removed) {
            let remaining = (1.0 - fraction).max(0.0);
            p.volume *= remaining;
            p.solute *= remaining;
            p.heat *= remaining;
        }
        for (g, (volume, solute, heat, momentum)) in self.grains.iter_mut().zip(gained) {
            let old_momentum = g.mass(rho) * g.velocity;
            g.absorbed += volume;
            g.retained_solute += solute;
            g.heat += heat;
            g.velocity = (old_momentum + momentum) / g.mass(rho);
        }
    }

    fn dissolve(&mut self, dt: f64, water: &Bins, grains: &Bins) {
        let rho = self.rho();
        let mut lost = vec![[0.0; 3]; self.grains.len()];
        let mut gained = vec![(0.0, 0.0, DVec3::ZERO); self.water.len()];
        let c_sat = (self.mats.c_sat as f64).clamp(0.0, 0.99);
        for (ws, gs) in water.iter().zip(grains) {
            let volume: f64 = ws.iter().map(|&(i, w)| w * self.water[i].volume).sum();
            if volume <= 1e-30 || gs.is_empty() {
                continue;
            }
            let solute: f64 = ws.iter().map(|&(i, w)| w * self.water[i].solute).sum();
            let concentration = solute / (rho * volume + solute);
            let room = (c_sat / (1.0 - c_sat) * rho * volume - solute).max(0.0);
            if room == 0.0 {
                continue;
            }
            let velocity: DVec3 = ws
                .iter()
                .map(|&(i, w)| w * self.water[i].volume * self.water[i].velocity)
                .sum::<DVec3>()
                / volume;
            let mut requests = Vec::with_capacity(gs.len());
            let mut total = 0.0;
            for &(i, w) in gs {
                let g = &self.grains[i];
                let t = g.heat / self.grain_heat_capacity(g).max(1e-30);
                // Nonzero diffusive baseline. Flow enhances surface renewal; it does not
                // turn diffusion off in an immersion brew. Shared kinetics otherwise.
                let renewal = 1.0
                    + extraction::flux_factor(
                        (velocity - g.velocity).length() as f32,
                        self.mats.u_half,
                    ) as f64;
                let factor = self.cfg.extract_rate as f64
                    * renewal
                    * extraction::arrhenius(t as f32, self.mats.ea_over_r, self.mats.t_ref) as f64
                    * extraction::area_factor(self.mats.grain_diameter, self.mats.d_ref) as f64
                    * extraction::driving(concentration as f32, self.mats.c_sat) as f64;
                let release = |pool: f64, rate: f32| {
                    w * (extraction::release(pool as f32, (factor * rate as f64) as f32, dt as f32)
                        as f64)
                        .min(pool)
                };
                // Fast and slow pools represent surface dissolution and interior diffusion.
                // Local liquid contact is sufficient; absorption can be independently off.
                let retained_equilibrium =
                    concentration / (1.0 - concentration).max(1e-12) * rho * g.absorbed;
                let retained_excess = (g.retained_solute - retained_equilibrium).max(0.0);
                let r = [
                    release(g.fast, self.mats.k0_fast),
                    release(g.slow, self.mats.k0_slow),
                    release(retained_excess, self.mats.k0_slow),
                ];
                total += r.iter().sum::<f64>();
                requests.push((i, r));
            }
            if total <= 0.0 {
                continue;
            }
            let scale = (room / total).min(1.0);
            let mut heat = 0.0;
            let mut momentum = DVec3::ZERO;
            for (i, r) in requests {
                let amount = scale * r.iter().sum::<f64>();
                for (a, amount) in r.iter().enumerate() {
                    lost[i][a] += scale * amount;
                }
                let g = &self.grains[i];
                heat += amount * self.mats.cp_grain as f64 * g.heat
                    / self.grain_heat_capacity(g).max(1e-30);
                momentum += amount * g.velocity;
            }
            for &(i, w) in ws {
                let share = w * self.water[i].volume / volume;
                gained[i].0 += share * total * scale;
                gained[i].1 += share * heat;
                gained[i].2 += share * momentum;
            }
        }
        let cp = self.mats.cp_grain as f64;
        let temperatures: Vec<f64> = self
            .grains
            .iter()
            .map(|g| g.heat / self.grain_heat_capacity(g).max(1e-30))
            .collect();
        for ((g, r), temperature) in self.grains.iter_mut().zip(lost).zip(temperatures) {
            g.fast -= r[0];
            g.slow -= r[1];
            g.retained_solute -= r[2];
            g.heat -= r.iter().sum::<f64>() * cp * temperature;
        }
        for (p, (solute, heat, momentum)) in self.water.iter_mut().zip(gained) {
            let old_momentum = (rho * p.volume + p.solute) * p.velocity;
            p.solute += solute;
            p.heat += heat;
            if rho * p.volume + p.solute > 0.0 {
                p.velocity = (old_momentum + momentum) / (rho * p.volume + p.solute);
            }
        }
    }

    fn exchange_heat(&mut self, dt: f64, water: &Bins, grains: &Bins) {
        let wc: Vec<f64> = self
            .water
            .iter()
            .map(|p| self.water_heat_capacity(p))
            .collect();
        let gc: Vec<f64> = self
            .grains
            .iter()
            .map(|g| self.grain_heat_capacity(g))
            .collect();
        let mut dw = vec![0.0; wc.len()];
        let mut dg = vec![0.0; gc.len()];
        for (ws, gs) in water.iter().zip(grains) {
            let cw: f64 = ws.iter().map(|&(i, w)| w * wc[i]).sum();
            let cg: f64 = gs.iter().map(|&(i, w)| w * gc[i]).sum();
            if cw <= 0.0 || cg <= 0.0 {
                continue;
            }
            let tw = ws.iter().map(|&(i, w)| w * self.water[i].heat).sum::<f64>() / cw;
            let tg = gs
                .iter()
                .map(|&(i, w)| w * self.grains[i].heat)
                .sum::<f64>()
                / cg;
            let q = thermal::pair_heat(
                tw as f32,
                cw as f32,
                tg as f32,
                cg as f32,
                self.mats.kappa,
                dt as f32,
            ) as f64;
            for &(i, w) in ws {
                dw[i] += q * w * wc[i] / cw;
            }
            for &(i, w) in gs {
                dg[i] -= q * w * gc[i] / cg;
            }
        }
        for (p, q) in self.water.iter_mut().zip(dw) {
            p.heat += q;
        }
        for (g, q) in self.grains.iter_mut().zip(dg) {
            g.heat += q;
        }
    }
}

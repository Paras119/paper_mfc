#!/usr/bin/env python3
"""
CUDA microbial fuel-cell model with resumable OAT sensitivity analysis.

The default entry point runs one baseline and +/-30% cases for nine model
parameters using seed 43 and a 240-hour (10-day) simulation. On Kaggle, outputs are
written to /kaggle/working/sensitivity_outputs and packaged as
/kaggle/working/sensitivity_outputs.zip.

Useful commands
---------------
Full Kaggle/local run:
    python mfc_oat_sensitivity_kaggle.py

Reduced five-case interface check:
    python mfc_oat_sensitivity_kaggle.py --smoke-test

Resume is automatic. Use --overwrite only when completed cases must be rerun.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tqdm
import math
import warnings
import os
import argparse
import contextlib
import gc
import json
import re
import shutil
import time as walltime
import traceback
from pathlib import Path

try:
    import seaborn as sns
except ImportError:
    class _SeabornFallback:
        @staticmethod
        def set_theme(**kwargs):
            plt.style.use("default")

        @staticmethod
        def despine(**kwargs):
            return None

    sns = _SeabornFallback()

os.environ.setdefault("NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS", "0")
warnings.filterwarnings(
    "ignore",
    message=r".*Grid size .*will likely result in GPU under-utilization.*")
warnings.filterwarnings("ignore", message=r".*Grid size .*low occupancy.*")
from numba import cuda
from numba.cuda.random import (create_xoroshiro128p_states,
                                xoroshiro128p_uniform_float32)
try:
    from numba.core import config as numba_config
    numba_config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
except Exception:
    pass
try:
    from numba.core.errors import NumbaPerformanceWarning
except Exception:
    NumbaPerformanceWarning = None

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", message=r".*Grid size .*low occupancy.*")
warnings.filterwarnings(
    "ignore",
    message=r".*Grid size .*will likely result in GPU under-utilization.*")
warnings.filterwarnings("ignore", module=r"numba_cuda\.numba\.cuda\.dispatcher")
if NumbaPerformanceWarning is not None:
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
else:
    warnings.filterwarnings("ignore", message=r"Grid size .*low occupancy.*")

# ===========================================================================
# MODE CONSTANTS — passed to kernels as int8 to avoid string comparisons
# ===========================================================================

MODE_SINGLE      = np.int8(0)   # Org1 only; Org2 absent
MODE_COMPETITIVE = np.int8(1)   # Org1 + Org2 both consume acetate
MODE_COMMENSAL   = np.int8(2)   # Org2 ferments complex_carbon → acetate → Org1

SCRIPT_VERSION = "mfc-oat-resumable-2026-07-19"


# ===========================================================================
# SECTION 1 — CONFIGURATION
# ===========================================================================

class Config:
    """
    Central parameter store for the unified three-mode MFC simulation.

    Active mode derivation
    ----------------------
    Set `enable_species2` and `species2_role` ("competitive" or "commensal"),
    then optionally override the initial agent counts.  After all keyword
    overrides are applied, `active_mode` is derived deterministically:

        enable_species2=False  OR  (bf2 + pk2 == 0)  →  active_mode = "single"
        enable_species2=True, role="competitive"       →  active_mode = "competitive"
        enable_species2=True, role="commensal"         →  active_mode = "commensal"

    Code everywhere else reads only `active_mode` and `mode_code` (int8).

    Substrate convention (all three modes)
    ----------------------------------------
    Acetate grid (d_substrate):
        [mM]  — electron donor for Org1.
        In "single" and "competitive" modes this is fed directly at
        S_feed_concentration.  In "commensal" mode it is produced by Org2
        and consumed by Org1 (feed concentration set to 0 at batch start).

    Complex-carbon grid (d_complex_carbon):
        [mM]  — used only in "commensal" mode.
        Org2 consumes it at standard Monod kinetics and converts a fraction
        to acetate (yield_complex_to_acetate).  Diffuses with its own
        diffusivity.

    COD accounting
    --------------
    Yields (Y_sp1, Y_sp2) are in g-VSS / g-COD.
    mol_mass_sub  = 64 g-COD / mol acetate  (2 mol O2 × 32 g/mol).
    mol_mass_cmplx= COD equivalent of the complex carbon source [g-COD/mol].
    Using the molecular weight of acetate (59 g/mol) instead of 64 g-COD/mol
    underestimates moles consumed by ~8 %; this bug is corrected here.
    """
    def __init__(self, **kwargs):
        # ── Mode flags ────────────────────────────────────────────────────
        self.enable_species2  = False           # Whether Org2 is present at all
        self.species2_role    = "competitive"   # "competitive" | "commensal"

        # ── Reactor geometry ──────────────────────────────────────────────
        self.L          = 100e-6         # [m]
        self.N_grid     = 96             # [-]
        self.max_agents = 10_000_000      # [-]

        # ── Temporal ─────────────────────────────────────────────────────
        self.dt_bio            = 3600.0           # [s]
        self.feeding_interval  = 10 * 24 * 3600   # [s]

        # ── Diffusion solver controls ────────────────────────────────────
        self.max_pseudo_iter          = 5000
        self.diffusion_tol            = 1e-4
        self.diffusion_check_interval = 50

        # ── Acetate substrate ─────────────────────────────────────────────
        # Used in single/competitive modes as direct feed; in commensal mode
        # initial bulk acetate is 0 (produced by Org2 from complex carbon).
        self.S_feed_concentration = 15.625        # [mM]
        self.S_bulk               = 15.625        # [mM]
        self.D_s                  = 1.0e-9        # [m²/s]
        self.biofilm_diffusivity_ratio_sub = 0.80 # [-]

        # ── Complex carbon (commensal mode only) ──────────────────────────
        # Org2 consumes complex_carbon and excretes acetate.
        # All parameters are inactive when active_mode != "commensal".
        self.C_feed_concentration        = 15.625   # [mM]  complex carbon feed
        self.D_cmplx                     = 0.8e-9   # [m²/s]  (slightly slower)
        self.biofilm_diffusivity_ratio_cmplx = 0.75 # [-]
        # Stoichiometric acetate yield of Org2 fermentation:
        # mol acetate produced per mol complex carbon consumed.
        # E.g. glucose → 2 pyruvate → 2 acetate + 2 CO2 + 4H gives ~1.0;
        # a conservative default of 0.5 accounts for partial oxidation paths.
        self.yield_complex_to_acetate    = 0.5      # [mol acetate / mol complex]
        self.mol_mass_cmplx              = 192.0    # [g-COD/mol] (glucose-equivalent)

        # ── Initial agent counts ──────────────────────────────────────────
        self.initial_bf1_count = 10     # biofilm  Org1 seeds
        self.initial_pk1_count = 5000    # planktonic Org1 seeds
        self.initial_bf2_count = 0     # biofilm  Org2 seeds
        self.initial_pk2_count = 10    # planktonic Org2 seeds

        # ── Species 1 kinetics — electroactive (Geobacter-like) ──────────
        self.mu_max_sp1   = 0.09 / 3600  # [s⁻¹]
        self.Ks_sp1       = 5.0          # [mM]  Monod half-saturation, acetate
        self.yield_sp1    = 0.09         # [g-VSS / g-COD]g
        self.k_decay_sp1  = 0.009 / 86400 # [s⁻¹]  0.03 d⁻¹

        # ── Species 2 kinetics — non-electroactive ────────────────────────
        # In competitive mode: consumes acetate (same substrate grid).
        # In commensal  mode: consumes complex carbon (d_complex_carbon grid).
        self.mu_max_sp2   = 0.50 / 3600  # [s⁻¹]
        self.Ks_sp2       = 5.0            # [mM]  half-saturation (acetate OR complex)
        self.yield_sp2    = 0.40         # [g-VSS / g-COD]
        self.k_decay_sp2  = 0.05 / 86400 # [s⁻¹]  0.05 d⁻¹
        self.min_viable_mass_fraction = 0.20

        # ── Biofilm attachment ────────────────────────────────────────────
        self.prob_attach          = 0.95  # [-]
        self.max_voxels_per_step  = 5.0   # [-]

        # ── Biofilm mechanics ─────────────────────────────────────────────
        self.max_cells_per_voxel  = 5     # [-]
        self.min_crowding_factor  = 0.10  # [-]
        self.shove_iterations     = 4     # [-]
        self.shear_rate           = 0.20  # [s⁻¹]
        self.reference_shear_rate = 0.20  # [s⁻¹]
        self.detachment_rate      = 2e-5  # [s⁻¹]
        self.eps_cohesion         = 0.75  # [-]

        # ── EET parameters (Species 1 only) ─────────────────────────────
        self.L_pili_base        = 20e-6   # [m]
        self.L_pili_growth_rate = 0.5     # [µm/day]

        # ── Electrochemistry ──────────────────────────────────────────────
        self.E_ka             = -0.235    # [V vs SHE]
        self.E0_cathode       = 0.54      # [V vs SHE]
        self.i0_anode         = 1.0       # [A/m²]
        self.R_ext            = 1000.0    # [Ω]
        self.R_contact        = 65.0      # [Ω]
        self.L_gap            = 0.02      # [m]
        self.base_sigma_aq    = 0.6       # [S/m]
        self.t_mem            = 178e-6    # [m]
        self.sigma_mem        = 10.0      # [S/m]
        self.electrode_area_m2 = 0.0048  # [m²]
        self.reactor_vol_anode = 0.3      # [L]
        self.smoothing         = 0.3      # [-]
        self.voltage_tau_rise  = 8 * 3600     # [s]
        self.voltage_tau_decay = 48 * 3600    # [s]

        # ── Proton transport ─────────────────────────────────────────────
        self.D_h                        = 9.3e-9   # [m²/s]
        self.biofilm_diffusivity_ratio_h = 0.85    # [-]
        self.t_H                         = 0.25    # [-]
        self.D_h_PEM                     = 1e-10   # [m²/s]

        # ── pH model ────────────────────────────────────────────────────
        self.init_pH         = 7.0          # [-]
        self.init_H_mM       = 0.0          # [mM]
        self.buffer_capacity = 100.0        # [mM/pH unit]
        self.pH_optimal      = 7.0          # [-]
        self.pH_sigma        = 0.5          # [-]

        # ── Thermodynamic constants ──────────────────────────────────────
        self.F = 96485.3    # [C/mol]
        self.R = 8.314      # [J/mol/K]
        self.T = 298.15     # [K]

        # ── Cell mass and substrate COD factor ───────────────────────────
        self.mass_cell    = 1e-12   # [g]
        # mol_mass_sub: g-COD per mol acetate.
        # Acetate (CH3COO⁻) has 2 mol O2 demand → 64 g-COD/mol.
        # Using acetate MW (59 g/mol) would underestimate moles consumed by ~8%.
        self.mol_mass_sub = 64.0    # [g-COD / mol acetate]

        self.active_area_fraction = 0.01

        # ── Apply keyword overrides ──────────────────────────────────────
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Config: unknown parameter '{k}'.")

        # ── Validate role ─────────────────────────────────────────────────
        if self.species2_role not in ("competitive", "commensal"):
            raise ValueError(
                "species2_role must be 'competitive' or 'commensal'; "
                f"got '{self.species2_role}'.")

        # ── Derive active_mode (single source of truth) ──────────────────
        # Force Org2 counts to zero when the flag is off, so that
        # has_species2 correctly reflects intent even if counts were
        # accidentally non-zero.
        if not self.enable_species2:
            self.initial_bf2_count = 0
            self.initial_pk2_count = 0

        self.has_species2 = (
            self.enable_species2
            and (self.initial_bf2_count + self.initial_pk2_count) > 0
        )

        if not self.has_species2:
            self.active_mode = "single"
        else:
            self.active_mode = self.species2_role  # "competitive" or "commensal"

        # Integer mode code for kernel dispatch
        _code_map = {"single": MODE_SINGLE,
                     "competitive": MODE_COMPETITIVE,
                     "commensal": MODE_COMMENSAL}
        self.mode_code = _code_map[self.active_mode]

        # In commensal mode the acetate bulk starts at zero; it is produced
        # by Org2.  Override S_bulk/S_feed here after mode is known.
        if self.active_mode == "commensal":
            self.S_bulk = 0.0
            self.S_feed_concentration = 0.0

        # ── Derived quantities ────────────────────────────────────────────
        self.dx           = self.L / self.N_grid
        self.dV           = self.dx ** 3
        self.dV_liters    = self.dV * 1000.0
        self.k_m_PEM      = self.D_h_PEM / self.t_mem

        self._alpha_sub       = 0.15
        self._alpha_h         = 0.15
        self._alpha_cmplx     = 0.15   # pseudo-time step weight for complex carbon
        self._pseudo_dt_sub   = self._alpha_sub   * (self.dx**2) / self.D_s
        self._pseudo_dt_h     = self._alpha_h     * (self.dx**2) / self.D_h
        self._pseudo_dt_cmplx = self._alpha_cmplx * (self.dx**2) / self.D_cmplx


# ===========================================================================
# SECTION 2 — CUDA KERNELS
# ===========================================================================

# ---------------------------------------------------------------------------
# 2a — Helper: local diffusivity ratio (device function, shared by all kernels)
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def local_diffusivity_ratio(cell_count, z, y, x,
                             max_cells_per_voxel, dense_ratio):
    """
    Returns a value in [dense_ratio, 1.0]: 1.0 in empty voxels, approaching
    dense_ratio as the voxel fills to max_cells_per_voxel.
    """
    fraction = min(1.0, cell_count[z, y, x] / float(max_cells_per_voxel))
    return 1.0 - (1.0 - dense_ratio) * fraction


# ---------------------------------------------------------------------------
# 2b — Biology kernel (two-species, three-mode)
# ---------------------------------------------------------------------------

@cuda.jit
def biology_kernel(
        agent_pos, agent_mass, agent_type,
        n_active_start, active_count, cell_count,
        # Acetate grid (Org1 substrate; Org2 substrate in competitive mode)
        substrate_grid,
        # Complex-carbon grid (Org2 substrate in commensal mode; ignored otherwise)
        complex_grid,
        # Output rate grids
        uptake_sp1_grid,      # EET rate [mM-COD/L/s] reaching anode (Org1 only)
        uptake_total_grid,    # total metabolic COD sink [mM-COD/L/s] (all species)
        acetate_gen_grid,     # acetate produced by Org2 [mM/L/s] (commensal only)
        uptake_cmplx_grid,    # complex-carbon sink [mM/L/s] (commensal only)
        proton_grid,
        proton_gen_rate_grid,
        dt_bio, mode_code,
        mu_max_sp1, k_decay_sp1, Ks_sp1, Y_sp1,
        mu_max_sp2, k_decay_sp2, Ks_sp2, Y_sp2,
        E_anode_actual, E_ka, F, R, T,
        mass_cell, max_agents, dV_liters, dx,
        init_pH, buffer_capacity, pH_optimal, pH_sigma,
        S_bulk, C_bulk, current_time_s, L_pili_base, pili_growth_rate,
        L_biofilm_um, max_cells_per_voxel, min_crowding_factor,
        mol_mass_sub, mol_mass_cmplx, yield_complex_to_acetate):
    """
    Per-agent biological update.  Dispatches Org2 behaviour based on mode_code.

    Type encoding
    -------------
    0 = planktonic Org1    1 = biofilm Org1
    2 = planktonic Org2    3 = biofilm Org2

    MODE_SINGLE (0)
    ---------------
    Org2 agents are absent by construction.  Any agent with type 2/3 is
    silently skipped (guard against stray seeds).  Org1 behaviour unchanged.

    MODE_COMPETITIVE (1)
    --------------------
    Org2 consumes acetate from `substrate_grid` at its own Monod parameters.
    Org2 contributes to `uptake_total_grid` (acetate sink) but not to EET.

    MODE_COMMENSAL (2)
    ------------------
    Org2 consumes complex carbon from `complex_grid`.
    Org2 excretes acetate at rate:
        acetate_excretion_rate = (specific_uptake_cmplx
                                  / mol_mass_cmplx)     [mol-cmplx/g-VSS/s]
                                 * yield_complex_to_acetate   [mol-ace/mol-cmplx]
                                 * 1000                       [→ mM/L/s per unit mass]
    Org2 contributes to `acetate_gen_grid` (acetate source for Org1) but NOT
    to `uptake_total_grid` (which tracks acetate consumption, not production).

    Grid semantics (final, consistent with step() host logic)
    ---------------------------------------------------------
    uptake_sp1_grid    — EET rate fraction [mM-COD/L/s] (current source)
    uptake_total_grid  — acetate metabolic sink [mM-COD/L/s] (all acetate consumers)
    acetate_gen_grid   — acetate production [mM/L/s] (commensal Org2 only)
    uptake_cmplx_grid  — complex-carbon sink [mM/L/s] (commensal Org2 only)
    proton_gen_rate_grid — anodic proton source (pre-circuit scale; rescaled on host)

    EET / pH / crowding / BV-potential logic
    ----------------------------------------
    Identical to the single-species baseline for Org1.  pH inhibition is
    applied once via mu_max_adj (no double-application).  Crowding is
    applied to both species identically.
    """
    tid = cuda.grid(1)
    if tid >= n_active_start:
        return

    x     = agent_pos[tid, 0]
    y     = agent_pos[tid, 1]
    z     = agent_pos[tid, 2]
    mass  = agent_mass[tid]
    atype = agent_type[tid]

    is_sp1         = (atype == 0 or atype == 1)
    is_sp2         = (atype == 2 or atype == 3)
    is_biofilm_sp1 = (atype == 1)
    is_biofilm_sp2 = (atype == 3)
    is_biofilm     = (is_biofilm_sp1 or is_biofilm_sp2)

    # Skip Org2 agents in single mode (safety guard)
    if is_sp2 and mode_code == 0:   # MODE_SINGLE
        return

    # ── Crowding factor ───────────────────────────────────────────────────
    crowding_factor = 1.0
    if is_biofilm:
        local_count = cell_count[z, y, x]
        if local_count > max_cells_per_voxel:
            crowding_factor = max(
                min_crowding_factor,
                max_cells_per_voxel / float(local_count))

    # ── EET efficiency (Org1 only) ────────────────────────────────────────
    z_dist_um    = z * dx * 1e6
    time_days    = current_time_s / 86400.0
    pili_range_um = (L_pili_base * 1e6) + (pili_growth_rate * time_days)

    if is_sp1 and is_biofilm_sp1:
        span_12 = pili_range_um - 5.0
        if span_12 < 0.1:
            span_12 = 0.1
        span_23 = L_biofilm_um - pili_range_um
        if span_23 < 0.1:
            span_23 = 0.1

        k1 = math.log(0.95 / 0.55) / span_12
        k2 = math.log(0.55 / 0.30) / span_23

        if z_dist_um < 5.0:
            efficiency_factor = 0.95
        elif z_dist_um <= pili_range_um:
            efficiency_factor = 0.95 * math.exp(-k1 * (z_dist_um - 5.0))
        elif z_dist_um <= L_biofilm_um:
            efficiency_factor = 0.55 * math.exp(-k2 * (z_dist_um - pili_range_um))
        else:
            efficiency_factor = 0.30
    elif is_sp2:
        efficiency_factor = 0.0    # Org2 has no EET in any mode
    else:
        efficiency_factor = 0.3    # planktonic Org1 is not anode-connected

    # ── pH inhibition (computed once via mu_max_adj) ──────────────────────
    local_added_H_mM = proton_grid[z, y, x]
    local_pH = max(1.0, init_pH - (local_added_H_mM / buffer_capacity))
    I_pH = math.exp(-((local_pH - pH_optimal) ** 2) / (2.0 * pH_sigma ** 2))
    I_pH = max(0.1, I_pH)

    # ── Species-specific parameters ───────────────────────────────────────
    if is_sp1:
        mu_max_use  = mu_max_sp1
        Ks_use      = Ks_sp1
        Y_use       = Y_sp1
        k_decay_use = k_decay_sp1
    else:
        mu_max_use  = mu_max_sp2
        Ks_use      = Ks_sp2
        Y_use       = Y_sp2
        k_decay_use = k_decay_sp2

    mu_max_adj = mu_max_use * I_pH

    # ── Substrate availability — mode-dependent for Org2 ─────────────────
    # For Org1 and competitive Org2: read from substrate_grid (acetate).
    # For commensal Org2: read from complex_grid (complex carbon).
    if is_sp1:
        S_local = substrate_grid[z, y, x]
        S_ref   = S_bulk          # Monod guard: skip if bulk depleted
    elif mode_code == 2:           # MODE_COMMENSAL — Org2 on complex carbon
        S_local = complex_grid[z, y, x]
        S_ref   = C_bulk          # bulk complex carbon gates fermentation
    else:                          # MODE_COMPETITIVE — Org2 on acetate
        S_local = substrate_grid[z, y, x]
        S_ref   = S_bulk

    # ── Specific uptake rate (total metabolic, substrate-basis) ───────────
    # Identical form for both species; substrate source differs (see above).
    # Rationale for excluding efficiency_factor here: see biology_kernel
    # docstring in the original two-species file.
    specific_uptake_rate = 0.0
    if S_ref > 0.05 and S_local > 0.0:
        f_sub = S_local / (Ks_use + S_local)
        specific_uptake_rate = (mu_max_adj / Y_use) * f_sub * crowding_factor

    # ── EET rate (Org1 only) ─────────────────────────────────────────────
    eet_rate     = 0.0
    eet_rate_mM_s = 0.0
    if is_sp1 and specific_uptake_rate > 0.0:
        bv_arg = -(F / (R * T)) * (E_anode_actual - E_ka)
        if bv_arg < -50.0:
            f_pot = 1.0
        elif bv_arg > 50.0:
            f_pot = 0.0
        else:
            f_pot = 1.0 / (1.0 + math.exp(bv_arg))
        eet_rate = specific_uptake_rate * efficiency_factor * f_pot

    # ── Mass update (full metabolic uptake, both species) ─────────────────
    growth_g = specific_uptake_rate * Y_use * mass * dt_bio
    decay_g  = k_decay_use * mass * dt_bio
    new_mass = max(1e-15, mass + growth_g - decay_g)
    agent_mass[tid] = new_mass

    # ── Grid accumulation ─────────────────────────────────────────────────
    total_uptake_g  = specific_uptake_rate * mass * dt_bio

    if is_sp1:
        # Org1: accumulate EET rate (current source) and total acetate sink.
        eet_uptake_g  = eet_rate * mass * dt_bio
        eet_rate_mM_s = ((eet_uptake_g / mol_mass_sub) * 1000.0 / dV_liters) / dt_bio
        total_rate_mM_s = ((total_uptake_g / mol_mass_sub) * 1000.0 / dV_liters) / dt_bio
        cuda.atomic.add(uptake_sp1_grid,   (z, y, x), eet_rate_mM_s)
        cuda.atomic.add(uptake_total_grid, (z, y, x), total_rate_mM_s)

    else:
        # Org2 paths depend on mode.
        if mode_code == 2:   # MODE_COMMENSAL
            # Org2 consumes complex carbon → excrete acetate.
            # complex carbon consumption rate [mM-COD/L/s]:
            cmplx_rate_mM_s = ((total_uptake_g / mol_mass_cmplx) * 1000.0
                                / dV_liters) / dt_bio
            # Acetate excretion rate [mM/L/s]:
            # mol-complex consumed × yield → mol-acetate produced
            ace_gen_mM_s = cmplx_rate_mM_s * yield_complex_to_acetate
            cuda.atomic.add(acetate_gen_grid, (z, y, x), ace_gen_mM_s)
            cuda.atomic.add(uptake_cmplx_grid, (z, y, x), cmplx_rate_mM_s)
            # Note: uptake_total_grid tracks acetate sinks only; commensal
            # Org2 does not consume acetate, so no addition here.

        else:                # MODE_COMPETITIVE
            # Org2 consumes acetate — add to the shared acetate sink.
            total_rate_mM_s = ((total_uptake_g / mol_mass_sub) * 1000.0
                                / dV_liters) / dt_bio
            cuda.atomic.add(uptake_total_grid, (z, y, x), total_rate_mM_s)

    # ── Proton generation (anodic Org1 biofilm only) ──────────────────────
    # Accumulated as uncapped EET proxy; rescaled on host after circuit solve.
    # Coefficient 7.0: 8 H⁺/mol acetate minus 1 consumed in bicarbonate buffer.
    if is_biofilm_sp1 and eet_rate_mM_s > 0.0:
        cuda.atomic.add(proton_gen_rate_grid, (z, y, x),
                        eet_rate_mM_s * 7.0)

    # ── Cell division ─────────────────────────────────────────────────────
    if new_mass > 2.0 * mass_cell:
        new_idx = cuda.atomic.add(active_count, 0, 1)
        if new_idx < max_agents:
            agent_mass[tid]       = new_mass / 2.0
            agent_pos[new_idx, 0] = x
            agent_pos[new_idx, 1] = y
            agent_pos[new_idx, 2] = z
            agent_mass[new_idx]   = new_mass / 2.0
            agent_type[new_idx]   = atype
        else:
            cuda.atomic.add(active_count, 0, -1)


# ---------------------------------------------------------------------------
# 2c — Substrate diffusion (acetate; shared by all modes)
# ---------------------------------------------------------------------------

@cuda.jit
def diffusion_kernel_consume(grid, new_grid, rate_grid, cell_count,
                              alpha, pseudo_dt, bulk,
                              max_cells_per_voxel, dense_ratio):
    """
    Explicit pseudo-time FD diffusion with net consumption sink.

    In competitive mode `rate_grid` holds combined Org1+Org2 acetate
    consumption.  In commensal mode it holds Org1 consumption minus Org2
    acetate production (handled on host before calling by scaling; see
    step() for details).

    Boundary: Dirichlet (C = bulk) at z = N-1 (bulk face), Neumann elsewhere.
    """
    x, y, z = cuda.grid(3)
    nx, ny, nz = grid.shape
    if x >= nx or y >= ny or z >= nz:
        return

    if z == nz - 1:
        new_grid[z, y, x] = bulk
        return

    c   = grid[z, y, x]
    d_c = local_diffusivity_ratio(cell_count, z, y, x,
                                   max_cells_per_voxel, dense_ratio)
    flux = 0.0

    d_n = local_diffusivity_ratio(cell_count, z+1, y, x,
                                   max_cells_per_voxel, dense_ratio)
    flux += 0.5*(d_c+d_n)*(grid[z+1, y, x] - c)
    if z > 0:
        d_n = local_diffusivity_ratio(cell_count, z-1, y, x,
                                       max_cells_per_voxel, dense_ratio)
        flux += 0.5*(d_c+d_n)*(grid[z-1, y, x] - c)
    if y < ny-1:
        d_n = local_diffusivity_ratio(cell_count, z, y+1, x,
                                       max_cells_per_voxel, dense_ratio)
        flux += 0.5*(d_c+d_n)*(grid[z, y+1, x] - c)
    if y > 0:
        d_n = local_diffusivity_ratio(cell_count, z, y-1, x,
                                       max_cells_per_voxel, dense_ratio)
        flux += 0.5*(d_c+d_n)*(grid[z, y-1, x] - c)
    if x < nx-1:
        d_n = local_diffusivity_ratio(cell_count, z, y, x+1,
                                       max_cells_per_voxel, dense_ratio)
        flux += 0.5*(d_c+d_n)*(grid[z, y, x+1] - c)
    if x > 0:
        d_n = local_diffusivity_ratio(cell_count, z, y, x-1,
                                       max_cells_per_voxel, dense_ratio)
        flux += 0.5*(d_c+d_n)*(grid[z, y, x-1] - c)

    new_grid[z, y, x] = max(0.0,
        c + alpha*flux - rate_grid[z, y, x]*pseudo_dt)


# ---------------------------------------------------------------------------
# 2d — Complex-carbon diffusion (commensal mode only)
#      Production sink = Org2 consumption; no EET; Dirichlet at bulk face.
# ---------------------------------------------------------------------------

@cuda.jit
def diffusion_kernel_complex(grid, new_grid, rate_grid, cell_count,
                              alpha, pseudo_dt, bulk,
                              max_cells_per_voxel, dense_ratio):
    """
    Pseudo-time FD diffusion for complex carbon (commensal mode).

    `rate_grid` is the complex-carbon consumption rate [mM-COD/L/s] from
    Org2 (accumulated in uptake_cmplx_grid on the host side).
    Boundary: Dirichlet at z=N-1 (C_feed_concentration), Neumann elsewhere.
    Structurally identical to diffusion_kernel_consume; kept separate so
    that substrate (acetate) and complex carbon can have different pseudo_dt
    values (D_cmplx ≠ D_s).
    """
    x, y, z = cuda.grid(3)
    nx, ny, nz = grid.shape
    if x >= nx or y >= ny or z >= nz:
        return

    if z == nz - 1:
        new_grid[z, y, x] = bulk
        return

    c   = grid[z, y, x]
    d_c = local_diffusivity_ratio(cell_count, z, y, x,
                                   max_cells_per_voxel, dense_ratio)
    flux = 0.0

    d_n = local_diffusivity_ratio(cell_count, z+1, y, x,
                                   max_cells_per_voxel, dense_ratio)
    flux += 0.5*(d_c+d_n)*(grid[z+1, y, x] - c)
    if z > 0:
        d_n = local_diffusivity_ratio(cell_count, z-1, y, x,
                                       max_cells_per_voxel, dense_ratio)
        flux += 0.5*(d_c+d_n)*(grid[z-1, y, x] - c)
    if y < ny-1:
        d_n = local_diffusivity_ratio(cell_count, z, y+1, x,
                                       max_cells_per_voxel, dense_ratio)
        flux += 0.5*(d_c+d_n)*(grid[z, y+1, x] - c)
    if y > 0:
        d_n = local_diffusivity_ratio(cell_count, z, y-1, x,
                                       max_cells_per_voxel, dense_ratio)
        flux += 0.5*(d_c+d_n)*(grid[z, y-1, x] - c)
    if x < nx-1:
        d_n = local_diffusivity_ratio(cell_count, z, y, x+1,
                                       max_cells_per_voxel, dense_ratio)
        flux += 0.5*(d_c+d_n)*(grid[z, y, x+1] - c)
    if x > 0:
        d_n = local_diffusivity_ratio(cell_count, z, y, x-1,
                                       max_cells_per_voxel, dense_ratio)
        flux += 0.5*(d_c+d_n)*(grid[z, y, x-1] - c)

    new_grid[z, y, x] = max(0.0,
        c + alpha*flux - rate_grid[z, y, x]*pseudo_dt)


# ---------------------------------------------------------------------------
# 2e — Acetate-with-production diffusion (commensal mode)
#      rate_grid = net_rate = uptake_total (sink) - acetate_gen (source)
#      Dirichlet at z=N-1 set to S_bulk (≈0 at start; accumulates over time)
# ---------------------------------------------------------------------------

@cuda.jit
def diffusion_kernel_acetate_commensal(grid, new_grid, net_rate_grid,
                                        cell_count, alpha, pseudo_dt, bulk,
                                        max_cells_per_voxel, dense_ratio):
    """
    Acetate diffusion for commensal mode.

    net_rate_grid = uptake_total_grid - acetate_gen_grid  [mM/L/s]
    (positive values = net consumption; negative = net production region).
    Identical stencil to diffusion_kernel_consume; separation is for clarity
    and to allow future boundary-condition differences.
    """
    x, y, z = cuda.grid(3)
    nx, ny, nz = grid.shape
    if x >= nx or y >= ny or z >= nz:
        return

    if z == nz - 1:
        new_grid[z, y, x] = bulk
        return

    c   = grid[z, y, x]
    d_c = local_diffusivity_ratio(cell_count, z, y, x,
                                   max_cells_per_voxel, dense_ratio)
    flux = 0.0

    d_n = local_diffusivity_ratio(cell_count, z+1, y, x,
                                   max_cells_per_voxel, dense_ratio)
    flux += 0.5*(d_c+d_n)*(grid[z+1, y, x] - c)
    if z > 0:
        d_n = local_diffusivity_ratio(cell_count, z-1, y, x,
                                       max_cells_per_voxel, dense_ratio)
        flux += 0.5*(d_c+d_n)*(grid[z-1, y, x] - c)
    if y < ny-1:
        d_n = local_diffusivity_ratio(cell_count, z, y+1, x,
                                       max_cells_per_voxel, dense_ratio)
        flux += 0.5*(d_c+d_n)*(grid[z, y+1, x] - c)
    if y > 0:
        d_n = local_diffusivity_ratio(cell_count, z, y-1, x,
                                       max_cells_per_voxel, dense_ratio)
        flux += 0.5*(d_c+d_n)*(grid[z, y-1, x] - c)
    if x < nx-1:
        d_n = local_diffusivity_ratio(cell_count, z, y, x+1,
                                       max_cells_per_voxel, dense_ratio)
        flux += 0.5*(d_c+d_n)*(grid[z, y, x+1] - c)
    if x > 0:
        d_n = local_diffusivity_ratio(cell_count, z, y, x-1,
                                       max_cells_per_voxel, dense_ratio)
        flux += 0.5*(d_c+d_n)*(grid[z, y, x-1] - c)

    new_grid[z, y, x] = max(0.0,
        c + alpha*flux - net_rate_grid[z, y, x]*pseudo_dt)


# ---------------------------------------------------------------------------
# 2f — Proton diffusion with PEM loss
# ---------------------------------------------------------------------------

@cuda.jit
def diffusion_kernel_proton_pem(grid, new_grid, rate_grid, cell_count,
                                 alpha, pseudo_dt,
                                 k_m_PEM, dx, mig_rate,
                                 max_cells_per_voxel, dense_ratio):
    """
    Proton diffusion with PEM removal at the top face (z = N-1).

    mig_rate = (j_actual [A/m²] * t_H / F) / dx   [mM/s per voxel thickness]
    This is the proton migration flux density driven by the actual current
    density; the original bug (passing total current instead of current
    density) over-applied the PEM sink by ~480×.
    """
    x, y, z = cuda.grid(3)
    nx, ny, nz = grid.shape
    if x >= nx or y >= ny or z >= nz:
        return

    c   = grid[z, y, x]
    d_c = local_diffusivity_ratio(cell_count, z, y, x,
                                   max_cells_per_voxel, dense_ratio)
    flux = 0.0

    if z > 0:
        d_n = local_diffusivity_ratio(cell_count, z-1, y, x,
                                       max_cells_per_voxel, dense_ratio)
        flux += 0.5*(d_c+d_n)*(grid[z-1, y, x] - c)
    if y < ny-1:
        d_n = local_diffusivity_ratio(cell_count, z, y+1, x,
                                       max_cells_per_voxel, dense_ratio)
        flux += 0.5*(d_c+d_n)*(grid[z, y+1, x] - c)
    if y > 0:
        d_n = local_diffusivity_ratio(cell_count, z, y-1, x,
                                       max_cells_per_voxel, dense_ratio)
        flux += 0.5*(d_c+d_n)*(grid[z, y-1, x] - c)
    if x < nx-1:
        d_n = local_diffusivity_ratio(cell_count, z, y, x+1,
                                       max_cells_per_voxel, dense_ratio)
        flux += 0.5*(d_c+d_n)*(grid[z, y, x+1] - c)
    if x > 0:
        d_n = local_diffusivity_ratio(cell_count, z, y, x-1,
                                       max_cells_per_voxel, dense_ratio)
        flux += 0.5*(d_c+d_n)*(grid[z, y, x-1] - c)

    if z == nz-1:
        flux_diff = k_m_PEM * c * pseudo_dt / dx
        flux_mig  = mig_rate * pseudo_dt
        new_grid[z, y, x] = max(0.0,
            c + alpha*flux + rate_grid[z, y, x]*pseudo_dt
            - flux_diff - flux_mig)
    else:
        d_n = local_diffusivity_ratio(cell_count, z+1, y, x,
                                       max_cells_per_voxel, dense_ratio)
        flux += 0.5*(d_c+d_n)*(grid[z+1, y, x] - c)
        new_grid[z, y, x] = max(0.0,
            c + alpha*flux + rate_grid[z, y, x]*pseudo_dt)


# ---------------------------------------------------------------------------
# 2g — Movement, attachment, mechanics kernels (unchanged from original)
# ---------------------------------------------------------------------------

@cuda.jit
def planktonic_move_kernel(pos, agent_type, n_active,
                            nx, ny, nz, rng, max_v):
    tid = cuda.grid(1)
    if tid >= n_active[0]:
        return
    atype = agent_type[tid]
    if atype == 1 or atype == 3:
        return
    jr  = xoroshiro128p_uniform_float32(rng, tid) * max_v
    phi = xoroshiro128p_uniform_float32(rng, tid) * 2.0 * math.pi
    th  = xoroshiro128p_uniform_float32(rng, tid) * math.pi
    pos[tid, 0] = max(0, min(nx-1,
        pos[tid, 0] + int(jr*math.sin(th)*math.cos(phi))))
    pos[tid, 1] = max(0, min(ny-1,
        pos[tid, 1] + int(jr*math.sin(th)*math.sin(phi))))
    pos[tid, 2] = max(0, min(nz-1,
        pos[tid, 2] + int(jr*math.cos(th))))


@cuda.jit
def fill_occupancy(pos, agent_type, occ, n_active):
    tid = cuda.grid(1)
    if tid < n_active[0]:
        atype = agent_type[tid]
        if atype == 1 or atype == 3:
            cuda.atomic.add(occ,
                (pos[tid, 2], pos[tid, 1], pos[tid, 0]), 1)


@cuda.jit
def attachment_kernel(pos, agent_type, occ, n_active, prob, rng):
    tid = cuda.grid(1)
    if tid >= n_active[0]:
        return
    atype = agent_type[tid]
    if atype == 1 or atype == 3:
        return
    x, y, z = pos[tid, 0], pos[tid, 1], pos[tid, 2]
    if z == 0 or (z > 0 and occ[z-1, y, x] > 0):
        if xoroshiro128p_uniform_float32(rng, tid) < prob:
            agent_type[tid] = atype + 1


@cuda.jit
def shove_kernel(pos, agent_type, n_active, cell_count,
                 max_cells_per_voxel, nx, ny, nz, rng):
    tid = cuda.grid(1)
    if tid >= n_active[0]:
        return
    atype = agent_type[tid]
    if atype != 1 and atype != 3:
        return

    x, y, z = pos[tid, 0], pos[tid, 1], pos[tid, 2]
    if cell_count[z, y, x] <= max_cells_per_voxel:
        return

    direction = int(xoroshiro128p_uniform_float32(rng, tid) * 5)
    tx, ty, tz = x, y, z
    if direction == 0 and x < nx-1:
        tx += 1
    elif direction == 1 and x > 0:
        tx -= 1
    elif direction == 2 and y < ny-1:
        ty += 1
    elif direction == 3 and y > 0:
        ty -= 1
    elif direction == 4 and z < nz-1:
        tz += 1
    else:
        return

    source_prev = cuda.atomic.add(cell_count, (z, y, x), -1)
    if source_prev <= max_cells_per_voxel:
        cuda.atomic.add(cell_count, (z, y, x), 1)
        return

    dest_prev = cuda.atomic.add(cell_count, (tz, ty, tx), 1)
    if dest_prev >= max_cells_per_voxel:
        cuda.atomic.add(cell_count, (tz, ty, tx), -1)
        cuda.atomic.add(cell_count, (z, y, x), 1)
        return

    pos[tid, 0] = tx
    pos[tid, 1] = ty
    pos[tid, 2] = tz


@cuda.jit
def shear_detachment_kernel(pos, agent_type, n_active, cell_count,
                             dt_bio, shear_rate, reference_shear_rate,
                             detachment_rate, eps_cohesion,
                             nx, ny, nz, rng, detached_count):
    tid = cuda.grid(1)
    if tid >= n_active[0]:
        return
    atype = agent_type[tid]
    if atype != 1 and atype != 3:
        return

    x, y, z = pos[tid, 0], pos[tid, 1], pos[tid, 2]
    if z == 0:
        return

    empty_faces = 0
    if x == 0 or cell_count[z, y, x-1] == 0:
        empty_faces += 1
    if x == nx-1 or cell_count[z, y, x+1] == 0:
        empty_faces += 1
    if y == 0 or cell_count[z, y-1, x] == 0:
        empty_faces += 1
    if y == ny-1 or cell_count[z, y+1, x] == 0:
        empty_faces += 1
    if cell_count[z-1, y, x] == 0:
        empty_faces += 1
    if z == nz-1 or cell_count[z+1, y, x] == 0:
        empty_faces += 1

    exposure = empty_faces / 6.0
    if exposure <= 0.0:
        return

    shear_ratio    = shear_rate / max(reference_shear_rate, 1e-12)
    height_factor  = (z + 1.0) / nz
    cohesion_factor = max(0.02, 1.0 - eps_cohesion)
    hazard = (detachment_rate * shear_ratio * exposure
              * (0.25 + 0.75*height_factor) * cohesion_factor)
    probability = 1.0 - math.exp(-hazard * dt_bio)

    if xoroshiro128p_uniform_float32(rng, tid) < probability:
        agent_type[tid] = atype - 1
        pos[tid, 0] = min(nx-1, x+1)
        pos[tid, 2] = min(nz-1, z+1)
        cuda.atomic.add(detached_count, 0, 1)


# ===========================================================================
# SECTION 3 — SIMULATION CLASS
# ===========================================================================

class UnifiedMFCSimulation:
    """
    Unified three-mode MFC simulation.

    Mode is determined by Config.active_mode (derived from flags + counts).
    All three modes share the same Org1 biology and electrochemistry; only
    the Org2 substrate routing and diffusion kernels differ.

    GPU array allocation
    --------------------
    d_complex_carbon / d_new_complex_carbon / d_uptake_cmplx:
        Allocated in all modes but only written/read in commensal mode.
        Allocating unconditionally avoids branching in __init__ and keeps
        memory footprint negligible relative to the agent arrays.

    d_acetate_gen_grid:
        Acetate production rate [mM/L/s] from commensal Org2.
        Zero in single and competitive modes.
    """

    def __init__(self, random_seed=42, **kwargs):
        self.cfg            = Config(**kwargs)
        self.time           = 0.0
        self.last_feed      = 0.0
        self.voltage_smooth = 0.0
        self.S_bulk         = self.cfg.S_bulk   # may be 0 for commensal
        self.C_bulk         = (self.cfg.C_feed_concentration
                               if self.cfg.active_mode == "commensal" else 0.0)
        self.host_rng       = np.random.default_rng(random_seed)
        self.E_anode_actual = -0.150

        self.shape = (self.cfg.N_grid,) * 3

        # ── GPU agent arrays ──────────────────────────────────────────────
        N = self.cfg.max_agents
        self.d_agent_pos       = cuda.device_array((N, 3), dtype=np.int32)
        self.d_agent_mass      = cuda.device_array(N, dtype=np.float32)
        self.d_agent_type      = cuda.device_array(N, dtype=np.int8)
        self.d_occupancy       = cuda.device_array(self.shape, dtype=np.int32)
        self.d_active_count    = cuda.to_device(np.array([0], dtype=np.int32))
        self.d_detached_count  = cuda.to_device(np.array([0], dtype=np.int32))
        self.rng_states        = create_xoroshiro128p_states(N, seed=random_seed)
        self._zero_occupancy   = np.zeros(self.shape, dtype=np.int32)

        # ── Substrate (acetate) grids ─────────────────────────────────────
        self.d_substrate     = cuda.to_device(
            np.full(self.shape, self.S_bulk, dtype=np.float32))
        self.d_new_substrate = cuda.device_array(self.shape, dtype=np.float32)

        # ── Complex-carbon grids (commensal only; zero otherwise) ─────────
        self.d_complex_carbon     = cuda.to_device(
            np.full(self.shape, self.C_bulk, dtype=np.float32))
        self.d_new_complex_carbon = cuda.device_array(self.shape, dtype=np.float32)

        # ── Rate grids ────────────────────────────────────────────────────
        self.d_uptake_sp1     = cuda.device_array(self.shape, dtype=np.float32)
        # uptake_total_grid: acetate consumption sink (all acetate consumers).
        # In commensal mode this is Org1 only; in competitive mode Org1+Org2.
        self.d_uptake_total   = cuda.device_array(self.shape, dtype=np.float32)
        # Complex-carbon consumption rate [mM-COD/L/s]; commensal Org2 only.
        self.d_uptake_cmplx   = cuda.device_array(self.shape, dtype=np.float32)
        # Acetate production [mM/L/s]; commensal Org2 only.
        self.d_acetate_gen    = cuda.device_array(self.shape, dtype=np.float32)

        # ── Proton grids ──────────────────────────────────────────────────
        self.d_proton          = cuda.to_device(
            np.full(self.shape, self.cfg.init_H_mM, dtype=np.float32))
        self.d_new_proton      = cuda.device_array(self.shape, dtype=np.float32)
        self.d_proton_gen_rate = cuda.device_array(self.shape, dtype=np.float32)

        # ── Pinned convergence buffers ────────────────────────────────────
        # Four buffers: prev/cur for (substrate, proton).
        # In commensal mode two extra for complex carbon are allocated below.
        self._h_sub_prev  = cuda.pinned_array(self.shape, dtype=np.float32)
        self._h_sub_cur   = cuda.pinned_array(self.shape, dtype=np.float32)
        self._h_prot_prev = cuda.pinned_array(self.shape, dtype=np.float32)
        self._h_prot_cur  = cuda.pinned_array(self.shape, dtype=np.float32)
        if self.cfg.active_mode == "commensal":
            self._h_cmplx_prev = cuda.pinned_array(self.shape, dtype=np.float32)
            self._h_cmplx_cur  = cuda.pinned_array(self.shape, dtype=np.float32)
        else:
            self._h_cmplx_prev = None
            self._h_cmplx_cur  = None

        # ── Net-rate scratch buffer for commensal acetate diffusion ───────
        # net_rate = uptake_total (sink) - acetate_gen (source); computed on
        # host once per timestep and uploaded before the diffusion loop.
        # Shape matches self.shape; dtype float32.
        self._h_net_acetate_rate = np.zeros(self.shape, dtype=np.float32)
        self.d_net_acetate_rate  = cuda.device_array(self.shape, dtype=np.float32)

        # ── CUDA launch config ────────────────────────────────────────────
        self.threads      = 256
        self.grid_threads = (8, 8, 8)
        ng                = math.ceil(self.cfg.N_grid / 8)
        self.grid_blocks  = (ng, ng, ng)

        # ── History arrays ────────────────────────────────────────────────
        self.history_n_sp1       = []
        self.history_n_sp2       = []
        self.history_n_bf1       = []
        self.history_n_bf2       = []
        self.history_substrate   = []    # bulk acetate [mM]
        self.history_complex     = []    # bulk complex carbon [mM]; 0 if not commensal
        self.history_voltage     = []
        self.history_j_sp1       = []
        self.history_min_pH      = []
        self.history_E_anode     = []
        self.history_f_EET       = []

        self.spatial_snapshot = None
        self.last_diffusion_iters = 0
        self.last_diffusion_hit_max = False
        self.last_n_sp1 = 0
        self.last_n_sp2 = 0
        self.last_n_bf1 = 0
        self.last_n_bf2 = 0

        self._seed_agents()

    def _seed_agents(self):
        cfg = self.cfg
        ng  = cfg.N_grid
        counts = {
            'bf1': cfg.initial_bf1_count,
            'pk1': cfg.initial_pk1_count,
            'bf2': cfg.initial_bf2_count,
            'pk2': cfg.initial_pk2_count,
        }
        n_total = sum(counts.values())
        if n_total == 0:
            return

        hp = np.zeros((n_total, 3), dtype=np.int32)
        hm = np.zeros(n_total, dtype=np.float32)
        ht = np.zeros(n_total, dtype=np.int8)

        idx = 0
        # Biofilm Org1 — seeded at z=0 (electrode face)
        for _ in range(counts['bf1']):
            hp[idx, 0] = self.host_rng.integers(0, ng)
            hp[idx, 1] = self.host_rng.integers(0, ng)
            hp[idx, 2] = 0
            ht[idx]    = 1
            hm[idx]    = self.host_rng.uniform(0.1, 1.9) * cfg.mass_cell
            idx += 1
        # Planktonic Org1
        for _ in range(counts['pk1']):
            hp[idx, 0] = self.host_rng.integers(0, ng)
            hp[idx, 1] = self.host_rng.integers(0, ng)
            hp[idx, 2] = self.host_rng.integers(1, ng)
            ht[idx]    = 0
            hm[idx]    = self.host_rng.uniform(0.1, 1.9) * cfg.mass_cell
            idx += 1
        # Biofilm Org2
        for _ in range(counts['bf2']):
            hp[idx, 0] = self.host_rng.integers(0, ng)
            hp[idx, 1] = self.host_rng.integers(0, ng)
            hp[idx, 2] = 0
            ht[idx]    = 3
            hm[idx]    = self.host_rng.uniform(0.1, 1.9) * cfg.mass_cell
            idx += 1
        # Planktonic Org2
        for _ in range(counts['pk2']):
            hp[idx, 0] = self.host_rng.integers(0, ng)
            hp[idx, 1] = self.host_rng.integers(0, ng)
            hp[idx, 2] = self.host_rng.integers(1, ng)
            ht[idx]    = 2
            hm[idx]    = self.host_rng.uniform(0.1, 1.9) * cfg.mass_cell
            idx += 1

        self.d_agent_pos[:n_total].copy_to_device(hp)
        self.d_agent_mass[:n_total].copy_to_device(hm)
        self.d_agent_type[:n_total].copy_to_device(ht)
        self.d_active_count.copy_to_device(
            np.array([n_total], dtype=np.int32))

    def _agent_blocks(self, n):
        return max(1, math.ceil(n / self.threads))

    def _rebuild_cell_counts(self, n_active):
        self.d_occupancy.copy_to_device(self._zero_occupancy)
        if n_active > 0:
            fill_occupancy[self._agent_blocks(n_active), self.threads](
                self.d_agent_pos, self.d_agent_type,
                self.d_occupancy, self.d_active_count)

    def to_timeseries_dataframe(self):
        """Return aligned scalar histories as a tidy DataFrame."""
        n_rows = len(self.history_voltage)
        time_days = (
            np.arange(n_rows, dtype=np.float64)
            * self.cfg.dt_bio / 86400.0
        )
        voltage = np.asarray(self.history_voltage, dtype=np.float64)
        current_density = np.asarray(self.history_j_sp1, dtype=np.float64)
        current = current_density * self.cfg.electrode_area_m2

        return pd.DataFrame({
            "time_days": time_days,
            "voltage_V": voltage,
            "current_A": current,
            "power_W": voltage * current,
            "current_density_A_m2": current_density,
            "n_sp1": self.history_n_sp1,
            "n_sp2": self.history_n_sp2,
            "n_bf1": self.history_n_bf1,
            "n_bf2": self.history_n_bf2,
            "bulk_substrate_mM": self.history_substrate,
            "bulk_complex_mM": self.history_complex,
            "domain_min_pH": self.history_min_pH,
            "E_anode_V": self.history_E_anode,
            "f_EET": self.history_f_EET,
        })

    def _solve_circuit(self, I_bio_A, biofilm_age_days):
        """Circuit solver — Org1 EET current only (unchanged from original)."""
        cfg = self.cfg

        _A_join  = 0.1 + 0.7*(1.0 - math.exp(-4.0/2.0))
        _A_tanh  = 1.0 - _A_join
        _tau_mat = _A_tanh / (0.7*0.5*math.exp(-4.0/2.0))

        if biofilm_age_days < 1:
            mat = 0.1 * biofilm_age_days
        elif biofilm_age_days < 5:
            mat = 0.1 + 0.7*(1.0 - math.exp(-(biofilm_age_days-1)/2.0))
        else:
            mat = _A_join + _A_tanh*math.tanh((biofilm_age_days-5)/_tau_mat)

        E_anode_eq = -0.150 + (cfg.E_ka - (-0.150))*mat

        sigma_aq = cfg.base_sigma_aq + (self.S_bulk/100.0)*0.1
        R_elec   = cfg.L_gap / (sigma_aq*cfg.electrode_area_m2)
        R_mem    = cfg.t_mem / (cfg.sigma_mem*cfg.electrode_area_m2)
        R_int    = cfg.R_contact + R_elec + R_mem
        OCV      = cfg.E0_cathode - E_anode_eq

        I_guess = OCV / (R_int + cfg.R_ext)
        eta_bv  = 0.0
        for _ in range(10):
            j_guess  = I_guess / cfg.electrode_area_m2
            eta_bv   = (2.0*cfg.R*cfg.T/cfg.F)*math.asinh(j_guess/(2.0*cfg.i0_anode))
            if I_bio_A > 1e-12:
                ratio    = min(0.9999, I_guess/I_bio_A)
                eta_conc = -(cfg.R*cfg.T/(2.0*cfg.F))*math.log(1.0-ratio)
            else:
                eta_conc = 0.0
            V_cell  = OCV - eta_bv - eta_conc - I_guess*R_int
            I_new   = max(1e-12, V_cell/cfg.R_ext)
            I_guess = 0.3*I_new + 0.7*I_guess

        I_actual   = min(I_bio_A, I_guess)
        j_actual   = I_actual / cfg.electrode_area_m2
        eta_bv     = (2.0*cfg.R*cfg.T/cfg.F)*math.asinh(j_actual/(2.0*cfg.i0_anode))
        V_cell     = I_actual * cfg.R_ext
        E_anode_new = E_anode_eq + eta_bv

        return I_actual, eta_bv, E_anode_new, V_cell

    def calculate_polarization(
            self, resistances_ohm=None, electrical_iterations=40,
            relaxation=0.1, voltage_noise_sd=0.005, random_seed=43):
        """Calculate an instantaneous polarization sweep at the current state.

        Agent states, biomass, concentration fields, and biologically available
        current remain fixed during the sweep. Each resistance point is solved
        independently and is therefore an electrical post-processing result,
        not a biologically or transport-equilibrated steady state.
        """
        cfg = self.cfg
        if resistances_ohm is None:
            resistances_ohm = np.logspace(
                np.log10(50_000.0), np.log10(10.0), 60)
        resistances_ohm = np.asarray(resistances_ohm, dtype=np.float64)
        if (resistances_ohm <= 0).any():
            raise ValueError("All external resistances must be positive.")
        if electrical_iterations <= 0:
            raise ValueError("electrical_iterations must be positive.")
        if not 0.0 < relaxation <= 1.0:
            raise ValueError("relaxation must be in (0, 1].")

        rate_eet_mM_s = float(np.sum(
            self.d_uptake_sp1.copy_to_host(), dtype=np.float64))
        if rate_eet_mM_s <= 0.0:
            raise RuntimeError(
                "Polarization requires a state with non-zero Org1 EET uptake.")

        moles_eet_s = (rate_eet_mM_s * cfg.dV_liters) / 1000.0
        area_scale = (
            cfg.active_area_fraction * cfg.electrode_area_m2 / (cfg.L**2))
        I_bio_A = moles_eet_s * area_scale * 8.0 * cfg.F

        sigma_aq = cfg.base_sigma_aq + (self.S_bulk / 100.0) * 0.1
        R_elec = cfg.L_gap / (sigma_aq * cfg.electrode_area_m2)
        R_mem = cfg.t_mem / (cfg.sigma_mem * cfg.electrode_area_m2)
        R_int = cfg.R_contact + R_elec + R_mem

        age_days = self.time / 86400.0
        A_join = 0.1 + 0.7 * (1.0 - math.exp(-2.0))
        A_tanh = 1.0 - A_join
        tau_mat = A_tanh / (0.35 * math.exp(-2.0))
        if age_days < 1.0:
            maturation = 0.1 * age_days
        elif age_days < 5.0:
            maturation = 0.1 + 0.7 * (
                1.0 - math.exp(-(age_days - 1.0) / 2.0))
        else:
            maturation = A_join + A_tanh * math.tanh(
                (age_days - 5.0) / tau_mat)
        E_anode_eq = -0.150 + (cfg.E_ka + 0.150) * maturation
        OCV = cfg.E0_cathode - E_anode_eq

        rng = np.random.default_rng(random_seed)
        rows = []
        for R_ext in resistances_ohm:
            I_guess = OCV / (R_int + R_ext)
            for _ in range(int(electrical_iterations)):
                j_guess = I_guess / cfg.electrode_area_m2
                eta_act = (2.0 * cfg.R * cfg.T / cfg.F) * math.asinh(
                    j_guess / (2.0 * cfg.i0_anode))
                ratio = min(0.9999, I_guess / I_bio_A)
                eta_conc = -(cfg.R * cfg.T / (2.0 * cfg.F)) * math.log(
                    1.0 - ratio)
                V_cell = OCV - eta_act - eta_conc - I_guess * R_int
                I_new = max(1e-12, V_cell / R_ext)
                I_guess = (
                    relaxation * I_new + (1.0 - relaxation) * I_guess)

            voltage_V = max(
                0.0, I_guess * R_ext + rng.normal(0.0, voltage_noise_sd))
            current_density_A_m2 = I_guess / cfg.electrode_area_m2
            rows.append({
                "external_resistance_ohm": float(R_ext),
                "current_A": float(I_guess),
                "current_density_A_m2": float(current_density_A_m2),
                "current_density_mA_m2": float(current_density_A_m2 * 1000.0),
                "voltage_V": float(voltage_V),
                "power_density_mW_m2": float(
                    voltage_V * current_density_A_m2 * 1000.0),
                "I_bio_A": float(I_bio_A),
                "OCV_V": float(OCV),
                "R_internal_ohm": float(R_int),
            })
        return pd.DataFrame(rows)

    def step(self):
        cfg = self.cfg

        # ── Batch feed ────────────────────────────────────────────────────
        if (self.time > 0
                and self.time - self.last_feed >= cfg.feeding_interval):
            self.last_feed      = self.time

            if cfg.active_mode == "commensal":
                # Refeed complex carbon; acetate starts back at zero.
                self.C_bulk = cfg.C_feed_concentration
                self.S_bulk = 0.0
                self.d_complex_carbon.copy_to_device(
                    np.full(self.shape, self.C_bulk, dtype=np.float32))
                self.d_substrate.copy_to_device(
                    np.full(self.shape, 0.0, dtype=np.float32))
            else:
                self.S_bulk = cfg.S_feed_concentration
                self.d_substrate.copy_to_device(
                    np.full(self.shape, self.S_bulk, dtype=np.float32))

            self.d_proton.copy_to_device(
                np.full(self.shape, cfg.init_H_mM, dtype=np.float32))

        # ── Movement and attachment ───────────────────────────────────────
        count_now = int(self.d_active_count.copy_to_host()[0])
        ab = self._agent_blocks(count_now)
        self._rebuild_cell_counts(count_now)
        planktonic_move_kernel[ab, self.threads](
            self.d_agent_pos, self.d_agent_type, self.d_active_count,
            cfg.N_grid, cfg.N_grid, cfg.N_grid,
            self.rng_states, cfg.max_voxels_per_step)
        attachment_kernel[ab, self.threads](
            self.d_agent_pos, self.d_agent_type, self.d_occupancy,
            self.d_active_count, cfg.prob_attach, self.rng_states)
        self._rebuild_cell_counts(count_now)

        # ── Zero rate grids ───────────────────────────────────────────────
        zeros = np.zeros(self.shape, dtype=np.float32)
        self.d_uptake_sp1.copy_to_device(zeros)
        self.d_uptake_total.copy_to_device(zeros)
        self.d_uptake_cmplx.copy_to_device(zeros)
        self.d_acetate_gen.copy_to_device(zeros)
        self.d_proton_gen_rate.copy_to_device(zeros)

        # ── Biofilm height estimate (for EET efficiency) ──────────────────
        if count_now > 0:
            pos_h  = self.d_agent_pos[:count_now].copy_to_host()
            type_h = self.d_agent_type[:count_now].copy_to_host()
            bf1    = (type_h == 1)
            L_biofilm_um = (float(np.max(pos_h[bf1, 2]))*cfg.dx*1e6
                            if bf1.any()
                            else (cfg.L_pili_base*1e6 + cfg.L_pili_growth_rate))
        else:
            L_biofilm_um = cfg.L_pili_base*1e6 + cfg.L_pili_growth_rate

        pili_now_um = (cfg.L_pili_base*1e6
                       + cfg.L_pili_growth_rate*(self.time/86400.0))
        if L_biofilm_um < pili_now_um + 0.5:
            L_biofilm_um = pili_now_um + 0.5

        # ── Biology kernel ────────────────────────────────────────────────
        # Passes complex_grid unconditionally; kernel ignores it when
        # mode_code != MODE_COMMENSAL.
        if count_now > 0:
            biology_kernel[ab, self.threads](
                self.d_agent_pos, self.d_agent_mass, self.d_agent_type,
                count_now, self.d_active_count, self.d_occupancy,
                self.d_substrate,       # acetate grid
                self.d_complex_carbon,  # complex carbon grid
                self.d_uptake_sp1,      # EET rate (current source)
                self.d_uptake_total,    # acetate sink (all consumers)
                self.d_acetate_gen,     # acetate production (commensal Org2)
                self.d_uptake_cmplx,    # complex-carbon sink (commensal Org2)
                self.d_proton,
                self.d_proton_gen_rate,
                cfg.dt_bio, cfg.mode_code,
                cfg.mu_max_sp1, cfg.k_decay_sp1, cfg.Ks_sp1, cfg.yield_sp1,
                cfg.mu_max_sp2, cfg.k_decay_sp2, cfg.Ks_sp2, cfg.yield_sp2,
                self.E_anode_actual, cfg.E_ka, cfg.F, cfg.R, cfg.T,
                cfg.mass_cell, cfg.max_agents, cfg.dV_liters, cfg.dx,
                cfg.init_pH, cfg.buffer_capacity,
                cfg.pH_optimal, cfg.pH_sigma,
                self.S_bulk, self.C_bulk, self.time,
                cfg.L_pili_base, cfg.L_pili_growth_rate,
                L_biofilm_um, cfg.max_cells_per_voxel, cfg.min_crowding_factor,
                cfg.mol_mass_sub, cfg.mol_mass_cmplx, cfg.yield_complex_to_acetate)

        # ── Circuit solve (Org1 EET only) ────────────────────────────────
        eet_arr   = self.d_uptake_sp1.copy_to_host()    # EET → current
        total_arr = self.d_uptake_total.copy_to_host()   # acetate sink → delta_S

        rate_eet_mM_s = float(np.sum(eet_arr))
        moles_eet_s   = (rate_eet_mM_s * cfg.dV_liters) / 1000.0
        
        scale = cfg.active_area_fraction * cfg.electrode_area_m2 / (cfg.L**2)
        # scale         = cfg.electrode_area_m2 / (cfg.L**2)
        # scale = 1
        I_bio_A       = moles_eet_s * scale * 8.0 * cfg.F

        age_days = self.time / 86400.0
        I_actual_A, eta_bv, E_anode_new, V_cell = \
            self._solve_circuit(I_bio_A, age_days)

        f_rise = 1.0 / (1.0 + math.exp(-3.0*(age_days - 2.0)))
        if self.last_feed == 0.0:
            dt_age = max(0.0, age_days - 3.0)
        else:
            dt_age = max(0.0, (self.time - self.last_feed)/86400.0 - 1.0)
        f_aging    = math.exp(-0.0226*dt_age)
        # V_measured = f_rise * f_aging * V_cell
        V_measured = V_cell

        self.E_anode_actual = E_anode_new
        tau_v = (cfg.voltage_tau_rise if V_measured >= self.voltage_smooth
                 else cfg.voltage_tau_decay)
        alpha_v = math.exp(-cfg.dt_bio / max(tau_v, cfg.dt_bio))
        self.voltage_smooth = (alpha_v*self.voltage_smooth
                               + (1.0-alpha_v)*V_measured)

        # ── Rescale proton generation to match accepted circuit current ───
        if I_bio_A > 1e-12:
            circuit_ratio = float(I_actual_A) / float(I_bio_A)
            pgen_h = self.d_proton_gen_rate.copy_to_host()
            pgen_h *= circuit_ratio
            self.d_proton_gen_rate.copy_to_device(pgen_h)

        # ── Cell death pruning ────────────────────────────────────────────
        count_raw = int(self.d_active_count.copy_to_host()[0])
        if count_raw > 0:
            masses = self.d_agent_mass[:count_raw].copy_to_host()
            alive  = np.where(masses > cfg.min_viable_mass_fraction*cfg.mass_cell)[0]
            new_n  = len(alive)
            if new_n < count_raw:
                pos_h  = self.d_agent_pos[:count_raw].copy_to_host()
                type_h = self.d_agent_type[:count_raw].copy_to_host()
                self.d_agent_pos[:new_n].copy_to_device(pos_h[alive])
                self.d_agent_mass[:new_n].copy_to_device(masses[alive])
                self.d_agent_type[:new_n].copy_to_device(type_h[alive])
                self.d_active_count.copy_to_device(
                    np.array([new_n], dtype=np.int32))
                count_raw = new_n

        # ── Bulk substrate / complex-carbon depletion ────────────────────
        rate_total_mM_s        = float(np.sum(total_arr))  # acetate sink
        moles_ace_s            = (rate_total_mM_s * cfg.dV_liters) / 1000.0
        moles_ace_consumed_s   = moles_ace_s * scale
        delta_S_consumed       = ((moles_ace_consumed_s * cfg.dt_bio * 1000.0)
                                  / cfg.reactor_vol_anode)

        if cfg.active_mode == "commensal":
            # Acetate bulk: subtract Org1 consumption, add Org2 production.
            C_before        = self.C_bulk
            ace_gen_arr      = self.d_acetate_gen.copy_to_host()
            rate_ace_gen_s   = float(np.sum(ace_gen_arr))   # [mM/L/s] production
            moles_ace_gen_s  = (rate_ace_gen_s * cfg.dV_liters) / 1000.0
            moles_ace_prod_s = moles_ace_gen_s * scale
            delta_S_produced = ((moles_ace_prod_s * cfg.dt_bio * 1000.0)
                                / cfg.reactor_vol_anode)

            # Complex carbon bulk: subtract Org2 consumption.
            # We derive Org2's complex-carbon molar consumption from the acetate
            # production rate via the stoichiometric yield.
            # ace_gen [mM acetate/L/s] = cmplx_consumed [mM cmplx/L/s] * yield_complex_to_acetate
            # → cmplx_consumed = ace_gen / yield_complex_to_acetate
            if cfg.yield_complex_to_acetate > 1e-12:
                cmplx_consumed_mM_s = rate_ace_gen_s / cfg.yield_complex_to_acetate
            else:
                cmplx_consumed_mM_s = 0.0
            moles_cmplx_s     = (cmplx_consumed_mM_s * cfg.dV_liters) / 1000.0
            moles_cmplx_cons  = moles_cmplx_s * scale
            delta_C_raw       = ((moles_cmplx_cons * cfg.dt_bio * 1000.0)
                                 / cfg.reactor_vol_anode)

            if delta_C_raw > C_before and delta_C_raw > 0.0:
                complex_scale = C_before / delta_C_raw
                delta_C = C_before
                delta_S_produced *= complex_scale

                ace_gen_arr *= complex_scale
                self.d_acetate_gen.copy_to_device(ace_gen_arr)

                cmplx_arr = self.d_uptake_cmplx.copy_to_host()
                cmplx_arr *= complex_scale
                self.d_uptake_cmplx.copy_to_device(cmplx_arr)
            else:
                delta_C = delta_C_raw

            self.S_bulk = max(0.0, self.S_bulk - delta_S_consumed + delta_S_produced)
            self.C_bulk = max(0.0, self.C_bulk - delta_C)
            if self.C_bulk <= 1e-12:
                self.d_complex_carbon.copy_to_device(zeros)
                self.d_uptake_cmplx.copy_to_device(zeros)

        else:
            # Single / competitive: acetate only.
            self.S_bulk = max(0.0, self.S_bulk - delta_S_consumed)

        # ── Proton migration rate at PEM face ────────────────────────────
        j_actual_density = I_actual_A / cfg.electrode_area_m2   # [A/m²]
        mig_rate = (cfg.t_H * j_actual_density / cfg.F) / cfg.dx

        # ── Diffusion solve ───────────────────────────────────────────────
        self._rebuild_cell_counts(count_raw)

        # Prepare net-rate grid for commensal acetate diffusion:
        # net_rate = uptake_total (sink) - acetate_gen (source)
        if cfg.active_mode == "commensal":
            ace_gen_arr_local = self.d_acetate_gen.copy_to_host()
            np.subtract(total_arr, ace_gen_arr_local,
                        out=self._h_net_acetate_rate)
            self.d_net_acetate_rate.copy_to_device(self._h_net_acetate_rate)

        # Initialise convergence snapshots
        self.d_substrate.copy_to_host(self._h_sub_prev)
        self.d_proton.copy_to_host(self._h_prot_prev)
        if cfg.active_mode == "commensal":
            self.d_complex_carbon.copy_to_host(self._h_cmplx_prev)

        check_interval = max(1, int(cfg.diffusion_check_interval))
        diffusion_converged = False
        for it in range(cfg.max_pseudo_iter):
            # Acetate diffusion — kernel choice depends on mode.
            if cfg.active_mode == "commensal":
                diffusion_kernel_acetate_commensal[
                    self.grid_blocks, self.grid_threads](
                    self.d_substrate, self.d_new_substrate,
                    self.d_net_acetate_rate, self.d_occupancy,
                    cfg._alpha_sub, cfg._pseudo_dt_sub, self.S_bulk,
                    cfg.max_cells_per_voxel, cfg.biofilm_diffusivity_ratio_sub)
            else:
                diffusion_kernel_consume[self.grid_blocks, self.grid_threads](
                    self.d_substrate, self.d_new_substrate,
                    self.d_uptake_total, self.d_occupancy,
                    cfg._alpha_sub, cfg._pseudo_dt_sub, self.S_bulk,
                    cfg.max_cells_per_voxel, cfg.biofilm_diffusivity_ratio_sub)
            self.d_substrate, self.d_new_substrate = (
                self.d_new_substrate, self.d_substrate)

            # Complex-carbon diffusion (commensal only)
            if cfg.active_mode == "commensal" and self.C_bulk > 1e-12:
                diffusion_kernel_complex[self.grid_blocks, self.grid_threads](
                    self.d_complex_carbon, self.d_new_complex_carbon,
                    self.d_uptake_cmplx, self.d_occupancy,
                    cfg._alpha_cmplx, cfg._pseudo_dt_cmplx, self.C_bulk,
                    cfg.max_cells_per_voxel, cfg.biofilm_diffusivity_ratio_cmplx)
                self.d_complex_carbon, self.d_new_complex_carbon = (
                    self.d_new_complex_carbon, self.d_complex_carbon)

            # Proton diffusion (all modes)
            diffusion_kernel_proton_pem[self.grid_blocks, self.grid_threads](
                self.d_proton, self.d_new_proton,
                self.d_proton_gen_rate, self.d_occupancy, cfg._alpha_h,
                cfg._pseudo_dt_h, cfg.k_m_PEM, cfg.dx, mig_rate,
                cfg.max_cells_per_voxel, cfg.biofilm_diffusivity_ratio_h)
            self.d_proton, self.d_new_proton = (
                self.d_new_proton, self.d_proton)

            # Convergence check every `diffusion_check_interval` iterations.
            if it % check_interval == check_interval - 1:
                self.d_substrate.copy_to_host(self._h_sub_cur)
                self.d_proton.copy_to_host(self._h_prot_cur)
                sub_conv  = (np.max(np.abs(
                    self._h_sub_cur - self._h_sub_prev)) < cfg.diffusion_tol)
                prot_conv = (np.max(np.abs(
                    self._h_prot_cur - self._h_prot_prev)) < cfg.diffusion_tol)
                cmplx_conv = True
                if cfg.active_mode == "commensal" and self.C_bulk > 1e-12:
                    self.d_complex_carbon.copy_to_host(self._h_cmplx_cur)
                    cmplx_conv = (np.max(np.abs(
                        self._h_cmplx_cur - self._h_cmplx_prev)) < cfg.diffusion_tol)
                if sub_conv and prot_conv and cmplx_conv:
                    diffusion_converged = True
                    break
                np.copyto(self._h_sub_prev,  self._h_sub_cur)
                np.copyto(self._h_prot_prev, self._h_prot_cur)
                if cfg.active_mode == "commensal" and self.C_bulk > 1e-12:
                    np.copyto(self._h_cmplx_prev, self._h_cmplx_cur)

        self.last_diffusion_iters = it + 1
        self.last_diffusion_hit_max = (
            not diffusion_converged
            and self.last_diffusion_iters >= cfg.max_pseudo_iter)

        # ── Mechanics ─────────────────────────────────────────────────────
        mb = self._agent_blocks(count_raw)
        self._rebuild_cell_counts(count_raw)
        for _ in range(cfg.shove_iterations):
            shove_kernel[mb, self.threads](
                self.d_agent_pos, self.d_agent_type, self.d_active_count,
                self.d_occupancy, cfg.max_cells_per_voxel,
                cfg.N_grid, cfg.N_grid, cfg.N_grid, self.rng_states)

        self.d_detached_count.copy_to_device(np.array([0], dtype=np.int32))
        shear_detachment_kernel[mb, self.threads](
            self.d_agent_pos, self.d_agent_type, self.d_active_count,
            self.d_occupancy, cfg.dt_bio, cfg.shear_rate,
            cfg.reference_shear_rate, cfg.detachment_rate,
            cfg.eps_cohesion, cfg.N_grid, cfg.N_grid, cfg.N_grid,
            self.rng_states, self.d_detached_count)

        # ── Record history ────────────────────────────────────────────────
        count_final = int(self.d_active_count.copy_to_host()[0])
        if count_final > 0:
            type_h = self.d_agent_type[:count_final].copy_to_host()
            n_sp1  = int(np.sum((type_h == 0) | (type_h == 1)))
            n_sp2  = int(np.sum((type_h == 2) | (type_h == 3)))
            n_bf1  = int(np.sum(type_h == 1))
            n_bf2  = int(np.sum(type_h == 3))
            n_bf   = n_bf1 + n_bf2
            f_EET  = (n_bf1 / n_bf) if n_bf > 0 else 0.0
        else:
            n_sp1 = n_sp2 = n_bf1 = n_bf2 = 0
            f_EET = 0.0

        self.last_n_sp1 = n_sp1
        self.last_n_sp2 = n_sp2
        self.last_n_bf1 = n_bf1
        self.last_n_bf2 = n_bf2

        max_H_mM = float(np.max(self.d_proton.copy_to_host()))

        self.history_n_sp1.append(n_sp1)
        self.history_n_sp2.append(n_sp2)
        self.history_n_bf1.append(n_bf1)
        self.history_n_bf2.append(n_bf2)
        self.history_substrate.append(self.S_bulk)
        self.history_complex.append(self.C_bulk)
        self.history_voltage.append(self.voltage_smooth)
        self.history_j_sp1.append(
            V_measured / cfg.R_ext / cfg.electrode_area_m2)
        self.history_min_pH.append(
            max(1.0, cfg.init_pH - max_H_mM / cfg.buffer_capacity))
        self.history_E_anode.append(self.E_anode_actual)
        self.history_f_EET.append(f_EET)

        self.time += cfg.dt_bio


# ===========================================================================
# SECTION 4 — EMPIRICAL DATA (Sun et al. 2008)
# ===========================================================================

EMPIRICAL_DATA = np.array([
    [0.215, 0.0171], [1.294, 0.0533], [1.567, 0.0997],
    [1.742, 0.1409], [1.873, 0.2680], [2.009, 0.3143],
    [2.160, 0.4242], [2.412, 0.5135], [2.680, 0.6440],
    [3.111, 0.6750], [4.035, 0.6785], [5.864, 0.6581],
    [8.184, 0.6395], [9.816, 0.6276], [10.227, 0.6774],
    [10.658, 0.6929], [12.094, 0.6776], [14.100, 0.6590],
    [15.673, 0.6420],
], dtype=np.float64)

df_emp = pd.DataFrame(EMPIRICAL_DATA, columns=['Time (Days)', 'Voltage (V)'])


# ===========================================================================
# SECTION 5 — RUN HELPERS
# ===========================================================================

def run_simulation(
        seed=42, sim_hours=384, plot_every=0, verbose=True,
        status_every=24, **config_kwargs):
    """
    Run a simulation with any active mode.

    Pass Config keywords via config_kwargs, e.g.:
        run_simulation(enable_species2=True, species2_role='commensal')
    """
    model = UnifiedMFCSimulation(random_seed=seed, **config_kwargs)
    cfg   = model.cfg

    n_steps = int(math.ceil(float(sim_hours) * 3600.0 / cfg.dt_bio))
    steps_per_feed = math.ceil(cfg.feeding_interval / cfg.dt_bio)
    snapshot_step = steps_per_feed - 1

    label = cfg.active_mode.upper()
    if verbose:
        print(f"Running {label} mode simulation ({sim_hours:g} hours)...")
        print(
            f"{SCRIPT_VERSION} | N_grid={cfg.N_grid} "
            f"area_scale={cfg.electrode_area_m2 / (cfg.L**2):.3g} "
            f"max_pseudo_iter={cfg.max_pseudo_iter} "
            f"diffusion_tol={cfg.diffusion_tol:g} "
            f"check_every={cfg.diffusion_check_interval} "
            f"voltage_tau_decay_h={cfg.voltage_tau_decay/3600:g} "
            f"min_mass={cfg.min_viable_mass_fraction:g}",
            flush=True)

    iterator = tqdm.tqdm(range(n_steps), disable=not verbose)
    for step_index in iterator:
        model.step()
        n_act = int(model.d_active_count.copy_to_host()[0])
        diff_note = f"diffusion_iters={model.last_diffusion_iters}"
        if model.last_diffusion_hit_max:
            diff_note += "/MAX"
        if verbose and (
                step_index == 0
                or (step_index + 1) % max(1, int(status_every)) == 0
                or step_index + 1 == n_steps):
            elapsed_hours = (step_index + 1) * cfg.dt_bio / 3600.0
            status = (
                f"hour={elapsed_hours:.1f}/{sim_hours:g} agents={n_act} "
                f"org1={model.last_n_sp1} org2={model.last_n_sp2} "
                f"bf1={model.last_n_bf1} bf2={model.last_n_bf2} "
                f"acetate={model.S_bulk:.6g} mM")
            if cfg.active_mode == "commensal":
                status += f" complex={model.C_bulk:.6g} mM"
            print(f"{status} {diff_note}", flush=True)

        if step_index == snapshot_step:
            model.spatial_snapshot = {
                'pos':     model.d_agent_pos[:n_act].copy_to_host(),
                'type':    model.d_agent_type[:n_act].copy_to_host(),
                'sub':     model.d_substrate.copy_to_host().copy(),
                'complex': model.d_complex_carbon.copy_to_host().copy(),
            }

        if plot_every and (step_index + 1) % int(plot_every) == 0:
            checkpoint_prefix = (
                f"Fig_{cfg.active_mode}_000_{step_index + 1:03d}")
            checkpoint_emp = df_emp if cfg.active_mode == "single" else None
            render_figures(model, df_emp=checkpoint_emp,
                           prefix=checkpoint_prefix)
    return model


def run_polarization_analysis(
        seed=43, sim_hours=240, output_dir=Path("polarization_outputs"),
        voltage_noise_sd=0.005, **config_kwargs):
    """Run one baseline simulation and export its polarization calculation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = run_simulation(
        seed=seed, sim_hours=sim_hours, verbose=True, **config_kwargs)
    polarization = model.calculate_polarization(
        voltage_noise_sd=voltage_noise_sd, random_seed=seed)
    csv_path = output_dir / "polarization_day10.csv"
    polarization.to_csv(csv_path, index=False)

    fig, ax_voltage = plt.subplots(figsize=(7.2, 4.8))
    ax_power = ax_voltage.twinx()
    x = polarization["current_density_mA_m2"]
    ax_voltage.plot(x, polarization["voltage_V"], color="tab:blue",
                    marker="o", markersize=3, label="Cell voltage")
    ax_power.plot(x, polarization["power_density_mW_m2"], color="tab:red",
                  marker="s", markersize=3, label="Power density")
    ax_voltage.set_xlabel("Current density (mA m$^{-2}$)")
    ax_voltage.set_ylabel("Cell voltage (V)", color="tab:blue")
    ax_power.set_ylabel("Power density (mW m$^{-2}$)", color="tab:red")
    ax_voltage.tick_params(axis="y", labelcolor="tab:blue")
    ax_power.tick_params(axis="y", labelcolor="tab:red")
    fig.tight_layout()
    figure_path = output_dir / "polarization_day10.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Polarization CSV: {csv_path}", flush=True)
    print(f"Polarization figure: {figure_path}", flush=True)
    return model, polarization


def compute_coulombic_efficiency(model):
    """
    Coulombic efficiency for the first batch.

    In single/competitive mode the theoretical charge is calculated from
    acetate consumed (S0 − S_end), identical to the original implementation.

    In commensal mode the acetate electron donor is generated in-situ, so
    we use the total moles of acetate ever converted to current (Q_act) and
    report CE against the maximum theoretically extractable charge from the
    total acetate produced (estimated from the complex-carbon consumed, which
    equals the acetate generation capacity).
    """
    cfg = model.cfg
    steps_per_feed = math.ceil(cfg.feeding_interval / cfg.dt_bio)
    cutoff         = min(steps_per_feed, len(model.history_substrate))

    Q_act = float(np.sum(np.array(model.history_j_sp1[:cutoff])
                         * cfg.electrode_area_m2 * cfg.dt_bio))

    if cfg.active_mode == "commensal":
        # Theoretical charge from all acetate that could have been produced.
        # history_complex[0] = C_bulk at step 1 ≈ C_feed; [cutoff-1] = C_bulk end.
        C0    = cfg.C_feed_concentration
        C_end = model.history_complex[cutoff-1] if cutoff > 0 else C0
        dC    = max(0.0, C0 - C_end)
        # mol acetate = mol complex × yield; each acetate → 8 electrons
        mol_ace_th = (dC / 1000.0) * cfg.reactor_vol_anode * cfg.yield_complex_to_acetate
        Q_th  = mol_ace_th * 8.0 * cfg.F
    else:
        S0    = cfg.S_feed_concentration
        S_end = model.history_substrate[cutoff-1] if cutoff > 0 else S0
        dS    = max(0.0, S0 - S_end)
        Q_th  = (dS / 1000.0) * cfg.reactor_vol_anode * 8.0 * cfg.F

    CE = (Q_act / Q_th * 100.0) if Q_th > 0 else 0.0
    print(f"CE = {CE:.1f}%  (Q_th={Q_th:.1f} C, Q_act={Q_act:.1f} C, "
          f"steps={cutoff}, mode={cfg.active_mode})")
    return CE


# ===========================================================================
# SECTION 6 — FIGURES
# ===========================================================================

def render_figures(model, df_emp=None, prefix=None):
    """
    Render all output figures.  Prefix defaults to the active mode name.
    df_emp (empirical data) is shown on the voltage figure only in single mode.
    """
    sns.set_theme(style='ticks', context='paper', font_scale=1.15)
    plt.rcParams.update({'axes.linewidth': 0.8, 'font.family': 'sans-serif'})

    cfg    = model.cfg
    mode   = cfg.active_mode
    prefix = prefix or f"Fig_{mode}"
    days   = np.arange(len(model.history_voltage)) * cfg.dt_bio / 86400.0

    # ── Fig 1: Voltage validation ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if df_emp is not None and mode == "single":
        ax.scatter(df_emp['Time (Days)'], df_emp['Voltage (V)'],
                   marker='s', s=40, facecolors='none', edgecolors='black',
                   linewidths=0.9, label='Sun et al. (2008)', zorder=5)
    label_sim = {'single': 'Single-species ABM',
                 'competitive': 'Competitive two-species ABM',
                 'commensal':   'Commensal two-species ABM'}.get(mode, mode)
    ax.plot(days, model.history_voltage,
            color='#C2185B', lw=2.0, label=label_sim)
    ax.axvline(10, color='#757575', ls='--', lw=0.8, alpha=0.7,
               label='Refeed (day 10)')
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Cell voltage (V)")
    ax.set_xlim(0, max(days)+0.5); ax.set_ylim(-0.02, 0.80)
    ax.legend(fontsize=8.5); ax.grid(alpha=0.15)
    sns.despine(fig=fig); plt.tight_layout()
    fname = f"{prefix}_voltage.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(); print(f"Saved {fname}")

    # ── Fig 2: Community fraction ─────────────────────────────────────────
    if cfg.has_species2:
        fig, ax = plt.subplots(figsize=(7, 4.0))
        n1 = np.array(model.history_n_sp1, dtype=float)
        n2 = np.array(model.history_n_sp2, dtype=float)
        tot = np.maximum(n1+n2, 1)
        sp2_label = ('Org2 — competitor' if mode == 'competitive'
                     else 'Org2 — commensal fermenter')
        ax.stackplot(days, n1/tot*100, n2/tot*100,
                     labels=['Org1 — electroactive', sp2_label],
                     colors=['#1565C0', '#E65100'], alpha=0.80)
        ax.axvline(10, color='#212121', ls='--', lw=0.8, alpha=0.6)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Community fraction (%)")
        ax.set_xlim(0, max(days)+0.5); ax.set_ylim(0, 100)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.15); sns.despine(fig=fig); plt.tight_layout()
        fname = f"{prefix}_community.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(); print(f"Saved {fname}")

    # ── Fig 3: Cell counts ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.0))
    n1 = np.array(model.history_n_sp1, dtype=float)
    ax.plot(days, n1/1e3, color='#1565C0', lw=1.8,
            label='Org1 — electroactive (×10³)')
    if cfg.has_species2:
        n2 = np.array(model.history_n_sp2, dtype=float)
        sp2_label = ('Org2 — competitor (×10³)' if mode == 'competitive'
                     else 'Org2 — commensal (×10³)')
        ax.plot(days, n2/1e3, color='#E65100', lw=1.8, ls='--',
                label=sp2_label)
    ax.axvline(10, color='#757575', ls='--', lw=0.8, alpha=0.7)
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Cell count (×10³)")
    ax.set_xlim(0, max(days)+0.5); ax.legend(fontsize=9)
    ax.grid(alpha=0.15); sns.despine(fig=fig); plt.tight_layout()
    fname = f"{prefix}_counts.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(); print(f"Saved {fname}")

    # ── Fig 4: Electroactive fraction ─────────────────────────────────────
    if cfg.has_species2:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.plot(days, np.array(model.history_f_EET)*100,
                color='#2E7D32', lw=2.0)
        ax.axvline(10, color='#757575', ls='--', lw=0.8, alpha=0.7,
                   label='Refeed (day 10)')
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Biofilm electroactive fraction (%)")
        ax.set_xlim(0, max(days)+0.5); ax.set_ylim(0, 100)
        ax.legend(fontsize=9); ax.grid(alpha=0.15)
        sns.despine(fig=fig); plt.tight_layout()
        fname = f"{prefix}_fEET.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(); print(f"Saved {fname}")

    # ── Fig 5: Substrate + pH ─────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(7, 3.8))
    S = np.array(model.history_substrate)
    ax1.plot(days, S, color='#E65100', lw=2.0, label='Bulk acetate')
    ax1.fill_between(days, S, alpha=0.12, color='#E65100')
    if mode == "commensal":
        C = np.array(model.history_complex)
        ax1.plot(days, C, color='#7B1FA2', lw=1.6, ls='--',
                 label='Bulk complex carbon')
    ax1.axvline(10, color='#757575', ls='--', lw=0.8, alpha=0.7)
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Concentration (mM)", color='#E65100')
    ax1.tick_params(axis='y', labelcolor='#E65100')
    ax2 = ax1.twinx()
    ax2.plot(days, model.history_min_pH, color='#6A1B9A', lw=1.5,
             ls=':', label='Min local pH')
    ax2.set_ylabel("Min local pH (−)", color='#6A1B9A')
    ax2.tick_params(axis='y', labelcolor='#6A1B9A')
    ax1.set_xlim(0, max(days)+0.5)
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labs1+labs2, fontsize=8.5, loc='center right')
    ax1.grid(alpha=0.15); sns.despine(fig=fig, right=False)
    plt.tight_layout()
    fname = f"{prefix}_substrate.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(); print(f"Saved {fname}")

    # ── Fig 6: 3-D spatial snapshot ───────────────────────────────────────
    if model.spatial_snapshot is not None:
        snap  = model.spatial_snapshot
        pos   = snap['pos']
        types = snap['type']
        mask1 = (types == 1)
        mask2 = (types == 3)

        fig = plt.figure(figsize=(6, 5))
        ax3 = fig.add_subplot(111, projection='3d')
        if mask1.any():
            p1 = pos[mask1]
            n_samp = min(len(p1), 30000)
            idx1 = np.random.default_rng(0).choice(len(p1), n_samp, replace=False)
            ax3.scatter(p1[idx1,0]*cfg.dx*1e6,
                        p1[idx1,1]*cfg.dx*1e6,
                        p1[idx1,2]*cfg.dx*1e6,
                        c='#1565C0', s=1.2, alpha=0.4,
                        label='Org1 (electroactive)')
        if mask2.any():
            p2 = pos[mask2]
            n_samp = min(len(p2), 30000)
            idx2 = np.random.default_rng(1).choice(len(p2), n_samp, replace=False)
            sp2_label_3d = ('Org2 (competitor)' if mode == 'competitive'
                            else 'Org2 (fermenter)')
            ax3.scatter(p2[idx2,0]*cfg.dx*1e6,
                        p2[idx2,1]*cfg.dx*1e6,
                        p2[idx2,2]*cfg.dx*1e6,
                        c='#E65100', s=1.2, alpha=0.4,
                        label=sp2_label_3d)
        ax3.set_xlabel("x (µm)", fontsize=9)
        ax3.set_ylabel("y (µm)", fontsize=9)
        ax3.set_zlabel("z (µm)", fontsize=9)
        ax3.legend(fontsize=8, loc='upper left')
        for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor('none')
        ax3.grid(False)
        ax3.view_init(elev=22, azim=40)
        plt.tight_layout()
        fname = f"{prefix}_3d.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(); print(f"Saved {fname}")

# ===========================================================================
# SECTION 7 — RESUMABLE ONE-AT-A-TIME SENSITIVITY ANALYSIS
# ===========================================================================

OAT_PARAMETER_SPECS = (
    {
        "parameter": "R_contact",
        "label": "Internal/contact resistance",
        "symbol": r"$R_{\mathrm{contact}}$",
        "config_keys": ("R_contact",),
    },
    {
        "parameter": "mu_max_sp1",
        "label": "Maximum specific growth rate",
        "symbol": r"$\mu_{\max}$",
        "config_keys": ("mu_max_sp1",),
    },
    {
        "parameter": "S_feed_concentration",
        "label": "Initial acetate concentration",
        "symbol": r"$S_0$",
        "config_keys": ("S_feed_concentration", "S_bulk"),
    },
    {
        "parameter": "i0_anode",
        "label": "Exchange current density",
        "symbol": r"$i_0$",
        "config_keys": ("i0_anode",),
    },
    {
        "parameter": "Ks_sp1",
        "label": "Half-saturation constant",
        "symbol": r"$K_s$",
        "config_keys": ("Ks_sp1",),
    },
    {
        "parameter": "D_s",
        "label": "Substrate diffusivity",
        "symbol": r"$D_s$",
        "config_keys": ("D_s",),
    },
    {
        "parameter": "L_pili_base",
        "label": "Initial pili length",
        "symbol": r"$L_{\mathrm{pili},0}$",
        "config_keys": ("L_pili_base",),
    },
    {
        "parameter": "prob_attach",
        "label": "Biofilm attachment probability",
        "symbol": r"$P_{\mathrm{attach}}$",
        "config_keys": ("prob_attach",),
    },
    {
        "parameter": "D_h_PEM",
        "label": "PEM proton diffusivity",
        "symbol": r"$D_{\mathrm{H^+,PEM}}$",
        "config_keys": ("D_h_PEM",),
    },
)

OAT_OUTPUT_METRICS = (
    "max_voltage_V",
    "time_to_max_voltage_day",
    "mean_voltage_V",
    "final_voltage_V",
    "max_current_A",
    "final_current_A",
    "max_power_W",
    "final_power_W",
    "total_charge_C",
    "total_energy_J",
    "CE_percent",
    "final_total_agents",
    "final_bf1_agents",
    "final_bulk_substrate_mM",
    "final_biofilm_avg_thickness_um",
    "final_biofilm_occupied_thickness_um",
    "final_biofilm_coverage_fraction",
    "final_biofilm_pH_mean",
    "final_biofilm_substrate_mean_mM",
)


def _default_oat_output_dir():
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working / "sensitivity_outputs"
    return Path.cwd() / "sensitivity_outputs"


def _json_compatible(value):
    """Convert NumPy values and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_compatible(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, payload):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(_json_compatible(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _trapezoidal_integral(values, coordinates):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, x=coordinates)
    return np.trapz(values, x=coordinates)


def _serializable_config(config):
    scalar_types = (str, bool, int, float, np.integer, np.floating)
    return {
        key: _json_compatible(value)
        for key, value in vars(config).items()
        if isinstance(value, scalar_types)
    }


def _final_spatial_metrics(model):
    """Calculate final biofilm summaries once, after a completed case."""
    cfg = model.cfg
    n_active = int(model.d_active_count.copy_to_host()[0])
    empty = {
        "final_biofilm_count": 0,
        "final_biofilm_avg_thickness_um": 0.0,
        "final_biofilm_occupied_thickness_um": 0.0,
        "final_biofilm_coverage_fraction": 0.0,
        "final_biofilm_pH_mean": np.nan,
        "final_biofilm_substrate_mean_mM": np.nan,
    }
    if n_active <= 0:
        return empty

    positions = model.d_agent_pos[:n_active].copy_to_host()
    agent_types = model.d_agent_type[:n_active].copy_to_host()
    biofilm_mask = (agent_types == 1) | (agent_types == 3)
    if not np.any(biofilm_mask):
        return empty

    biofilm_positions = positions[biofilm_mask]
    lateral_index = (
        biofilm_positions[:, 0].astype(np.int64) * cfg.N_grid
        + biofilm_positions[:, 1].astype(np.int64)
    )
    unique_columns, inverse = np.unique(lateral_index, return_inverse=True)
    column_top_voxels = np.zeros(len(unique_columns), dtype=np.int32)
    np.maximum.at(column_top_voxels, inverse, biofilm_positions[:, 2] + 1)

    voxel_height_um = cfg.dx * 1e6
    occupied_thickness_um = float(
        np.mean(column_top_voxels) * voxel_height_um)
    coverage_fraction = float(len(unique_columns) / (cfg.N_grid ** 2))
    full_area_thickness_um = occupied_thickness_um * coverage_fraction

    substrate = model.d_substrate.copy_to_host()
    proton = model.d_proton.copy_to_host()
    x = biofilm_positions[:, 0]
    y = biofilm_positions[:, 1]
    z = biofilm_positions[:, 2]
    local_substrate = substrate[x, y, z].astype(np.float64)
    local_proton = proton[x, y, z].astype(np.float64)
    local_pH = np.maximum(
        1.0, cfg.init_pH - local_proton / cfg.buffer_capacity)

    return {
        "final_biofilm_count": int(np.sum(biofilm_mask)),
        "final_biofilm_avg_thickness_um": full_area_thickness_um,
        "final_biofilm_occupied_thickness_um": occupied_thickness_um,
        "final_biofilm_coverage_fraction": coverage_fraction,
        "final_biofilm_pH_mean": float(np.mean(local_pH)),
        "final_biofilm_substrate_mean_mM": float(np.mean(local_substrate)),
    }


def _case_metrics(model, coulombic_efficiency_percent):
    timeseries = model.to_timeseries_dataframe().copy()
    if timeseries.empty:
        raise RuntimeError("Simulation returned an empty time series.")

    time_days = timeseries["time_days"].to_numpy(dtype=np.float64)
    voltage = timeseries["voltage_V"].to_numpy(dtype=np.float64)
    current = timeseries["current_A"].to_numpy(dtype=np.float64)
    power = timeseries["power_W"].to_numpy(dtype=np.float64)
    peak_index = int(np.nanargmax(voltage))
    time_seconds = time_days * 86400.0

    current_valid = np.isfinite(time_seconds) & np.isfinite(current)
    power_valid = np.isfinite(time_seconds) & np.isfinite(power)
    total_charge = (
        float(_trapezoidal_integral(
            current[current_valid], time_seconds[current_valid]))
        if np.count_nonzero(current_valid) >= 2 else np.nan
    )
    total_energy = (
        float(_trapezoidal_integral(
            power[power_valid], time_seconds[power_valid]))
        if np.count_nonzero(power_valid) >= 2 else np.nan
    )

    final = timeseries.iloc[-1]
    metrics = {
        "max_voltage_V": float(voltage[peak_index]),
        "time_to_max_voltage_day": float(time_days[peak_index]),
        "mean_voltage_V": float(np.nanmean(voltage)),
        "final_voltage_V": float(final["voltage_V"]),
        "max_current_A": float(np.nanmax(current)),
        "final_current_A": float(final["current_A"]),
        "max_power_W": float(np.nanmax(power)),
        "final_power_W": float(final["power_W"]),
        "total_charge_C": total_charge,
        "total_energy_J": total_energy,
        "total_energy_Wh": total_energy / 3600.0,
        "CE_percent": float(coulombic_efficiency_percent),
        "final_agents_sp1": int(final["n_sp1"]),
        "final_agents_sp2": int(final["n_sp2"]),
        "final_bf1_agents": int(final["n_bf1"]),
        "final_bf2_agents": int(final["n_bf2"]),
        "final_total_agents": int(final["n_sp1"] + final["n_sp2"]),
        "final_bulk_substrate_mM": float(final["bulk_substrate_mM"]),
        "final_domain_min_pH": float(final["domain_min_pH"]),
        "final_E_anode_V": float(final["E_anode_V"]),
        "final_electroactive_fraction": float(final["f_EET"]),
    }
    metrics.update(_final_spatial_metrics(model))
    return timeseries, metrics


def _select_oat_parameters(parameter_names=None):
    if parameter_names is None:
        return list(OAT_PARAMETER_SPECS)

    requested = {str(name).strip() for name in parameter_names if str(name).strip()}
    known = {specification["parameter"] for specification in OAT_PARAMETER_SPECS}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(
            "Unknown OAT parameter(s): " + ", ".join(unknown)
            + ". Available: " + ", ".join(sorted(known)))
    return [specification for specification in OAT_PARAMETER_SPECS
            if specification["parameter"] in requested]


def _build_oat_run_specs(base_config, base_kwargs, parameters, variation):
    run_specs = [{
        "case_id": "baseline",
        "parameter": "baseline",
        "label": "Baseline",
        "symbol": "Baseline",
        "direction": "baseline",
        "factor": 1.0,
        "actual_factor": 1.0,
        "base_value": np.nan,
        "test_value": np.nan,
        "config_kwargs": dict(base_kwargs),
    }]

    for specification in parameters:
        parameter = specification["parameter"]
        base_value = float(getattr(base_config, parameter))
        for direction, factor in (
                ("minus30", 1.0 - variation),
                ("plus30", 1.0 + variation)):
            test_value = base_value * factor
            if parameter == "prob_attach":
                test_value = min(max(test_value, 0.0), 1.0)
            actual_factor = (
                test_value / base_value if abs(base_value) > 1e-30
                else np.nan)
            config_kwargs = dict(base_kwargs)
            for config_key in specification["config_keys"]:
                config_kwargs[config_key] = test_value

            run_specs.append({
                "case_id": f"{parameter}_{direction}",
                "parameter": parameter,
                "label": specification["label"],
                "symbol": specification["symbol"],
                "direction": direction,
                "factor": factor,
                "actual_factor": actual_factor,
                "base_value": base_value,
                "test_value": test_value,
                "config_kwargs": config_kwargs,
            })
    return run_specs


def _case_metadata(specification):
    return {
        key: value for key, value in specification.items()
        if key != "config_kwargs"
    }


def _create_oat_plots(successful, parameters, figures_dir, variation):
    perturbations = successful.loc[
        successful["parameter"] != "baseline"].copy()
    ordered_parameters = [item["parameter"] for item in parameters]
    labels = {item["parameter"]: item["label"] for item in parameters}
    decrease_label = f"Parameter decreased by {variation * 100:.0f}%"
    increase_label = f"Parameter increased by {variation * 100:.0f}%"
    bounded_increase = np.any(
        (perturbations["direction"] == "plus30")
        & ~np.isclose(
            perturbations["actual_factor"].astype(float),
            1.0 + variation,
            rtol=1e-10,
            atol=1e-12,
        )
    )
    if bounded_increase:
        increase_label += " (bounded for probability)"
    colors = ("#0072B2", "#D55E00")

    def values_for(column):
        minus_values = []
        plus_values = []
        axis_labels = []
        for parameter in ordered_parameters:
            subset = perturbations.loc[
                perturbations["parameter"] == parameter]
            low = subset.loc[subset["direction"] == "minus30"]
            high = subset.loc[subset["direction"] == "plus30"]
            minus_values.append(
                float(low[column].iloc[0]) if not low.empty else np.nan)
            plus_values.append(
                float(high[column].iloc[0]) if not high.empty else np.nan)
            axis_labels.append(labels[parameter])
        return axis_labels, np.asarray(minus_values), np.asarray(plus_values)

    def draw_panel(axis, column, xlabel, panel_label=None):
        axis_labels, minus_values, plus_values = values_for(column)
        y = np.arange(len(axis_labels))
        height = 0.36
        axis.barh(
            y + height / 2, minus_values, height,
            color=colors[0], label=decrease_label)
        axis.barh(
            y - height / 2, plus_values, height,
            color=colors[1], label=increase_label)
        axis.axvline(0.0, color="#333333", linewidth=0.8)
        axis.set_yticks(y)
        axis.set_yticklabels(axis_labels)
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", color="#D0D0D0", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if panel_label:
            axis.text(
                -0.02, 1.02, panel_label, transform=axis.transAxes,
                fontsize=12, fontweight="bold", va="bottom", ha="right")

    sns.set_theme(style="ticks", context="paper", font_scale=1.05)
    plot_specs = (
        (
            "max_voltage_V_relative_change_percent",
            "Relative change in maximum voltage (%)",
            "sensitivity_max_voltage.png",
        ),
        (
            "CE_change_percentage_points",
            "Change in coulombic efficiency (percentage points)",
            "sensitivity_CE.png",
        ),
    )

    for column, xlabel, filename in plot_specs:
        figure, axis = plt.subplots(figsize=(8.0, 5.3))
        draw_panel(axis, column, xlabel)
        axis.legend(frameon=False, loc="best")
        figure.tight_layout()
        figure.savefig(
            figures_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(figure)

    figure, axes = plt.subplots(
        1, 2, figsize=(14.0, 5.7), sharey=True,
        gridspec_kw={"wspace": 0.08})
    draw_panel(
        axes[0], "max_voltage_V_relative_change_percent",
        "Relative change in maximum voltage (%)", "A")
    draw_panel(
        axes[1], "CE_change_percentage_points",
        "Change in coulombic efficiency (percentage points)", "B")
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, legend_labels, loc="lower center", ncol=2,
        frameon=False, bbox_to_anchor=(0.5, -0.01))
    figure.subplots_adjust(bottom=0.17, left=0.22, right=0.98, top=0.96)
    figure.savefig(
        figures_dir / "sensitivity_two_panel.png",
        dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_oat_sensitivity_analysis(
        sim_hours=240,
        seed=43,
        variation=0.30,
        output_dir=None,
        overwrite=False,
        base_config_overrides=None,
        parameter_names=None,
        verbose=True):
    """
    Run a resumable baseline and +/- one-at-a-time sensitivity analysis.

    All cases use the same random seed so each response isolates a parameter
    perturbation from variation in stochastic initialization. Coulombic
    efficiency is the first-batch value returned by
    ``compute_coulombic_efficiency``; other summaries cover the full run.
    """
    if not (0.0 < float(variation) < 1.0):
        raise ValueError("variation must be between 0 and 1.")
    if float(sim_hours) <= 0:
        raise ValueError("sim_hours must be positive.")

    output_root = Path(
        output_dir if output_dir is not None else _default_oat_output_dir()
    ).expanduser().resolve()
    cases_dir = output_root / "cases"
    logs_dir = output_root / "logs"
    figures_dir = output_root / "figures"
    for directory in (output_root, cases_dir, logs_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    job_result_path = output_root.parent / "JOB_RESULT.json"
    parameters = _select_oat_parameters(parameter_names)
    base_kwargs = dict(base_config_overrides or {})
    base_config = Config(**base_kwargs)
    run_specs = _build_oat_run_specs(
        base_config, base_kwargs, parameters, float(variation))

    _write_json(
        output_root / "baseline_config.json",
        _serializable_config(base_config))
    manifest_rows = []
    for specification in run_specs:
        row = _case_metadata(specification)
        row["seed"] = int(seed)
        row["sim_hours"] = float(sim_hours)
        row["config_overrides_json"] = json.dumps(
            _json_compatible(specification["config_kwargs"]),
            sort_keys=True)
        manifest_rows.append(row)
    pd.DataFrame(manifest_rows).to_csv(
        output_root / "run_manifest.csv", index=False)

    _write_json(job_result_path, {
        "status": "running",
        "script_version": SCRIPT_VERSION,
        "seed": int(seed),
        "sim_hours": float(sim_hours),
        "variation": float(variation),
        "output_dir": output_root,
        "completed_cases": 0,
        "total_cases": len(run_specs),
    })

    result_rows = []
    timeseries_frames = []
    failure_rows = []

    for run_number, specification in enumerate(run_specs, start=1):
        case_id = specification["case_id"]
        case_dir = cases_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        timeseries_path = case_dir / "timeseries.csv"
        metrics_path = case_dir / "metrics.json"
        complete_path = case_dir / "COMPLETE"
        failed_path = case_dir / "FAILED"
        log_path = logs_dir / f"{case_id}.log"

        print(
            f"[{run_number}/{len(run_specs)}] {case_id}",
            flush=True)

        if (
                not overwrite
                and complete_path.exists()
                and timeseries_path.exists()
                and metrics_path.exists()):
            try:
                saved_timeseries = pd.read_csv(timeseries_path)
                saved_metrics = json.loads(
                    metrics_path.read_text(encoding="utf-8"))
                saved_metrics = {
                    key: (np.nan if value is None else value)
                    for key, value in saved_metrics.items()
                }
                timeseries_frames.append(saved_timeseries)
                result_rows.append({
                    **_case_metadata(specification),
                    **saved_metrics,
                    "status": "ok",
                    "error": "",
                    "resumed": True,
                })
                print("  reused completed case", flush=True)
                continue
            except Exception as resume_error:
                print(
                    f"  checkpoint unreadable; rerunning: {resume_error!r}",
                    flush=True)

        if complete_path.exists():
            complete_path.unlink()
        started = walltime.time()
        case_config = {
            "script_version": SCRIPT_VERSION,
            "case_id": case_id,
            "seed": int(seed),
            "sim_hours": float(sim_hours),
            "variation": float(variation),
            **_case_metadata(specification),
            "config_kwargs": specification["config_kwargs"],
        }
        _write_json(case_dir / "config.json", case_config)

        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                with contextlib.redirect_stdout(log_file), \
                        contextlib.redirect_stderr(log_file):
                    print(json.dumps(
                        _json_compatible(case_config), indent=2), flush=True)
                    model = run_simulation(
                        seed=int(seed),
                        sim_hours=float(sim_hours),
                        plot_every=0,
                        verbose=True,
                        status_every=24,
                        **specification["config_kwargs"])
                    ce_percent = compute_coulombic_efficiency(model)
                    timeseries, metrics = _case_metrics(model, ce_percent)

            runtime_seconds = walltime.time() - started
            metrics["runtime_seconds"] = float(runtime_seconds)
            timeseries.insert(0, "seed", int(seed))
            timeseries.insert(1, "case_id", case_id)
            timeseries.insert(2, "parameter", specification["parameter"])
            timeseries.insert(3, "direction", specification["direction"])
            timeseries.insert(4, "base_value", specification["base_value"])
            timeseries.insert(5, "test_value", specification["test_value"])
            timeseries.insert(
                6, "actual_factor", specification["actual_factor"])

            temporary_timeseries = case_dir / "timeseries.csv.tmp"
            timeseries.to_csv(temporary_timeseries, index=False)
            temporary_timeseries.replace(timeseries_path)
            _write_json(metrics_path, metrics)
            complete_path.write_text(
                f"case_id={case_id}\nstatus=complete\n"
                f"runtime_seconds={runtime_seconds:.6f}\n",
                encoding="utf-8")
            if failed_path.exists():
                failed_path.unlink()

            result_rows.append({
                **_case_metadata(specification),
                **metrics,
                "status": "ok",
                "error": "",
                "resumed": False,
            })
            timeseries_frames.append(timeseries)
            print(
                f"  complete: Vmax={metrics['max_voltage_V']:.6f} V, "
                f"CE={metrics['CE_percent']:.4f}%, "
                f"runtime={runtime_seconds / 3600.0:.3f} h",
                flush=True)
            del model
            gc.collect()

        except Exception as error:
            failure = {
                **_case_metadata(specification),
                "status": "failed",
                "error": repr(error),
                "runtime_seconds": walltime.time() - started,
                "resumed": False,
            }
            result_rows.append(failure)
            failure_rows.append(failure)
            failed_path.write_text(traceback.format_exc(), encoding="utf-8")
            print(f"  FAILED: {error!r}", flush=True)

        partial = pd.DataFrame(result_rows)
        partial.to_csv(
            output_root / "sensitivity_raw_metrics_partial.csv",
            index=False)
        if failure_rows:
            pd.DataFrame(failure_rows).to_csv(
                output_root / "failed_cases.csv", index=False)
        _write_json(job_result_path, {
            "status": "running",
            "script_version": SCRIPT_VERSION,
            "seed": int(seed),
            "completed_cases": int(np.sum(partial["status"] == "ok")),
            "failed_cases": int(np.sum(partial["status"] == "failed")),
            "total_cases": len(run_specs),
            "last_case": case_id,
            "output_dir": output_root,
        })

    raw_metrics = pd.DataFrame(result_rows)
    raw_metrics.to_csv(
        output_root / "sensitivity_raw_metrics.csv", index=False)
    successful = raw_metrics.loc[raw_metrics["status"] == "ok"].copy()
    if successful.empty:
        raise RuntimeError("No OAT case completed successfully.")

    if timeseries_frames:
        all_timeseries = pd.concat(timeseries_frames, ignore_index=True)
        all_timeseries.to_csv(
            output_root / "all_timeseries.csv", index=False)
        try:
            all_timeseries.to_parquet(
                output_root / "all_timeseries.parquet", index=False)
        except Exception as parquet_error:
            print(
                f"Parquet output skipped: {parquet_error!r}", flush=True)
    else:
        all_timeseries = pd.DataFrame()

    baseline_rows = successful.loc[
        successful["parameter"] == "baseline"]
    if baseline_rows.empty:
        raise RuntimeError("The baseline case did not complete successfully.")
    baseline = baseline_rows.iloc[0]

    available_metrics = [
        metric for metric in OAT_OUTPUT_METRICS
        if metric in successful.columns]
    for metric in available_metrics:
        baseline_value = float(baseline[metric])
        relative_column = f"{metric}_relative_change_percent"
        if np.isfinite(baseline_value) and abs(baseline_value) > 1e-15:
            successful[relative_column] = (
                100.0
                * (successful[metric].astype(float) - baseline_value)
                / baseline_value)
        else:
            successful[relative_column] = np.nan
    successful["CE_change_percentage_points"] = (
        successful["CE_percent"].astype(float)
        - float(baseline["CE_percent"]))
    successful.to_csv(
        output_root / "sensitivity_relative_changes.csv", index=False)

    summary_rows = []
    for specification in parameters:
        parameter = specification["parameter"]
        subset = successful.loc[successful["parameter"] == parameter]
        low_rows = subset.loc[subset["direction"] == "minus30"]
        high_rows = subset.loc[subset["direction"] == "plus30"]
        if low_rows.empty or high_rows.empty:
            continue
        low = low_rows.iloc[0]
        high = high_rows.iloc[0]
        factor_span = float(high["actual_factor"] - low["actual_factor"])
        if abs(factor_span) < 1e-15:
            continue

        summary = {
            "parameter": parameter,
            "label": specification["label"],
            "symbol": specification["symbol"],
            "base_value": float(low["base_value"]),
            "minus30_value": float(low["test_value"]),
            "plus30_value": float(high["test_value"]),
        }
        for metric in available_metrics:
            baseline_value = float(baseline[metric])
            low_value = float(low[metric])
            high_value = float(high[metric])
            sensitivity = (
                (high_value - low_value) / baseline_value / factor_span
                if np.isfinite(baseline_value)
                and abs(baseline_value) > 1e-15 else np.nan)
            summary[f"sensitivity_{metric}"] = sensitivity
            summary[
                f"minus30_{metric}_relative_change_percent"] = low.get(
                    f"{metric}_relative_change_percent", np.nan)
            summary[
                f"plus30_{metric}_relative_change_percent"] = high.get(
                    f"{metric}_relative_change_percent", np.nan)
        summary["minus30_CE_change_percentage_points"] = float(
            low["CE_change_percentage_points"])
        summary["plus30_CE_change_percentage_points"] = float(
            high["CE_change_percentage_points"])
        summary_rows.append(summary)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "sensitivity_summary.csv", index=False)
    if not summary.empty:
        voltage_ranking = summary.copy()
        voltage_ranking["absolute_sensitivity"] = voltage_ranking[
            "sensitivity_max_voltage_V"].abs()
        voltage_ranking.sort_values(
            "absolute_sensitivity", ascending=False).to_csv(
                output_root / "sensitivity_ranked_voltage.csv", index=False)

        ce_ranking = summary.copy()
        ce_ranking["absolute_sensitivity"] = ce_ranking[
            "sensitivity_CE_percent"].abs()
        ce_ranking.sort_values(
            "absolute_sensitivity", ascending=False).to_csv(
                output_root / "sensitivity_ranked_CE.csv", index=False)

    complete_parameter_pairs = {
        parameter for parameter in successful["parameter"].unique()
        if parameter != "baseline"
        and set(successful.loc[
            successful["parameter"] == parameter, "direction"])
        >= {"minus30", "plus30"}
    }
    plot_parameters = [
        specification for specification in parameters
        if specification["parameter"] in complete_parameter_pairs]
    if plot_parameters:
        _create_oat_plots(
            successful, plot_parameters, figures_dir, float(variation))

    archive_path = Path(shutil.make_archive(
        base_name=str(output_root.parent / output_root.name),
        format="zip",
        root_dir=str(output_root)))
    completed_count = int(np.sum(raw_metrics["status"] == "ok"))
    failed_count = int(np.sum(raw_metrics["status"] == "failed"))
    final_status = "complete" if failed_count == 0 else "partial"
    _write_json(job_result_path, {
        "status": final_status,
        "script_version": SCRIPT_VERSION,
        "seed": int(seed),
        "sim_hours": float(sim_hours),
        "variation": float(variation),
        "completed_cases": completed_count,
        "failed_cases": failed_count,
        "total_cases": len(run_specs),
        "output_dir": output_root,
        "archive_name": archive_path.name,
        "archive_path": archive_path,
        "archive_size_bytes": archive_path.stat().st_size,
    })

    print("\nOAT sensitivity analysis finished.", flush=True)
    print(
        f"Completed cases: {completed_count}/{len(run_specs)}",
        flush=True)
    print(f"Output directory: {output_root}", flush=True)
    print(f"Archive: {archive_path}", flush=True)
    return {
        "raw_metrics": raw_metrics,
        "successful_metrics": successful,
        "summary": summary,
        "all_timeseries": all_timeseries,
        "archive_path": str(archive_path),
        "status": final_status,
    }


def run_single_oat_case_job(
        case_id,
        sim_hours=240,
        seed=43,
        variation=0.30,
        output_dir=None,
        base_config_overrides=None,
        cleanup_unpacked=None):
    """Run and archive exactly one OAT case for a Kaggle queue job."""
    if not (0.0 < float(variation) < 1.0):
        raise ValueError("variation must be between 0 and 1.")
    if float(sim_hours) <= 0:
        raise ValueError("sim_hours must be positive.")

    output_root = Path(
        output_dir if output_dir is not None else _default_oat_output_dir()
    ).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    job_result_path = output_root.parent / "JOB_RESULT.json"
    if cleanup_unpacked is None:
        cleanup_unpacked = output_root.parent == Path("/kaggle/working")

    base_kwargs = dict(base_config_overrides or {})
    base_config = Config(**base_kwargs)
    run_specs = _build_oat_run_specs(
        base_config,
        base_kwargs,
        list(OAT_PARAMETER_SPECS),
        float(variation),
    )
    specification_by_id = {
        specification["case_id"]: specification
        for specification in run_specs
    }
    if case_id not in specification_by_id:
        raise ValueError(
            f"Unknown OAT case '{case_id}'. Available cases: "
            + ", ".join(specification_by_id))
    specification = specification_by_id[case_id]
    safe_case_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)
    archive_base = output_root.parent / f"oat_{safe_case_id}_results"
    archive_path = Path(str(archive_base) + ".zip")
    log_path = output_root / "run.log"
    config_path = output_root / "config.json"
    metrics_path = output_root / "metrics.json"
    timeseries_path = output_root / "timeseries.csv"
    complete_path = output_root / "COMPLETE"

    if archive_path.exists():
        archive_path.unlink()
    if complete_path.exists():
        complete_path.unlink()

    case_config = {
        "script_version": SCRIPT_VERSION,
        "case_id": case_id,
        "seed": int(seed),
        "sim_hours": float(sim_hours),
        "simulation_days": float(sim_hours) / 24.0,
        "variation": float(variation),
        **_case_metadata(specification),
        "config_kwargs": specification["config_kwargs"],
    }
    _write_json(config_path, case_config)
    _write_json(job_result_path, {
        "status": "running",
        "script_version": SCRIPT_VERSION,
        "case_id": case_id,
        "seed": int(seed),
        "sim_hours": float(sim_hours),
    })

    started = walltime.time()
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            with contextlib.redirect_stdout(log_file), \
                    contextlib.redirect_stderr(log_file):
                print(json.dumps(
                    _json_compatible(case_config), indent=2), flush=True)
                model = run_simulation(
                    seed=int(seed),
                    sim_hours=float(sim_hours),
                    plot_every=0,
                    verbose=True,
                    status_every=24,
                    **specification["config_kwargs"],
                )
                ce_percent = compute_coulombic_efficiency(model)
                timeseries, metrics = _case_metrics(model, ce_percent)

        elapsed_seconds = walltime.time() - started
        metrics["runtime_seconds"] = float(elapsed_seconds)
        timeseries.insert(0, "seed", int(seed))
        timeseries.insert(1, "case_id", case_id)
        timeseries.insert(2, "parameter", specification["parameter"])
        timeseries.insert(3, "direction", specification["direction"])
        timeseries.insert(4, "base_value", specification["base_value"])
        timeseries.insert(5, "test_value", specification["test_value"])
        timeseries.insert(
            6, "actual_factor", specification["actual_factor"])
        timeseries.to_csv(timeseries_path, index=False)
        _write_json(metrics_path, metrics)
        complete_path.write_text(
            f"case_id={case_id}\nstatus=complete\n"
            f"runtime_seconds={elapsed_seconds:.6f}\n",
            encoding="utf-8",
        )

        archive_path = Path(shutil.make_archive(
            base_name=str(archive_base),
            format="zip",
            root_dir=str(output_root),
        ))
        payload = {
            "status": "complete",
            "script_version": SCRIPT_VERSION,
            "case_id": case_id,
            "parameter": specification["parameter"],
            "direction": specification["direction"],
            "seed": int(seed),
            "sim_hours": float(sim_hours),
            "simulation_days": float(sim_hours) / 24.0,
            "elapsed_wall_hours": elapsed_seconds / 3600.0,
            "archive_name": archive_path.name,
            "archive_size_bytes": archive_path.stat().st_size,
        }
        _write_json(job_result_path, payload)
        if cleanup_unpacked:
            shutil.rmtree(output_root)
        print(
            f"OAT_CASE_COMPLETE case={case_id} "
            f"elapsed_hours={elapsed_seconds / 3600.0:.3f} "
            f"archive={archive_path.name}",
            flush=True,
        )
        return payload

    except Exception as error:
        payload = {
            "status": "failed",
            "script_version": SCRIPT_VERSION,
            "case_id": case_id,
            "seed": int(seed),
            "sim_hours": float(sim_hours),
            "elapsed_wall_hours": (walltime.time() - started) / 3600.0,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(job_result_path, payload)
        print(f"OAT_CASE_FAILED case={case_id}: {error!r}", flush=True)
        raise


def sensitivity_analysis(data=None, **kwargs):
    """Backward-compatible wrapper for the former sensitivity function."""
    result = run_oat_sensitivity_analysis(**kwargs)
    return result["raw_metrics"], result["summary"]


# ===========================================================================
# SECTION 8 — KAGGLE AUTORUN / COMMAND-LINE ENTRY POINT
# ===========================================================================

KAGGLE_SENSITIVITY_SEED = 43
KAGGLE_SENSITIVITY_HOURS = 240
KAGGLE_SENSITIVITY_VARIATION = 0.30
KAGGLE_OAT_CASE_ID = "D_h_PEM_plus30"
def _parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run the resumable CUDA MFC one-at-a-time sensitivity analysis."))
    parser.add_argument(
        "--sim-hours", type=float, default=KAGGLE_SENSITIVITY_HOURS)
    parser.add_argument(
        "--seed", type=int, default=KAGGLE_SENSITIVITY_SEED)
    parser.add_argument(
        "--variation", type=float, default=KAGGLE_SENSITIVITY_VARIATION)
    parser.add_argument(
        "--output-dir", type=Path, default=_default_oat_output_dir())
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Rerun cases even when their COMPLETE markers are valid.")
    parser.add_argument(
        "--parameters", default=None,
        help="Comma-separated subset of Config parameter names.")
    parser.add_argument(
        "--case-id", default=None,
        help="Run and archive only this OAT case.")
    parser.add_argument(
        "--all-cases", action="store_true",
        help="Run every selected OAT case in the current process.")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run five reduced cases for an inexpensive interface check.")
    parser.add_argument(
        "--polarization", action="store_true",
        help=("Run one baseline simulation and export an instantaneous "
              "polarization/power-density sweep instead of the OAT cases."))
    parser.add_argument(
        "--polarization-noise-sd", type=float, default=0.005,
        help="Post-solution Gaussian voltage perturbation standard deviation.")
    return parser.parse_args()


def main():
    args = _parse_arguments()
    parameter_names = (
        [item.strip() for item in args.parameters.split(",")]
        if args.parameters else None)
    base_overrides = None
    sim_hours = args.sim_hours
    if args.smoke_test:
        sim_hours = min(float(sim_hours), 48.0)
        base_overrides = {
            "N_grid": 48,
            "max_agents": 1_000_000,
            "max_pseudo_iter": 500,
            "diffusion_tol": 5e-4,
            "diffusion_check_interval": 100,
        }
        if parameter_names is None:
            parameter_names = ["R_contact", "mu_max_sp1"]

    if args.polarization:
        run_polarization_analysis(
            seed=args.seed,
            sim_hours=sim_hours,
            output_dir=args.output_dir,
            voltage_noise_sd=args.polarization_noise_sd,
            **(base_overrides or {}),
        )
        return 0

    running_on_kaggle = Path("/kaggle/working").is_dir()
    selected_case_id = args.case_id
    if selected_case_id is None and running_on_kaggle and not args.all_cases:
        selected_case_id = KAGGLE_OAT_CASE_ID
    if selected_case_id is not None:
        run_single_oat_case_job(
            case_id=selected_case_id,
            sim_hours=sim_hours,
            seed=args.seed,
            variation=args.variation,
            output_dir=args.output_dir,
            base_config_overrides=base_overrides,
        )
        return 0

    result = run_oat_sensitivity_analysis(
        sim_hours=sim_hours,
        seed=args.seed,
        variation=args.variation,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        base_config_overrides=base_overrides,
        parameter_names=parameter_names,
        verbose=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

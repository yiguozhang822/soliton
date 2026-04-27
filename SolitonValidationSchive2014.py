"""
Validation: Numerical Soliton Profile vs. Schive et al. 2014
=============================================================
Compares the numerically solved density profile from our SP solver
against the analytic fitting formula from Schive et al. 2014 (arXiv:1407.7762),
which is the canonical soliton profile used throughout the ULDM literature.

Schive+2014 fitting formula (their Eq. 3), normalized so rho(0) = 1:

    rho_norm(x) = [ 1 + 0.091 * (x / x_core)^2 ]^(-8)

where x_core is the half-density core radius.

One can verify this is a half-density definition: at x = x_core,
    rho_norm = [1 + 0.091]^(-8) = [1.091]^(-8) ≈ 0.4982 ≈ 0.5  ✓

Agreement between the numerical and analytic profiles validates:
  1. The SP system is set up correctly
  2. The boundary conditions are correct
  3. The half-density core radius definition is consistent with literature
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp


# ─────────────────────────────────────────────
#  SP solver (isolated soliton)
#  Copied from IsolatedSolitonBaseline.py
# ─────────────────────────────────────────────
X_EPS = 1e-6
X_MAX = 200.0
N_PTS = 2500

def odes(x, y, eps):
    y1, y2, y3, y4 = y
    eps_val = eps[0]
    dy1 = y2
    dy2 = 2.0 * (y3 - eps_val) * y1
    dy3 = y4
    dy4 = y1**2 / x**2 - (2.0 / x) * y4
    return np.vstack([dy1, dy2, dy3, dy4])

def boundary_conditions(ya, yb, eps):
    return np.array([
        ya[0] - X_EPS,
        ya[1] - 1.0,
        ya[3] - 0.0,
        yb[0] - 0.0,
        yb[2] - 0.0,
    ])

def make_initial_guess(x_grid, eps_guess=-0.7):
    scale = 3.0
    s0  = x_grid * np.exp(-x_grid**2 / (2.0 * scale**2))
    s0  = s0 / s0[0] * x_grid[0]
    ds0 = np.gradient(s0, x_grid)
    v0  = -0.3 * np.exp(-x_grid**2 / (2.0 * scale**2))
    dv0 = np.gradient(v0, x_grid)
    return np.vstack([s0, ds0, v0, dv0]), np.array([eps_guess])

def find_core_radius(x, s):
    rho = (s / x)**2
    half_rho = 0.5 * rho[0]
    idx = np.where(rho < half_rho)[0]
    if len(idx) == 0:
        raise RuntimeError("Density never drops to half — increase X_MAX")
    i = idx[0]
    return x[i-1] + (half_rho - rho[i-1]) / (rho[i] - rho[i-1]) * (x[i] - x[i-1])

def solve_isolated_soliton():
    x_grid = np.geomspace(X_EPS, X_MAX, N_PTS)
    y_guess, eps_arr = make_initial_guess(x_grid)
    sol = solve_bvp(odes, boundary_conditions, x_grid, y_guess,
                    p=eps_arr, tol=1e-6, max_nodes=50000, verbose=0)
    if not sol.success:
        raise RuntimeError(f"BVP solver failed: {sol.message}")
    return sol.x, sol.y[0], sol.y[2], sol.p[0]


# ─────────────────────────────────────────────
#  Schive+2014 analytic fitting formula
# ─────────────────────────────────────────────
# This is the standard soliton profile from the ULDM literature.
# It was derived by Schive et al. by fitting to cosmological simulations.
# The formula is normalized so that rho(0) = 1 and rho(x_core) ≈ 0.5.
def schive_profile(x, x_core):
    return (1.0 + 0.091 * (x / x_core)**2)**(-8)


# ─────────────────────────────────────────────
#  Run solver and compute profiles
# ─────────────────────────────────────────────
print("Solving SP system...")
x, s, v, eps = solve_isolated_soliton()
x_core = find_core_radius(x, s)

# Numerical density profile, normalized so rho(0) = 1
rho_numerical = (s / x)**2
rho_numerical_norm = rho_numerical / rho_numerical[0]
# rho_numerical[0] is just 1.0. It's forced by our normalization choice s'(0) = 1.
# Since rho ~ (s/x)^2 and s/x -> s'(0) = 1 as x -> 0,
# the central density is always 1 by construction.

# Schive+2014 analytic profile on the same x grid
rho_schive = schive_profile(x, x_core)

# Residual between numerical and analytic (only meaningful near the core,
# x in [0, ~5*x_core], since the Schive formula is a fit valid in the core region)
core_mask = x <= 5.0 * x_core
residual = np.abs(rho_numerical_norm[core_mask] - rho_schive[core_mask])
max_residual = np.max(residual)
mean_residual = np.mean(residual)

print(f"  x_core        = {x_core:.8f}  (half-density, solver units)")
print(f"  eps           = {eps:.8f}")
print(f"  Max |residual| within 5*x_core  = {max_residual:.2e}")
print(f"  Mean|residual| within 5*x_core  = {mean_residual:.2e}")


# ─────────────────────────────────────────────
#  Plot: profile comparison + residual
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    "Our Numerical SP Solution vs. Schive et al. 2014 Shape",
    fontsize=12
)

# --- Panel 1: Profile comparison ---
ax = axes[0]

# Plot on a normalized x-axis: xi = x / x_core, so the core is always at xi = 1
xi = x / x_core
xi_schive = np.linspace(0.01, 6, 500)
rho_schive_plot = schive_profile(xi_schive * x_core, x_core)

ax.plot(xi_schive, rho_schive_plot, "C1", lw=6, label="Schive et al. 2014 fit", zorder=2)
ax.plot(xi, rho_numerical_norm, "C0--", lw=2, label="Numerical (this work)", zorder=3)
ax.axhline(0.5, color='gray', ls=':', lw=1, label='Half-density threshold')
ax.axvline(1.0, color='gray', ls=':', lw=1)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r"$x \, / \, x_{\rm core}$", fontsize=12)
ax.set_ylabel(r"$\rho \, / \, \rho_0$", fontsize=12)
ax.set_title("Normalized density profile")
ax.set_xlim([0.1, 5])
ax.set_ylim([1e-3, 1.5])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

# Annotate key values
ax.annotate(
    f"$\\varepsilon$ = {eps:.5f}\n$x_{{\\rm core}}$ = {x_core:.5f}",
    xy=(0.97, 0.97), xycoords='axes fraction',
    ha='right', va='top', fontsize=9,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

# --- Panel 2: Absolute residual ---
ax = axes[1]

xi_core = x[core_mask] / x_core
ax.loglog(xi_core, residual, 'C2', lw=1.5)
ax.set_xlabel(r"$x \, / \, x_{\rm core}$", fontsize=12)
ax.set_ylabel(r"$|\rho_{\rm numerical} - \rho_{\rm Schive}|$", fontsize=12)
ax.set_title("Absolute residual (numerical − analytic)")
ax.set_xlim([0.1, 5])
ax.grid(True, alpha=0.3, which='both')

ax.annotate(
    f"Max residual: {max_residual:.2e}\nMean residual: {mean_residual:.2e}",
    xy=(0.97, 0.97), xycoords='axes fraction',
    ha='right', va='top', fontsize=9,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

plt.tight_layout()
plt.savefig("validation_schive2014.png", dpi=150, bbox_inches='tight')
print("\n  Plot saved to: validation_schive2014.png")
plt.show()
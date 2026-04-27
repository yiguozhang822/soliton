"""
Isolated Soliton Baseline Solver7
==================================
Solves the dimensionless Schrödinger-Poisson (SP) system for the
isolated soliton (no external potential), using scipy.solve_bvp.
 
Dimensionless SP system:
    s''(x) = 2*(v(x) - eps)*s(x)          [Schrodinger], note, without the v_ext because this is the isolated case
    v''(x) + (2/x)*v'(x) = s(x)^2/x^2    [Poisson]
 
where x is the dimensionless radius, s is the wavefunction variable,
v is the self-gravitational potential, eps is the energy eigenvalue.
 
from the SP system we can get these 4 first-order ODEs
 
First-order system  y = [s, s', v, v']:s
    y1' = y2
    y2' = 2*(y3 - eps)*y1
    y3' = y4
    y4' = y1^2/x^2 - (2/x)*y4
 
these boundary conditions can be found in section 3, nondimensionalization:
    Inner (x = X_EPS):  y1 = X_EPS,  y2 = 1,  y4 = 0
    Outer (x = X_MAX):  y1 = 0,       y3 = 0
 
Core radius definition:
    rho(r_c) = rho(0) / 2   [half-density]
    rho ∝ s(x)^2 / x^2
 
Key outputs (raw solver units, no rescaling):
    eps     -- eigenvalue
    x_core  -- half-density core radius in solver units
 
When computing correction maps for deformed solitons, use:
    f_r   = x_core_deformed / x_core_raw   (core radius ratio)
    f_rho = rho0_deformed   / rho0_raw     (central density ratio)
"""
 
import numpy as np
from scipy.integrate import solve_bvp
 
 
# ─────────────────────────────────────────────
#  Grid parameters
# ─────────────────────────────────────────────
# The solver integrates from X_EPS to X_MAX on a geometric (log-spaced) grid.
# X_EPS is a small but nonzero inner boundary to avoid the 1/x singularity at x=0.
# X_MAX must be large enough that s and v have decayed to zero well before the edge.
# Geomspace (instead of linspace) puts more points near x=0 where the solution
# changes rapidly, which is critical for accuracy near the soliton core.
X_EPS = 1e-6
X_MAX = 200.0
N_PTS = 2500
 
 
# ─────────────────────────────────────────────
#  ODE right-hand side
# ─────────────────────────────────────────────
# scipy.solve_bvp expects a function f(x, y, p) that returns dy/dx.
# We rewrite the two coupled 2nd-order ODEs as four 1st-order ODEs by
# introducing y = [s, s', v, v'], so the solver only deals with first derivatives.
# eps is treated as an unknown parameter p[0] that the solver finds simultaneously
# with the solution — this is the eigenvalue problem aspect of the SP system.
def odes(x, y, eps):
    y1, y2, y3, y4 = y
    eps_val = eps[0]
 
    dy1 = y2                                    # s' = s'
    dy2 = 2.0 * (y3 - eps_val) * y1            # s'' = 2*(v - eps)*s   [Schrodinger]
    dy3 = y4                                    # v' = v'
    dy4 = y1**2 / x**2 - (2.0 / x) * y4       # v'' = s^2/x^2 - (2/x)*v'  [Poisson]
 
    return np.vstack([dy1, dy2, dy3, dy4])
 
 
# ─────────────────────────────────────────────
#  Boundary conditions
# ─────────────────────────────────────────────
# The BVP has 4 ODEs + 1 unknown parameter (eps) = 5 unknowns, so we need 5 BCs.
# ya = y evaluated at the inner boundary x = X_EPS
# yb = y evaluated at the outer boundary x = X_MAX
#
# Inner BCs come from physical regularity at the origin (section 3):
#   s(X_EPS) = X_EPS  : since s = r*psi and psi must be finite at r=0,
#                        s must vanish linearly, so s/x -> constant. We set that
#                        constant to 1 by imposing s'(0)=1 (normalization choice).
#   s'(X_EPS) = 1     : normalization — sets the overall amplitude of the solution.
#   v'(X_EPS) = 0     : gravitational field must be zero at the center by symmetry
#                        (no preferred direction at r=0).
#
# Outer BCs come from the soliton being a localized, bound object:
#   s(X_MAX) = 0      : wavefunction decays to zero far from the core.
#   v(X_MAX) = 0      : gauge choice — gravitational potential goes to zero at infinity.
def boundary_conditions(ya, yb, eps):
    return np.array([
        ya[0] - X_EPS,   # s(X_EPS) = X_EPS   [regularity at origin]
        ya[1] - 1.0,     # s'(X_EPS) = 1       [normalization]
        ya[3] - 0.0,     # v'(X_EPS) = 0       [no field at center]
        yb[0] - 0.0,     # s(X_MAX)  = 0       [localization]
        yb[2] - 0.0,     # v(X_MAX)  = 0       [gauge: v -> 0 at infinity]
    ])
 
 
# ─────────────────────────────────────────────
#  Initial guess
# ─────────────────────────────────────────────
# solve_bvp is an iterative solver — it needs a starting guess for s(x), v(x), and eps.
# The guess doesn't need to be accurate, but it must satisfy the rough shape:
#   s: starts at 0, peaks near the core, then decays to 0 at large x.
#      We use s ~ x * exp(-x^2 / 2*scale^2), which has exactly this shape.
#      The prefactor x ensures s(0) = 0 naturally.
#      We then normalize so that s(X_EPS) = X_EPS to match the inner BC exactly.
#   v: negative (gravitational well) near the center, decaying to 0 far out.
#      We use v ~ -0.3 * exp(-x^2 / 2*scale^2).
#   eps: a rough guess of -0.7, close to the known eigenvalue ~ -0.68.
# scale=3 makes the guess broad enough to avoid the solver getting stuck.

'this is very important!!! the initial guess, which i set was -0.7(this was determined by trial and error, since my final result was )'
def make_initial_guess(x_grid, eps_guess=-0.7):
    scale = 3.0
    s0  = x_grid * np.exp(-x_grid**2 / (2.0 * scale**2))
    s0  = s0 / s0[0] * x_grid[0]       # enforce s(X_EPS) = X_EPS exactly
    ds0 = np.gradient(s0, x_grid)      # numerical derivative for s'
    v0  = -0.3 * np.exp(-x_grid**2 / (2.0 * scale**2))
    dv0 = np.gradient(v0, x_grid)      # numerical derivative for v'
    y_guess = np.vstack([s0, ds0, v0, dv0])
    return y_guess, np.array([eps_guess])
 
 
# ─────────────────────────────────────────────
#  Half-density core radius
# ─────────────────────────────────────────────
# The core radius is defined as the point where the density drops to half its
# central value: rho(x_core) = rho(0) / 2.
# The density profile is rho ∝ (s/x)^2.
# With our normalization s'(0) = 1, we have s/x -> 1 as x -> 0,
# so rho(0) ∝ 1^2 = 1, and we look for where (s/x)^2 = 0.5.
# We find the first grid point where rho dips below the threshold,
# then linearly interpolate between that point and the one before it
# for a more precise crossing location.
def find_core_radius(x, s):
    rho = (s / x)**2
    half_rho = 0.5 * rho[0]
 
    idx = np.where(rho < half_rho)[0]
    if len(idx) == 0:
        raise RuntimeError("Density never drops to half — increase X_MAX")
 
    i = idx[0]
    # linear interpolation between the last point above and first point below threshold
    x_core = x[i-1] + (half_rho - rho[i-1]) / (rho[i] - rho[i-1]) * (x[i] - x[i-1])
    return x_core
 
 
# ─────────────────────────────────────────────
#  Main solver
# ─────────────────────────────────────────────
# Assembles the grid, guess, and calls scipy.solve_bvp.
# solve_bvp iteratively refines the solution and adds mesh nodes where needed
# until the residual drops below tol=1e-6 everywhere.
# After solving, we extract the three baseline numbers needed for the correction maps:
#   eps    -- the eigenvalue (reference for deformed cases)
#   x_core -- the half-density core radius (denominator of f_r)
#   rho0   -- the central density (denominator of f_rho)
def solve_isolated_soliton(verbose=True):
    x_grid = np.geomspace(X_EPS, X_MAX, N_PTS)
    y_guess, eps_arr = make_initial_guess(x_grid)
 
    if verbose:
        print("Solving isolated soliton BVP...")
 
    sol = solve_bvp(
        odes,
        boundary_conditions,
        x_grid,
        y_guess,
        p=eps_arr,
        tol=1e-6,
        max_nodes=50000,
        verbose=0,
    )
 
    if not sol.success:
        raise RuntimeError(f"BVP solver failed: {sol.message}")
 
    x   = sol.x
    s   = sol.y[0]
    v   = sol.y[2]
    eps = sol.p[0]
 
    x_core = find_core_radius(x, s)
    rho0   = (s[0] / x[0])**2      # central density: rho ∝ (s/x)^2 evaluated at x -> 0
 
    if verbose:
        print()
        print("=" * 50)
        print("  ISOLATED SOLITON BASELINE")
        print("=" * 50)
        print(f"  eps      = {eps:.15f}")
        print(f"  x_core   = {x_core:.15f}  (half-density)")
        print(f"  rho0     = {rho0:.6f}       (central density)")
        print(f"  v(0)     = {v[0]:.6f}       (gravitational well depth)")
        print("=" * 50)
        print()
        print("  For deformed cases, compute:")
        print("    f_r   = x_core_deformed / x_core")
        print("    f_rho = rho0_deformed   / rho0")
 
    return {
        "x":      x,
        "s":      s,
        "v":      v,
        "eps":    eps,
        "x_core": x_core,
        "rho0":   rho0,
    }
 
 
# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    res = solve_isolated_soliton(verbose=True)
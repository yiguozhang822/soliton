"""
step6_generate_library.py
=========================
Generates step6_library.npz by solving the soliton BVP over a 3-D
parameter grid (Xi, fb, eta).

Physics:
  s'' = 2*(v + v_ext - eps)*s
  v'' = s^2/x^2 - (2/x)*v'

  v_ext = -Xi/x  -  fb*(1+eta^2)^1.5 / sqrt(x^2+eta^2)

Boundary conditions (5 conditions + 1 eigenvalue eps):
  s(x_min) = X_EPS,  s'(x_min) = 1,  v'(x_min) = 0
  s(x_max) = 0,      v(x_max)  = 0

Outputs saved to: ~/outputs/step6_library.npz
"""

import os
import numpy as np
from scipy.integrate import solve_bvp, cumulative_trapezoid
from scipy.interpolate import interp1d
from numpy import trapezoid as trapz
import warnings
warnings.filterwarnings('ignore')

# ── Output directory ───────────────────────────────────────────────
OUT_DIR = os.path.expanduser('~/outputs')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, 'step6_library.npz')

# ── Grid parameters  (edit these to match your desired scan) ───────
Xi_vals  = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00])
fb_vals  = np.array([0.00, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00])
eta_vals = np.array([0.50, 1.00, 2.00, 5.00])

# ── Solver settings ────────────────────────────────────────────────
X_EPS = 1e-4
X_MAX = 40.0
N     = 4000
x_grid = np.linspace(X_EPS, X_MAX, N)

# ── Initial guess (Thomas-Fermi-like soliton profile) ─────────────
def build_guess(xg):
    a = 2.0**(0.25) - 1.0
    s = xg * (1 + a * xg**2)**(-4)
    M = np.zeros_like(xg)
    M[1:] = cumulative_trapezoid(s**2, xg)
    vp = M / xg**2;  vp[0] = 0.0
    v  = -(trapz(vp, xg) - cumulative_trapezoid(vp, xg, initial=0.0))
    y0 = np.zeros((4, xg.size))
    y0[0] = s
    y0[1] = np.gradient(s, xg)
    y0[2] = v
    y0[3] = vp
    return y0

# ── Boundary conditions ────────────────────────────────────────────
def _bc(ya, yb, p):
    return np.array([ya[0] - X_EPS,   # s(x_min) = X_EPS
                     ya[1] - 1.0,     # s'(x_min) = 1  (normalization)
                     ya[3],           # v'(x_min) = 0
                     yb[0],           # s(x_max)  = 0
                     yb[2]])          # v(x_max)  = 0

# ── Interpolate BVP solution onto fixed x_grid ────────────────────
def to_fixed(sx, sy):
    y = np.zeros((4, N))
    for i in range(4):
        f = interp1d(sx, sy[i], bounds_error=False,
                     fill_value=(sy[i][0], 0.0))
        y[i] = f(x_grid)
    y[0] = np.clip(y[0], 0.0, None)
    return y

# ── External potential ─────────────────────────────────────────────
def v_ext(x, Xi, fb, eta):
    out = np.zeros_like(x)
    if Xi != 0:
        out += -Xi / x
    if fb != 0:
        out += -fb * (1 + eta**2)**1.5 / np.sqrt(x**2 + eta**2)
    return out

# ── Helper: core radius (quarter-density crossing of s/x) ─────────
def core_radius(x, s):
    ratio = s / x
    idx = np.where(ratio <= 0.5)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return np.nan
    return x[i-1] + (0.5 - ratio[i-1]) * (x[i] - x[i-1]) / (ratio[i] - ratio[i-1])

# ══════════════════════════════════════════════════════════════════
# Step 1: Isolated soliton (Xi=fb=0)
# ══════════════════════════════════════════════════════════════════
print("="*60)
print("Solving isolated soliton baseline ...")

sol_iso = solve_bvp(
    lambda xg, yg, p: [yg[1],
                        2*(yg[2] - p[0])*yg[0],
                        yg[3],
                        yg[0]**2/xg**2 - (2/xg)*yg[3]],
    _bc, x_grid, build_guess(x_grid),
    p=[-0.682], tol=1e-8, max_nodes=80000, verbose=0)

if not sol_iso.success:
    raise RuntimeError("Isolated soliton solve failed — check initial guess.")

x_iso = sol_iso.x
s_iso = sol_iso.y[0]
eps0  = sol_iso.p[0]

xc0 = core_radius(x_iso, s_iso)
mu0 = trapz(s_iso**2, x_iso)

print(f"  eps0 = {eps0:.6f}")
print(f"  xc0  = {xc0:.6f}  (isolated core radius)")
print(f"  mu0  = {mu0:.6f}  (isolated mass integral)")

# Fixed-grid version of isolated solution (used as initial guess)
y_iso_f = to_fixed(sol_iso.x, sol_iso.y)

# ══════════════════════════════════════════════════════════════════
# Step 2: Sweep over (Xi, fb, eta)
# ══════════════════════════════════════════════════════════════════
nXi, nfb, neta = len(Xi_vals), len(fb_vals), len(eta_vals)
fr_grid   = np.full((nXi, nfb, neta), np.nan)
frho_grid = np.full((nXi, nfb, neta), np.nan)
ok_grid   = np.zeros((nXi, nfb, neta), dtype=bool)

total = nXi * nfb * neta
done  = 0

print(f"\nSweeping {total} parameter combinations ...\n")
print(f"  {'Ξ':>6} {'fb':>6} {'η':>5}  {'status':>12} {'f_r':>8} {'f_rho':>8}")
print("  " + "-"*52)

for iX, Xi in enumerate(Xi_vals):
    # Warm-start: carry solution from previous Xi step
    y_warm = y_iso_f.copy()

    for if_, fb in enumerate(fb_vals):
        for ie, eta in enumerate(eta_vals):
            done += 1

            def rhs(xg, yg, p, _Xi=Xi, _fb=fb, _eta=eta):
                return np.array([
                    yg[1],
                    2*(yg[2] + v_ext(xg, _Xi, _fb, _eta) - p[0]) * yg[0],
                    yg[3],
                    yg[0]**2/xg**2 - (2/xg)*yg[3]
                ])

            sol = solve_bvp(rhs, _bc, x_grid.copy(), y_warm.copy(),
                            p=[eps0], tol=1e-8, max_nodes=80000, verbose=0)

            if not sol.success:
                print(f"  {Xi:>6.3f} {fb:>6.3f} {eta:>5.2f}  {'FAILED':>12}")
                continue

            x = sol.x;  s = sol.y[0]
            xc  = core_radius(x, s)
            mu  = trapz(s**2, x)

            fr   = xc / xc0
            frho = mu0 / mu

            fr_grid[iX, if_, ie]   = fr
            frho_grid[iX, if_, ie] = frho
            ok_grid[iX, if_, ie]   = True

            print(f"  {Xi:>6.3f} {fb:>6.3f} {eta:>5.2f}  {'ok':>12} {fr:>8.4f} {frho:>8.4f}")

            # Update warm start for next iteration
            y_warm = to_fixed(sol.x, sol.y)

print(f"\nConverged: {int(ok_grid.sum())} / {total}")

# ══════════════════════════════════════════════════════════════════
# Step 3: Save
# ══════════════════════════════════════════════════════════════════
np.savez(OUT_FILE,
         Xi_vals=Xi_vals,
         fb_vals=fb_vals,
         eta_vals=eta_vals,
         fr_grid=fr_grid,
         frho_grid=frho_grid,
         ok_grid=ok_grid,
         xc0=xc0,
         mu0=mu0)

print(f"\n✓ Library saved → {OUT_FILE}")

"""
STEP 5: Schrödinger-Poisson BVP with External Potentials
=========================================================
The only change from Step 4: one extra term in the Schrödinger equation.

  STEP 4 (isolated):  s'' = 2*(v - eps)*s
  STEP 5 (perturbed): s'' = 2*(v + v_ext - eps)*s   ← v_ext added

Descriptor set: Π = (Ξ, f_b,core, η)
  Ξ       = M_bullet / M_sol,0        (SMBH point mass strength)
  fb_core = M_b(<r_c0) / M_sol,0      (baryon mass inside isolated core)
  η       = a / r_c0                  (Plummer scale / isolated core radius)

External potential:
  v_ext(x) = -Ξ/x  -  fb*(1+η²)^1.5 / √(x² + η²)

Observables:
  f_r   = r_c / r_c0  = x_c_perturbed / x_c_isolated
  f_rho = ρ_c / ρ_c0  = μ_0 / μ_perturbed  (at fixed soliton mass)
"""

import numpy as np
from scipy.integrate import solve_bvp, cumulative_trapezoid
from scipy.interpolate import interp1d
from numpy import trapezoid as trapz
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# GRID SETTINGS — linspace avoids node proliferation
# ══════════════════════════════════════════════════════════════
X_EPS = 1e-4     # inner boundary (small but not too small for linspace)
X_MAX = 40.0     # outer boundary (soliton decays well before this)
N     = 4000     # grid points

def build_initial_guess(x_grid):
    """
    Self-consistent initial guess from Schive+2014 profile + Poisson integration.
    Uses a = 2^(1/4) - 1 so that s(x)/x = 0.5 exactly at x = x_core_raw.
    """
    a = 2.0**(0.25) - 1.0
    s = x_grid * (1.0 + a * x_grid**2)**(-4)
    M = np.zeros_like(x_grid)
    M[1:] = cumulative_trapezoid(s**2, x_grid)
    vp = M / x_grid**2; vp[0] = 0.0
    v  = -(trapz(vp, x_grid) - cumulative_trapezoid(vp, x_grid, initial=0.0))
    y0 = np.zeros((4, x_grid.size))
    y0[0] = s; y0[1] = np.gradient(s, x_grid)
    y0[2] = v; y0[3] = vp
    return y0

def get_core_radius(x, s):
    """Find x_c where s(x)/x = 0.5 (quarter-density, normalization-independent)."""
    r = s / x
    idx = np.where(r <= 0.5)[0]
    if not len(idx): raise ValueError("Profile never reaches quarter-density")
    i = idx[0]
    return x[i-1] + (0.5 - r[i-1]) * (x[i] - x[i-1]) / (r[i] - r[i-1])

# ══════════════════════════════════════════════════════════════
# ISOLATED BASELINE SOLVE
# ══════════════════════════════════════════════════════════════
x_grid = np.linspace(X_EPS, X_MAX, N)

def _rhs_iso(xg, yg, p):
    eps = p[0]; s, sp, v, vp = yg
    return np.array([sp, 2*(v - eps)*s, vp, s**2/xg**2 - (2/xg)*vp])

def _bc(ya, yb, p):
    return np.array([ya[0]-X_EPS, ya[1]-1.0, ya[3], yb[0], yb[2]])

sol_iso = solve_bvp(_rhs_iso, _bc, x_grid, build_initial_guess(x_grid),
                    p=[-0.682], tol=1e-8, max_nodes=80000, verbose=0)
assert sol_iso.success, f"Isolated solve failed: {sol_iso.message}"

x0    = sol_iso.x
s0    = sol_iso.y[0]
v0    = sol_iso.y[2]
eps0  = sol_iso.p[0]
xc0   = get_core_radius(x0, s0)     # isolated core radius (raw units)
mu0   = trapz(s0**2, x0)            # normalization integral
y_iso = sol_iso.y                   # warm-start array for perturbed cases

# ══════════════════════════════════════════════════════════════
# EXTERNAL POTENTIAL
# ══════════════════════════════════════════════════════════════
def v_ext(x, Xi, fb, eta):
    """
    Dimensionless external potential from SMBH + Plummer baryons.
    Plummer prefactor: (1+η²)^1.5 converts fb_core → total M_b/M_sol,0.
    """
    out = np.zeros_like(x)
    if Xi  != 0.0: out += -Xi / x
    if fb  != 0.0: out += -fb * (1.0 + eta**2)**1.5 / np.sqrt(x**2 + eta**2)
    return out

# ══════════════════════════════════════════════════════════════
# PERTURBED SOLVER
# ══════════════════════════════════════════════════════════════
def solve_soliton(Xi, fb, eta, y_warm=None):
    """
    Solve perturbed SP BVP. Returns (f_r, f_rho, eps, y_array).
    Use y_warm = y_iso for the first call; can chain solves for large perturbations.
    """
    if y_warm is not None:
        y0 = np.zeros((4, N))
        for i in range(4):
            f = interp1d(x0, y_warm[i], bounds_error=False,
                         fill_value=(y_warm[i][0], 0.0))
            y0[i] = f(x_grid)
        y0[0] = np.clip(y0[0], 0.0, None)
    else:
        y0 = build_initial_guess(x_grid)

    def rhs_p(xg, yg, p):
        eps = p[0]; s, sp, v, vp = yg
        vx = v_ext(xg, Xi, fb, eta)
        return np.array([sp, 2*(v + vx - eps)*s, vp, s**2/xg**2 - (2/xg)*vp])

    sol = solve_bvp(rhs_p, _bc, x_grid.copy(), y0, p=[eps0],
                    tol=1e-8, max_nodes=80000, verbose=0)
    if not sol.success:
        raise RuntimeError(f"Failed Xi={Xi}, fb={fb}, eta={eta}: {sol.message}")

    xc   = get_core_radius(sol.x, sol.y[0])
    mu   = trapz(sol.y[0]**2, sol.x)
    f_r  = xc / xc0           # core radius ratio
    f_rho = mu0 / mu           # central density ratio at fixed soliton mass
    return f_r, f_rho, sol.p[0], sol.y


# ══════════════════════════════════════════════════════════════
# MAIN: PRINT RESULTS
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print(f"Isolated: eps={eps0:.6f}, x_core={xc0:.6f}, mu={mu0:.4f}")
    print(f"          eps_resc = {xc0**2*eps0:.4f}  (expect -2.4242)")

    print("\nVALIDATION TESTS")
    tests = [
        ("Zero perturbation",             0.0,  0.0,  1.0),
        ("Weak (Xi=0.01, fb=0.01)",       0.01, 0.01, 1.0),
        ("SMBH moderate (Xi=0.2)",        0.2,  0.0,  1.0),
        ("SMBH strong   (Xi=1.0)",        1.0,  0.0,  1.0),
        ("Baryons compact (fb=.5,η=.3)",  0.0,  0.5,  0.3),
        ("Baryons diffuse (fb=.5,η=3)",   0.0,  0.5,  3.0),
        ("Combined (Xi=.3,fb=.3,η=.5)",   0.3,  0.3,  0.5),
    ]
    print(f"  {'Label':<37} {'f_r':>7} {'f_rho':>7}")
    print("  " + "-"*55)
    for label, Xi, fb, eta in tests:
        try:
            fr, frho, eps, _ = solve_soliton(Xi, fb, eta, y_warm=y_iso)
            print(f"  {label:<37} {fr:>7.4f} {frho:>7.4f}")
        except Exception as e:
            print(f"  {label:<37} FAILED: {e}")
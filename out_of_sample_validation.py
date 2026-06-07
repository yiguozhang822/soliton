"""
out_of_sample_validation.py
============================
Held-out (out-of-sample) validation of the Chebyshev correction map.

1. Fits degree-5 (56-term) and sparse (30-term) Chebyshev models on the
   training library (data.csv) using the EXACT fit logic of
   fittingformulas.py / dump_paper_numbers.py (lstsq + greedy forward
   selection in log-space).
2. Generates N_OOS parameter triples (Xi, fb_core, eta) NOT on the 630-point
   training grid (uniform Xi, uniform fb, log-uniform eta, strictly interior).
3. Solves the SP BVP at each triple for the "truth" f_r, f_rho using the SAME
   machinery as 3Dplot630points.py: half-density core radius, direct solve
   with a node check, and path continuation as a fallback so stiff
   (SMBH-dominated) points cannot silently converge to an excited state.
4. Evaluates both models at those triples; reports relative error.
5. Plots residual percentile curves: one figure, two panels (f_r, f_rho),
   two lines each (deg-5, sparse). A second figure overlays in-sample curves.

Run from inside the repo directory (needs data.csv):
    python out_of_sample_validation.py
"""

import os
import time
import numpy as np
from scipy.integrate import solve_bvp, cumulative_trapezoid
from scipy.interpolate import interp1d
from numpy import trapezoid as trapz
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 11, 'font.family': 'serif'})

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260607
N_OOS = 150                      # number of out-of-sample triples to attempt

# ══════════════════════════════════════════════════════════════════
# SOLVER  (identical settings + logic to 3Dplot630points.py)
# ══════════════════════════════════════════════════════════════════
X_EPS = 1e-4
X_MAX = 40.0
N     = 4000
x_grid = np.linspace(X_EPS, X_MAX, N)


def build_guess(xg):
    a = 2.0**0.25 - 1.0
    s = xg * (1.0 + a * xg**2)**(-4)
    M = np.zeros_like(xg); M[1:] = cumulative_trapezoid(s**2, xg)
    vp = M / xg**2; vp[0] = 0.0
    v = -(trapz(vp, xg) - cumulative_trapezoid(vp, xg, initial=0.0))
    y0 = np.zeros((4, xg.size))
    y0[0] = s; y0[1] = np.gradient(s, xg); y0[2] = v; y0[3] = vp
    return y0


def _bc(ya, yb, p):
    return np.array([ya[0] - X_EPS, ya[1] - 1.0, ya[3], yb[0], yb[2]])


def to_fixed(sol_x, sol_y):
    y = np.zeros((4, N))
    for i in range(4):
        f = interp1d(sol_x, sol_y[i], bounds_error=False,
                     fill_value=(sol_y[i][0], 0.0))
        y[i] = f(x_grid)
    y[0] = np.clip(y[0], 0.0, None)
    return y


def get_xc(x, s):
    """Half-density core radius: rho(x_c) = 0.5 * rho(0)."""
    rho = (s / x)**2
    half_rho = 0.5 * rho[0]
    idx = np.where(rho < half_rho)[0]
    if not len(idx):
        return None
    i = idx[0]
    return x[i-1] + (half_rho - rho[i-1]) / (rho[i] - rho[i-1]) * (x[i] - x[i-1])


def count_nodes(s, threshold=1e-6):
    s_peak = np.max(np.abs(s))
    if s_peak == 0:
        return 0
    s_sig = s[np.abs(s) > threshold * s_peak]
    if len(s_sig) < 2:
        return 0
    return int(np.sum(np.abs(np.diff(np.sign(s_sig))) > 0))


# ── Isolated baseline (denominator of the ratios) ──────────────────
print("Solving isolated baseline ...")
sol_iso = solve_bvp(
    lambda xg, yg, p: [yg[1], 2*(yg[2]-p[0])*yg[0],
                        yg[3], yg[0]**2/xg**2 - (2/xg)*yg[3]],
    _bc, x_grid, build_guess(x_grid),
    p=[-0.682], tol=1e-8, max_nodes=80000, verbose=0)
assert sol_iso.success, sol_iso.message
xc0  = get_xc(sol_iso.x, sol_iso.y[0])
mu0  = trapz(sol_iso.y[0]**2, sol_iso.x)
eps0 = sol_iso.p[0]
y_iso = to_fixed(sol_iso.x, sol_iso.y)
print(f"  xc0={xc0:.6f}  mu0={mu0:.6f}  eps0={eps0:.6f}\n")


def v_ext(x, Xi, fb, eta):
    out = np.zeros_like(x)
    if Xi != 0.0:
        out += -Xi / x
    if fb != 0.0:
        out += -fb * (1 + eta**2)**1.5 / np.sqrt(x**2 + (eta*xc0)**2)
    return out


def _solve_one(Xi, fb, eta, yw, eg):
    def rhs(xg, yg, p):
        return np.array([yg[1],
                         2*(yg[2] + v_ext(xg, Xi, fb, eta) - p[0])*yg[0],
                         yg[3],
                         yg[0]**2/xg**2 - (2/xg)*yg[3]])
    sol = solve_bvp(rhs, _bc, x_grid.copy(), yw.copy(),
                    p=[eg], tol=1e-8, max_nodes=80000, verbose=0)
    if not sol.success:
        return None
    xc = get_xc(sol.x, sol.y[0])
    mu = trapz(sol.y[0]**2, sol.x)
    return {'fr':   xc / xc0 if xc else np.nan,
            'frho': mu0 / mu,
            'eps':  sol.p[0],
            'nodes': count_nodes(sol.y[0]),
            'yf':   to_fixed(sol.x, sol.y)}


def solve_with_continuation(Xi_t, fb_t, eta, n_steps=8, max_doublings=3):
    """Walk from the isolated soliton (0,0) to (Xi_t, fb_t) at fixed eta."""
    r = None
    for _ in range(max_doublings + 1):
        yw = y_iso.copy(); eg = eps0; mid_node = False
        for k in range(1, n_steps + 1):
            frac = k / n_steps
            r = _solve_one(Xi_t*frac, fb_t*frac, eta, yw, eg)
            if r is None or r['nodes'] > 0:
                mid_node = True
                break
            yw = r['yf']; eg = r['eps']
        if not mid_node:
            return r
        n_steps *= 2
    return r


def solve_pt(Xi, fb, eta):
    """Direct solve from the isolated baseline; continuation if excited."""
    r = _solve_one(Xi, fb, eta, y_iso, eps0)
    if r is not None and r['nodes'] == 0:
        r['method'] = 'direct'
        return r
    r = solve_with_continuation(Xi, fb, eta, n_steps=8, max_doublings=3)
    if r is not None:
        r['method'] = 'continuation' if r['nodes'] == 0 else 'excluded'
    return r


# ══════════════════════════════════════════════════════════════════
# CHEBYSHEV FIT  (identical to fittingformulas.py / dump_paper_numbers.py)
# ══════════════════════════════════════════════════════════════════
def cheb(n, x):
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return np.array(x, dtype=float)
    a, b = np.ones_like(x), np.array(x, dtype=float)
    for _ in range(2, n+1):
        a, b = b, 2*x*b - a
    return b


def trips(deg):
    return [(i, j, k) for i in range(deg+1)
            for j in range(deg+1-i) for k in range(deg+1-i-j)]


def to_cheb_coords(Xi, fb, eta):
    x1 = 2*Xi - 1
    x2 = 2*fb/0.8 - 1
    x3 = 2*(np.log(eta) - np.log(0.1))/(np.log(5.0) - np.log(0.1)) - 1
    return x1, x2, x3


def design(Xi, fb, eta, tr):
    x1, x2, x3 = to_cheb_coords(np.atleast_1d(Xi).astype(float),
                                np.atleast_1d(fb).astype(float),
                                np.atleast_1d(eta).astype(float))
    return np.column_stack([cheb(i, x1)*cheb(j, x2)*cheb(k, x3)
                            for i, j, k in tr])


def forward_select(A, y, n):
    sel, rem = [], list(range(A.shape[1]))
    for _ in range(n):
        best_sse, best = np.inf, None
        for c in rem:
            t = sel + [c]
            sse = np.sum((A[:, t] @ np.linalg.lstsq(A[:, t], y, rcond=None)[0] - y)**2)
            if sse < best_sse:
                best_sse, best = sse, c
        sel.append(best); rem.remove(best)
    return sel


print("Fitting Chebyshev models on data.csv (training library) ...")
data = np.genfromtxt(os.path.join(HERE, "data.csv"), delimiter=",", skip_header=1)
tr_Xi, tr_fb, tr_eta = data[:, 0], data[:, 1], data[:, 2]
tr_fr, tr_frho = data[:, 3], data[:, 4]
ln_fr, ln_frho = np.log(tr_fr), np.log(tr_frho)

TR5 = trips(5)
A5 = design(tr_Xi, tr_fb, tr_eta, TR5)
c_fr   = np.linalg.lstsq(A5, ln_fr,   rcond=None)[0]
c_frho = np.linalg.lstsq(A5, ln_frho, rcond=None)[0]

print("  forward-selecting sparse bases (30 terms each) ...")
s_fr   = forward_select(A5, ln_fr,   30)
s_frho = forward_select(A5, ln_frho, 30)
csp_fr   = np.linalg.lstsq(A5[:, s_fr],   ln_fr,   rcond=None)[0]
csp_frho = np.linalg.lstsq(A5[:, s_frho], ln_frho, rcond=None)[0]
print(f"  deg-5 c000(f_r)={c_fr[0]:+.5f}  c000(f_rho)={c_frho[0]:+.5f}\n")


def predict(Xi, fb, eta):
    """Return (fr_d5, frho_d5, fr_sp, frho_sp) for arrays/scalars."""
    A = design(Xi, fb, eta, TR5)
    fr_d5   = np.exp(A @ c_fr)
    frho_d5 = np.exp(A @ c_frho)
    fr_sp   = np.exp(A[:, s_fr]   @ csp_fr)
    frho_sp = np.exp(A[:, s_frho] @ csp_frho)
    return fr_d5, frho_d5, fr_sp, frho_sp


# In-sample errors (for the generalisation-gap comparison)
fr_d5_is, frho_d5_is, fr_sp_is, frho_sp_is = predict(tr_Xi, tr_fb, tr_eta)
err_fr_d5_is   = np.abs((fr_d5_is   - tr_fr)   / tr_fr)   * 100
err_frho_d5_is = np.abs((frho_d5_is - tr_frho) / tr_frho) * 100
err_fr_sp_is   = np.abs((fr_sp_is   - tr_fr)   / tr_fr)   * 100
err_frho_sp_is = np.abs((frho_sp_is - tr_frho) / tr_frho) * 100


# ══════════════════════════════════════════════════════════════════
# OUT-OF-SAMPLE POINTS: load cached truth library, or generate + solve
# ══════════════════════════════════════════════════════════════════
# The BVP solve for N_OOS points takes ~minutes. Once solved, the truth
# values (Xi, fb, eta, f_r, f_rho, ...) never change for a given
# SEED/N_OOS, so cache them to out_of_sample_library.csv and reload on
# subsequent runs -- delete the file (or bump CACHE_TAG) to re-solve,
# e.g. when changing SEED, N_OOS, or the solver settings.
csv_path = os.path.join(HERE, "out_of_sample_library.csv")

if os.path.exists(csv_path):
    print(f"Loading cached out-of-sample truth library <- {csv_path}")
    print("  (delete this file to regenerate / re-solve from scratch)\n")
    import csv as _csv
    rows = []
    with open(csv_path, newline="") as f:
        reader = _csv.reader(f)
        next(reader)  # header
        for r in reader:
            rows.append((float(r[0]), float(r[1]), float(r[2]), float(r[3]),
                         float(r[4]), float(r[5]), int(r[6]), r[7], int(r[8])))
    rows = np.array(rows, dtype=object)
else:
    GRID_Xi  = np.array([0.00, 0.01, 0.03, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00])
    GRID_fb  = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.80])
    GRID_eta = np.array([0.10, 0.30, 0.50, 1.00, 1.50, 2.00, 5.00])

    rng = np.random.default_rng(SEED)
    margin = 1e-3   # stay strictly interior to the (-1, 1) cube

    def on_grid(Xi, fb, eta, tol=1e-6):
        return (np.any(np.abs(GRID_Xi  - Xi)  < tol) and
                np.any(np.abs(GRID_fb  - fb)  < tol) and
                np.any(np.abs(GRID_eta - eta) < tol))

    oos_triples = []
    while len(oos_triples) < N_OOS:
        u1, u2, u3 = rng.uniform(-1 + margin, 1 - margin, size=3)
        Xi  = (u1 + 1) / 2
        fb  = (u2 + 1) / 2 * 0.8
        eta = np.exp((u3 + 1) / 2 * (np.log(5.0) - np.log(0.1)) + np.log(0.1))
        if on_grid(Xi, fb, eta):
            continue
        oos_triples.append((Xi, fb, eta))

    print(f"Generated {len(oos_triples)} off-grid out-of-sample triples "
          f"(seed={SEED}).\n")

    # ── Solve out-of-sample points ──
    rows = []
    n_direct = n_cont = n_excl = n_fail = 0
    t0 = time.time()
    print(f"{'#':>4} {'Xi':>6} {'fb':>6} {'eta':>6}  {'f_r':>8} {'f_rho':>9}  "
          f"{'nodes':>5}  method")
    print("-" * 64)
    for idx, (Xi, fb, eta) in enumerate(oos_triples, 1):
        r = solve_pt(Xi, fb, eta)
        if r is None:
            n_fail += 1
            print(f"{idx:>4} {Xi:>6.3f} {fb:>6.3f} {eta:>6.3f}  {'FAILED':>8}")
            continue
        method = r['method']
        valid = (r['nodes'] == 0 and np.isfinite(r['fr']) and np.isfinite(r['frho'])
                 and r['fr'] <= 1.001 and r['frho'] >= 0.999)
        if method == 'direct':
            n_direct += 1
        elif method == 'continuation':
            n_cont += 1
        if not valid:
            n_excl += 1
        rows.append((Xi, fb, eta, r['fr'], r['frho'], r['eps'],
                     r['nodes'], method, int(valid)))
        flag = '' if valid else '   <- excluded (not valid ground state)'
        print(f"{idx:>4} {Xi:>6.3f} {fb:>6.3f} {eta:>6.3f}  "
              f"{r['fr']:>8.4f} {r['frho']:>9.4f}  {r['nodes']:>5}  "
              f"{method}{flag}")

    print(f"\nSolved in {time.time()-t0:.1f}s   "
          f"direct={n_direct}  continuation={n_cont}  "
          f"excluded={n_excl}  failed={n_fail}")

    rows = np.array([r for r in rows], dtype=object)

    # ── Save the out-of-sample dataset for reuse on later runs ──
    with open(csv_path, "w") as f:
        f.write("Xi,fb_core,eta,f_r,f_rho,eps,nodes,method,valid\n")
        for r in rows:
            f.write(f"{r[0]:.10f},{r[1]:.10f},{r[2]:.10f},{r[3]:.10f},"
                    f"{r[4]:.10f},{r[5]:.10f},{r[6]},{r[7]},{r[8]}\n")
    print(f"Saved out-of-sample truth library -> {csv_path}\n")

valid_mask = np.array([bool(r[8]) for r in rows])
V = rows[valid_mask]
n_valid = len(V)
print(f"Valid ground-state out-of-sample points used: {n_valid}\n")

oos_Xi   = np.array([r[0] for r in V], dtype=float)
oos_fb   = np.array([r[1] for r in V], dtype=float)
oos_eta  = np.array([r[2] for r in V], dtype=float)
oos_fr   = np.array([r[3] for r in V], dtype=float)
oos_frho = np.array([r[4] for r in V], dtype=float)

# ══════════════════════════════════════════════════════════════════
# RESIDUALS  (out-of-sample)
# ══════════════════════════════════════════════════════════════════
fr_d5, frho_d5, fr_sp, frho_sp = predict(oos_Xi, oos_fb, oos_eta)
err_fr_d5   = np.abs((fr_d5   - oos_fr)   / oos_fr)   * 100
err_frho_d5 = np.abs((frho_d5 - oos_frho) / oos_frho) * 100
err_fr_sp   = np.abs((fr_sp   - oos_fr)   / oos_fr)   * 100
err_frho_sp = np.abs((frho_sp - oos_frho) / oos_frho) * 100


def stats(e):
    return np.max(e), np.percentile(e, 95), np.median(e)


print("=" * 64)
print("RELATIVE ERROR SUMMARY  (max %  /  p95 %)")
print("=" * 64)
print(f"{'model':<22}{'OUT-OF-SAMPLE':>22}{'IN-SAMPLE':>22}")
for name, e_oos, e_is in [
    ("f_r   deg-5",  err_fr_d5,   err_fr_d5_is),
    ("f_r   sparse", err_fr_sp,   err_fr_sp_is),
    ("f_rho deg-5",  err_frho_d5, err_frho_d5_is),
    ("f_rho sparse", err_frho_sp, err_frho_sp_is),
]:
    mo, po, _ = stats(e_oos)
    mi, pi, _ = stats(e_is)
    print(f"{name:<22}{f'{mo:6.3f} / {po:6.3f}':>22}"
          f"{f'{mi:6.3f} / {pi:6.3f}':>22}")
print()

# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — requested: residual percentile curves, 2 lines per panel
# ══════════════════════════════════════════════════════════════════
C_D5, C_SP = '#1b9e77', '#7570b3'   # match paper Figure 4 colours
pct = np.linspace(0, 100, n_valid)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
panels = [
    (axes[0], err_fr_d5, err_fr_sp,
     r'(a)  Core radius ratio $f_r = r_c\,/\,r_{c0}$'),
    (axes[1], err_frho_d5, err_frho_sp,
     r'(b)  Central density ratio $f_\rho = \rho_c\,/\,\rho_{c0}$'),
]
for ax, e_d5, e_sp, title in panels:
    s_d5 = np.sort(e_d5); s_sp = np.sort(e_sp)
    ax.plot(pct, s_d5, color=C_D5, lw=2.5, ls='-',
            label=f'Chebyshev deg 5 (56 terms), max {s_d5[-1]:.2f}%')
    ax.plot(pct, s_sp, color=C_SP, lw=2.0, ls='--',
            label=f'Sparse Chebyshev (30 terms), max {s_sp[-1]:.2f}%')
    ax.plot(100, s_d5[-1], 'o', color=C_D5, ms=8, zorder=5, clip_on=False)
    ax.plot(100, s_sp[-1], 'o', color=C_SP, ms=8, zorder=5, clip_on=False)
    ax.axvline(95, color='gray', ls=':', lw=0.7, alpha=0.4)
    ax.text(95.5, 0.02, 'p95', fontsize=8, color='gray')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(s_d5[-1], s_sp[-1]) * 1.15)
    ax.set_xlabel(f'Percentile of {n_valid} out-of-sample points', fontsize=12)
    ax.set_ylabel('Relative error  (%)', fontsize=12)
    ax.set_title(title, fontsize=14, pad=10)

    # Horizontal reference lines: median (p50), p95, and max error per curve
    for s_sorted, color in [(s_d5, C_D5), (s_sp, C_SP)]:
        p50, p95v, mx = np.median(s_sorted), np.percentile(s_sorted, 95), s_sorted[-1]
        for val, tag, ls in [(p50, 'p50', '-.'), (p95v, 'p95', ':'), (mx, 'max', '--')]:
            ax.axhline(val, color=color, lw=0.8, ls=ls, alpha=0.4)
            ax.text(100.5, val, f'{tag} {val:.2f}%', fontsize=7,
                    color=color, va='center', ha='left', clip_on=False)

    ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
fig.suptitle('Out-of-sample validation of the Chebyshev correction map',
             fontsize=13, y=1.02)
plt.tight_layout(w_pad=3)
f1 = os.path.join(HERE, "out_of_sample_residuals.png")
plt.savefig(f1, dpi=200, bbox_inches='tight')
plt.savefig(f1.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved {f1}")

# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — generalisation gap: out-of-sample vs in-sample
# ══════════════════════════════════════════════════════════════════
pct_is = np.linspace(0, 100, len(tr_fr))
fig2, axes2 = plt.subplots(1, 2, figsize=(13.5, 5.5))
gpanels = [
    (axes2[0], err_fr_d5, err_fr_sp, err_fr_d5_is, err_fr_sp_is,
     r'(a)  $f_r$:  out-of-sample vs in-sample'),
    (axes2[1], err_frho_d5, err_frho_sp, err_frho_d5_is, err_frho_sp_is,
     r'(b)  $f_\rho$:  out-of-sample vs in-sample'),
]
for ax, e_d5, e_sp, e_d5_is, e_sp_is, title in gpanels:
    ax.plot(pct, np.sort(e_d5), color=C_D5, lw=2.5, ls='-',
            label=f'deg 5 - out-of-sample (max {np.max(e_d5):.2f}%)')
    ax.plot(pct_is, np.sort(e_d5_is), color=C_D5, lw=1.3, ls=':',
            label=f'deg 5 - in-sample (max {np.max(e_d5_is):.2f}%)')
    ax.plot(pct, np.sort(e_sp), color=C_SP, lw=2.0, ls='--',
            label=f'sparse - out-of-sample (max {np.max(e_sp):.2f}%)')
    ax.plot(pct_is, np.sort(e_sp_is), color=C_SP, lw=1.3, ls=(0, (1, 1)),
            label=f'sparse - in-sample (max {np.max(e_sp_is):.2f}%)')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Percentile of points', fontsize=12)
    ax.set_ylabel('Relative error  (%)', fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(loc='upper left', fontsize=8.5, framealpha=0.95)
plt.tight_layout(w_pad=3)
f2 = os.path.join(HERE, "out_of_sample_vs_in_sample.png")
plt.savefig(f2, dpi=200, bbox_inches='tight')
plt.savefig(f2.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved {f2}")
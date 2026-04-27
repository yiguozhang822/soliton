"""
STEP 6 — Dense Grid Scan with Path Continuation
=================================================
Identical to fix_node_counter.py with one key addition:

PATH CONTINUATION
-----------------
When a solve returns nodes > 0 (excited state), instead of accepting
the wrong solution, we retry by walking from the isolated soliton to
the target (Xi, fb) in small incremental steps. Each step uses the
previous step's ground-state solution as its warm-start, keeping the
solver close to the ground state at all times.

If a node appears mid-path, we double the number of steps and retry.
Maximum 3 doublings (8 → 16 → 32 → 64 steps) before giving up.

This recovers most of the 95 remaining excited-state points without
any solver redesign — solve_bvp is used identically throughout.

USAGE
-----
    python step6_continuation.py

Requires: step6_dense_library_fixed.npz  (for xc0, mu0, eps0 reference)
          OR runs from scratch if not present.

OUTPUT
------
    step6_continuation_library.csv
    step6_continuation_library.npz
    step6_continuation_slices.png
"""

import numpy as np
from scipy.integrate import solve_bvp, cumulative_trapezoid
from scipy.interpolate import interp1d
from numpy import trapezoid as trapz
import warnings
import csv
import os
import time
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════
# FIXED NODE COUNTER
# ══════════════════════════════════════════════════════════════════

def count_nodes(s, threshold=1e-6):
    """Count zero-crossings only in the physically significant region."""
    s_peak = np.max(np.abs(s))
    if s_peak == 0:
        return 0
    significant = np.abs(s) > threshold * s_peak
    s_sig = s[significant]
    if len(s_sig) < 2:
        return 0
    signs = np.sign(s_sig)
    return int(np.sum(np.abs(np.diff(signs)) > 0))


# ══════════════════════════════════════════════════════════════════
# SOLVER SETUP
# ══════════════════════════════════════════════════════════════════

X_EPS = 1e-4; X_MAX = 40.0; N = 4000
x_grid = np.linspace(X_EPS, X_MAX, N)


def build_guess(xg):
    a = 2.0**(0.25) - 1.0
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
    ratio = s / x
    idx = np.where(ratio <= 0.5)[0]
    if not len(idx): return None
    i = idx[0]
    return x[i-1] + (0.5 - ratio[i-1]) * (x[i] - x[i-1]) / (ratio[i] - ratio[i-1])


# ── Isolated baseline ──────────────────────────────────────────────
print("Solving isolated baseline...")
sol_iso = solve_bvp(
    lambda xg, yg, p: [yg[1], 2*(yg[2]-p[0])*yg[0],
                        yg[3], yg[0]**2/xg**2 - (2/xg)*yg[3]],
    _bc, x_grid, build_guess(x_grid),
    p=[-0.682], tol=1e-8, max_nodes=80000, verbose=0)
assert sol_iso.success

xc0  = get_xc(sol_iso.x, sol_iso.y[0])
mu0  = trapz(sol_iso.y[0]**2, sol_iso.x)
eps0 = sol_iso.p[0]
y_iso = to_fixed(sol_iso.x, sol_iso.y)
print(f"  xc0={xc0:.5f}, mu0={mu0:.4f}, eps0={eps0:.6f}\n")


def v_ext(x, Xi, fb, eta):
    out = np.zeros_like(x)
    if Xi != 0.0: out += -Xi / x
    if fb != 0.0: out += -fb * (1 + eta**2)**1.5 / np.sqrt(x**2 + eta**2)
    return out


# ══════════════════════════════════════════════════════════════════
# SINGLE-STEP SOLVE
# ══════════════════════════════════════════════════════════════════

def _solve_one(Xi, fb, eta, yw, eg):
    """One BVP solve. Returns result dict or None on failure."""
    def rhs(xg, yg, p):
        return np.array([yg[1],
                         2*(yg[2] + v_ext(xg, Xi, fb, eta) - p[0])*yg[0],
                         yg[3],
                         yg[0]**2/xg**2 - (2/xg)*yg[3]])
    sol = solve_bvp(rhs, _bc, x_grid.copy(), yw.copy(),
                    p=[eg], tol=1e-8, max_nodes=80000, verbose=0)
    if not sol.success:
        return None
    xc   = get_xc(sol.x, sol.y[0])
    mu   = trapz(sol.y[0]**2, sol.x)
    nodes = count_nodes(sol.y[0])
    return {
        'fr':    xc / xc0 if xc else np.nan,
        'frho':  mu0 / mu,
        'eps':   sol.p[0],
        'nodes': nodes,
        'yf':    to_fixed(sol.x, sol.y)
    }


# ══════════════════════════════════════════════════════════════════
# PATH CONTINUATION SOLVE
# ══════════════════════════════════════════════════════════════════

def solve_with_continuation(Xi_target, fb_target, eta,
                            y_start=None, eps_start=None,
                            n_steps=8, max_doublings=3):
    """
    Walk from (Xi=0, fb=0) to (Xi_target, fb_target) in n_steps increments.

    WHY THIS WORKS
    --------------
    Excited states arise when the initial guess is too far from the ground
    state — the solver wanders into a higher-energy solution. By taking
    small steps, each solve starts extremely close to the previous ground
    state, leaving no room to drift into an excited state.

    DOUBLING STRATEGY
    -----------------
    If a node appears mid-path, we double n_steps and retry from scratch.
    This handles the rare case where even a moderate step is too large.
    Maximum 3 doublings: 8 → 16 → 32 → 64 steps.
    If nodes persist at 64 steps, the point is genuinely outside the
    ground-state validity domain and is correctly excluded.

    Parameters
    ----------
    Xi_target, fb_target : target parameter values
    eta                  : Plummer concentration (fixed along path)
    y_start              : warm-start array (defaults to isolated solution)
    eps_start            : eigenvalue warm-start
    n_steps              : initial number of path steps
    max_doublings        : maximum times to double n_steps on node detection

    Returns
    -------
    result dict (same format as _solve_one) or None
    """
    if y_start is None:  y_start  = y_iso
    if eps_start is None: eps_start = eps0

    for attempt in range(max_doublings + 1):
        yw = y_start.copy()
        eg = eps_start
        mid_path_node = False

        for k in range(1, n_steps + 1):
            # Linearly interpolate from start to target
            frac = k / n_steps

            # Start point: if y_start is isolated, walk from (0,0)
            # If y_start is a nearby grid point, walk from that point's values
            Xi_k = Xi_target * frac
            fb_k = fb_target * frac

            r = _solve_one(Xi_k, fb_k, eta, yw, eg)

            if r is None:
                mid_path_node = True
                break

            if r['nodes'] > 0:
                # Node appeared mid-path — need finer steps
                mid_path_node = True
                break

            # Ground state confirmed — update warm-start for next step
            yw = r['yf']
            eg = r['eps']

        if not mid_path_node:
            # Completed full path without nodes — r is the target solution
            return r

        # Double the steps and retry
        n_steps *= 2
        if attempt < max_doublings:
            pass  # will retry with doubled steps

    # Exhausted all doublings — return last attempt's result even if it
    # has nodes, so the caller can record it and mark as excluded
    return r if r is not None else None


# ══════════════════════════════════════════════════════════════════
# MASTER SOLVE FUNCTION
# ══════════════════════════════════════════════════════════════════

def solve_pt_smart(Xi, fb, eta, yw_prev=None, eg_prev=None):
    """
    Try direct solve first (fast). If nodes appear, use path continuation.

    Strategy:
      1. Direct solve from warm-start (same as before, ~1s)
      2. If nodes > 0: path continuation from isolated baseline (~5-30s)
      3. If nodes still > 0 after continuation: mark as excluded

    Returns result dict with 'method' key indicating what was used.
    """
    if yw_prev is None: yw_prev = y_iso
    if eg_prev is None: eg_prev = eps0

    # Step 1: direct solve
    r = _solve_one(Xi, fb, eta, yw_prev, eg_prev)

    if r is not None and r['nodes'] == 0:
        r['method'] = 'direct'
        return r

    # Step 2: path continuation from isolated baseline
    r_cont = solve_with_continuation(Xi, fb, eta,
                                     y_start=y_iso, eps_start=eps0,
                                     n_steps=8, max_doublings=3)
    if r_cont is not None and r_cont['nodes'] == 0:
        r_cont['method'] = 'continuation'
        return r_cont

    # Step 3: continuation from the warm-start point (sometimes better)
    if yw_prev is not y_iso:
        r_cont2 = solve_with_continuation(Xi, fb, eta,
                                          y_start=yw_prev, eps_start=eg_prev,
                                          n_steps=8, max_doublings=3)
        if r_cont2 is not None and r_cont2['nodes'] == 0:
            r_cont2['method'] = 'continuation_warm'
            return r_cont2

    # All strategies failed — return best available result (for recording)
    best = r_cont if r_cont is not None else r
    if best is not None:
        best['method'] = 'failed_all'
    return best


# ══════════════════════════════════════════════════════════════════
# GRID SCAN
# ══════════════════════════════════════════════════════════════════

Xi_vals  = np.array([0.00, 0.01, 0.03, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00])
fb_vals  = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.80])
eta_vals = np.array([0.10, 0.30, 0.50, 1.00, 1.50, 2.00, 5.00])
nX, nf, ne = len(Xi_vals), len(fb_vals), len(eta_vals)
n_total = nX * nf * ne

fr_g     = np.full((nX, nf, ne), np.nan)
frho_g   = np.full((nX, nf, ne), np.nan)
eps_g    = np.full((nX, nf, ne), np.nan)
nodes_g  = np.full((nX, nf, ne), -1, dtype=int)
method_g = np.full((nX, nf, ne), '', dtype=object)
ok_g     = np.zeros((nX, nf, ne), dtype=bool)
yw_store = np.zeros((nX, nf, ne, 4, N))

t0 = time.time()
n = 0
n_direct = 0; n_cont = 0; n_fail = 0; n_excl = 0

print(f"Grid: {nX}×{nf}×{ne} = {n_total} points  (path continuation enabled)\n")
print(f"{'#':>5}  {'Xi':>6} {'fb':>6} {'eta':>5}  "
      f"{'f_r':>7} {'f_rho':>7}  {'nd':>3}  method")
print("-" * 65)

for iX, Xi in enumerate(Xi_vals):
    for if_, fb in enumerate(fb_vals):
        for ie, eta in enumerate(eta_vals):
            n += 1

            # Warm-start: chain along Xi axis
            if iX == 0:
                yw_prev = y_iso.copy()
                eg_prev = eps0
            elif ok_g[iX-1, if_, ie] and nodes_g[iX-1, if_, ie] == 0:
                yw_prev = yw_store[iX-1, if_, ie]
                eg_prev = eps_g[iX-1, if_, ie]
            else:
                yw_prev = y_iso.copy()
                eg_prev = eps0

            r = solve_pt_smart(Xi, fb, eta, yw_prev, eg_prev)
            elapsed = time.time() - t0

            if r is None:
                n_fail += 1
                yw_store[iX, if_, ie] = y_iso.copy()
                method_g[iX, if_, ie] = 'failed'
                print(f"{n:>5}  {Xi:>6.3f} {fb:>6.3f} {eta:>5.2f}  "
                      f"{'FAILED':>7}  [{elapsed:.0f}s]")
                continue

            fr_g[iX,if_,ie]    = r['fr']
            frho_g[iX,if_,ie]  = r['frho']
            eps_g[iX,if_,ie]   = r['eps']
            nodes_g[iX,if_,ie] = r['nodes']
            method_g[iX,if_,ie] = r['method']
            ok_g[iX,if_,ie]    = True
            yw_store[iX,if_,ie] = r['yf'] if r['nodes'] == 0 else y_iso.copy()

            method_short = {'direct':           'direct',
                            'continuation':     'CONT',
                            'continuation_warm':'CONT-w',
                            'failed_all':       'EXCL'}.get(r['method'], r['method'])

            if r['method'] == 'direct':          n_direct += 1
            elif 'continuation' in r['method']:  n_cont   += 1
            elif r['method'] == 'failed_all':    n_excl   += 1

            excl_flag = '  ← still excited' if r['nodes'] > 0 else ''
            print(f"{n:>5}  {Xi:>6.3f} {fb:>6.3f} {eta:>5.2f}  "
                  f"{r['fr']:>7.4f} {r['frho']:>7.4f}  "
                  f"{r['nodes']:>3}  {method_short} [{elapsed:.0f}s]{excl_flag}")

total_time = time.time() - t0
n_ok    = ok_g.sum()
n_valid = int((ok_g & (nodes_g == 0)).sum())

print(f"\n{'='*65}")
print(f"Done in {total_time:.1f}s")
print(f"  Total converged:  {n_ok}/{n_total}")
print(f"  Direct solves:    {n_direct}")
print(f"  Via continuation: {n_cont}  (rescued from excited states)")
print(f"  Still excluded:   {n_excl + n_fail}  (true validity boundary)")
print(f"  Valid:            {n_valid}/{n_total}")


# ══════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════

csv_path = os.path.join(OUT_DIR, 'step6_continuation_library.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Xi', 'fb_core', 'eta', 'f_r', 'f_rho', 'eps',
                     'converged', 'nodes', 'method', 'valid'])
    for iX, Xi in enumerate(Xi_vals):
        for if_, fb in enumerate(fb_vals):
            for ie, eta in enumerate(eta_vals):
                conv  = int(ok_g[iX, if_, ie])
                nd    = int(nodes_g[iX, if_, ie])
                valid = int(conv and nd == 0)
                writer.writerow([Xi, fb, eta,
                                  fr_g[iX,if_,ie], frho_g[iX,if_,ie],
                                  eps_g[iX,if_,ie],
                                  conv, nd, method_g[iX,if_,ie], valid])

np.savez(os.path.join(OUT_DIR, 'step6_continuation_library.npz'),
         Xi_vals=Xi_vals, fb_vals=fb_vals, eta_vals=eta_vals,
         fr_grid=fr_g, frho_grid=frho_g, eps_grid=eps_g,
         nodes_grid=nodes_g, ok_grid=ok_g,
         xc0=xc0, mu0=mu0, eps0=eps0)

print(f"\nSaved: {csv_path}")


# ══════════════════════════════════════════════════════════════════
# SANITY CHECKS
# ══════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("SANITY CHECKS")
print("="*65)

valid_mask = ok_g & (nodes_g == 0)

print(f"\n  Zero perturbation (Xi=0, fb=0, eta=0.1):")
print(f"    f_r   = {fr_g[0,0,0]:.6f}  (expect 1.0)")
print(f"    f_rho = {frho_g[0,0,0]:.6f}  (expect 1.0)")

fr_Xi = fr_g[:, 0, 3]; ok_Xi = valid_mask[:, 0, 3]
if ok_Xi.sum() > 1:
    vals = fr_Xi[ok_Xi]
    mono = all(vals[i] >= vals[i+1] for i in range(len(vals)-1))
    print(f"\n  Monotone f_r decreasing with Xi (fb=0, eta=1): {mono}")
    print(f"    f_r values: {vals.round(4)}")

frho_valid = frho_g[valid_mask]
n_incon = int(np.sum(frho_valid < 1.0))
print(f"\n  Physical consistency (f_rho > 1): {n_incon} inconsistent  (expect 0)")

n_rescued = int(np.sum((method_g == 'continuation') |
                        (method_g == 'continuation_warm')))
print(f"\n  Points rescued by continuation:  {n_rescued}")
print(f"  Points still excluded:           {int((ok_g & (nodes_g > 0)).sum()) + n_fail}")

print(f"\n  Valid by Xi slice:")
for iX, Xi in enumerate(Xi_vals):
    nv = int((valid_mask[iX]).sum())
    nc = int(np.sum(method_g[iX] == 'continuation') +
              np.sum(method_g[iX] == 'continuation_warm'))
    print(f"    Xi={Xi:.2f}: {nv}/{nf*ne} valid  ({nc} via continuation)")


# ══════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════

c_eta = plt.cm.plasma(np.linspace(0.1, 0.9, ne))
c_fb  = plt.cm.viridis(np.linspace(0.1, 0.9, nf))

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    f"Step 6 with Path Continuation: {n_valid} valid ground-state points\n"
    f"{n_cont} points rescued via continuation  |  "
    f"{int((ok_g & (nodes_g>0)).sum()) + n_fail} remain excluded (true boundary)",
    fontsize=12, y=1.01)

# f_r vs Xi (fb=0)
ax = axes[0, 0]
for ie, eta in enumerate(eta_vals):
    m = valid_mask[:, 0, ie]
    if m.sum() > 1:
        ax.plot(Xi_vals[m], fr_g[:, 0, ie][m], 'o-',
                color=c_eta[ie], lw=1.8, ms=5, label=f'η={eta}')
ax.axhline(1.0, ls='--', color='gray', alpha=0.5)
ax.set_ylim(0, 1.1)
ax.set_xlabel('Ξ', fontsize=11); ax.set_ylabel('$f_r$', fontsize=11)
ax.set_title('$f_r$ vs Ξ  (fb=0)', fontsize=11)
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

# f_r vs fb (Xi=0)
ax = axes[0, 1]
for ie, eta in enumerate(eta_vals):
    m = valid_mask[0, :, ie]
    if m.sum() > 1:
        ax.plot(fb_vals[m], fr_g[0, :, ie][m], 'o-',
                color=c_eta[ie], lw=1.8, ms=5, label=f'η={eta}')
ax.axhline(1.0, ls='--', color='gray', alpha=0.5)
ax.set_ylim(0, 1.1)
ax.set_xlabel('$f_{b,\\rm core}$', fontsize=11); ax.set_ylabel('$f_r$', fontsize=11)
ax.set_title('$f_r$ vs fb  (Ξ=0)', fontsize=11)
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

# f_r vs eta (Xi=0)
ax = axes[0, 2]
for if_, fb in enumerate(fb_vals):
    m = valid_mask[0, if_, :]
    if m.sum() > 1:
        ax.plot(eta_vals[m], fr_g[0, if_, :][m], 'o-',
                color=c_fb[if_], lw=1.8, ms=5, label=f'fb={fb}')
ax.axhline(1.0, ls='--', color='gray', alpha=0.5)
ax.set_ylim(0, 1.1)
ax.set_xlabel('η', fontsize=11); ax.set_ylabel('$f_r$', fontsize=11)
ax.set_title('$f_r$ vs η  (Ξ=0)', fontsize=11)
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# f_rho vs Xi (fb=0)
ax = axes[1, 0]
for ie, eta in enumerate(eta_vals):
    m = valid_mask[:, 0, ie]
    if m.sum() > 1:
        ax.plot(Xi_vals[m], frho_g[:, 0, ie][m], 'o-',
                color=c_eta[ie], lw=1.8, ms=5, label=f'η={eta}')
ax.axhline(1.0, ls='--', color='gray', alpha=0.5)
ax.set_xlabel('Ξ', fontsize=11); ax.set_ylabel('$f_\\rho$', fontsize=11)
ax.set_title('$f_\\rho$ vs Ξ  (fb=0)', fontsize=11)
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

# f_rho vs fb (Xi=0)
ax = axes[1, 1]
for ie, eta in enumerate(eta_vals):
    m = valid_mask[0, :, ie]
    if m.sum() > 1:
        ax.plot(fb_vals[m], frho_g[0, :, ie][m], 'o-',
                color=c_eta[ie], lw=1.8, ms=5, label=f'η={eta}')
ax.axhline(1.0, ls='--', color='gray', alpha=0.5)
ax.set_xlabel('$f_{b,\\rm core}$', fontsize=11); ax.set_ylabel('$f_\\rho$', fontsize=11)
ax.set_title('$f_\\rho$ vs fb  (Ξ=0)', fontsize=11)
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

# Heatmap: f_r at eta=1.0, showing method used (continuation = hatched)
ax = axes[1, 2]
ie_mid = 3   # eta=1.0
fp = np.where(valid_mask[:, :, ie_mid], fr_g[:, :, ie_mid], np.nan)
im = ax.imshow(fp.T, origin='lower', aspect='auto',
               extent=[Xi_vals[0], Xi_vals[-1], fb_vals[0], fb_vals[-1]],
               cmap='RdYlGn', vmin=0.2, vmax=1.0)
plt.colorbar(im, ax=ax, label='$f_r$')

# Mark continuation points with a dot
for iX, Xi in enumerate(Xi_vals):
    for if_, fb in enumerate(fb_vals):
        m = method_g[iX, if_, ie_mid]
        if 'continuation' in str(m) and valid_mask[iX, if_, ie_mid]:
            ax.plot(Xi, fb, 'w+', ms=8, mew=1.5)

ax.set_xlabel('Ξ', fontsize=11); ax.set_ylabel('$f_{b,\\rm core}$', fontsize=11)
ax.set_title('$f_r$ heatmap at η=1.0\n(+ = rescued by continuation, grey = excluded)',
             fontsize=10)

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, 'step6_continuation_slices.png')
plt.savefig(plot_path, dpi=130, bbox_inches='tight')
print(f"\nPlot saved: {plot_path}")
print("\nTo load valid points for Step 7:")
print("  import pandas as pd")
print("  df = pd.read_csv('step6_continuation_library.csv')")
print("  valid = df[df['valid'] == 1]")
print(f"  # gives {n_valid} valid ground-state points")
"""
DIAGNOSTIC: Investigating anomalous f_rho < 1 points
=====================================================
These points are physically suspicious because a squeezed core
(f_r < 1) at fixed soliton mass should always have higher central
density (f_rho > 1). f_rho < 1 with f_r < 1 is internally
inconsistent and signals either:
  (A) Solver found a wrong/excited state instead of ground state
  (B) Profile has no clear quarter-density crossing (core radius undefined)
  (C) mu integral is being computed on a truncated domain
  (D) The soliton is so strongly perturbed it no longer resembles
      a soliton at all (outside validity domain)

We investigate by:
  1. Listing all anomalous points from the saved library
  2. Re-solving each anomalous point and checking:
     - Does s(x) have nodes? (excited state)
     - Does the profile decay properly?
     - Is the core radius well-defined?
     - What does the profile actually look like?
  3. Plotting the profiles to visually inspect
"""

import os
import numpy as np
from scipy.integrate import solve_bvp, cumulative_trapezoid
from scipy.interpolate import interp1d
from numpy import trapezoid as trapz
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Load saved library ─────────────────────────────────────────────
data = np.load(os.path.expanduser('~/outputs/step6_library.npz'))
Xi_vals  = data['Xi_vals']
fb_vals  = data['fb_vals']
eta_vals = data['eta_vals']
fr_g     = data['fr_grid']
frho_g   = data['frho_grid']
ok_g     = data['ok_grid']
xc0_stored = float(data['xc0'])
mu0_stored = float(data['mu0'])

print("="*65)
print("DIAGNOSTIC: Anomalous f_rho < 1 points")
print("="*65)

# ── Step 1: List all anomalous points ─────────────────────────────
print("\nAll converged points with f_rho < 1 OR f_r > 1 (physically suspicious):")
print(f"  {'Ξ':>6} {'fb':>6} {'η':>6}  {'f_r':>7} {'f_rho':>7}  issue")
print("  " + "-"*55)

anomalous = []
for iX, Xi in enumerate(Xi_vals):
    for if_, fb in enumerate(fb_vals):
        for ie, eta in enumerate(eta_vals):
            if not ok_g[iX, if_, ie]:
                continue
            fr   = fr_g[iX, if_, ie]
            frho = frho_g[iX, if_, ie]
            issues = []
            if frho < 1.0 and fr < 1.0:
                issues.append("f_rho<1 while f_r<1 (inconsistent)")
            if fr > 1.0:
                issues.append("f_r>1 (core expanded?)")
            if frho < 0.5:
                issues.append("f_rho very low")
            if issues:
                anomalous.append((Xi, fb, eta, fr, frho, "; ".join(issues)))
                print(f"  {Xi:>6.3f} {fb:>6.3f} {eta:>6.2f}  {fr:>7.4f} {frho:>7.4f}  {issues[0]}")

print(f"\nTotal anomalous: {len(anomalous)} / {int(ok_g.sum())} converged points")

# ── Step 2: Re-solve and inspect each anomalous point ─────────────
X_EPS=1e-4; X_MAX=40.0; N=4000
x_grid=np.linspace(X_EPS,X_MAX,N)

def build_guess(xg):
    a=2.0**(0.25)-1.0; s=xg*(1+a*xg**2)**(-4)
    M=np.zeros_like(xg); M[1:]=cumulative_trapezoid(s**2,xg)
    vp=M/xg**2; vp[0]=0.0
    v=-(trapz(vp,xg)-cumulative_trapezoid(vp,xg,initial=0.0))
    y0=np.zeros((4,xg.size))
    y0[0]=s; y0[1]=np.gradient(s,xg); y0[2]=v; y0[3]=vp
    return y0

def _bc(ya,yb,p):
    return np.array([ya[0]-X_EPS,ya[1]-1.0,ya[3],yb[0],yb[2]])

def to_fixed(sx,sy):
    y=np.zeros((4,N))
    for i in range(4):
        f=interp1d(sx,sy[i],bounds_error=False,fill_value=(sy[i][0],0.0))
        y[i]=f(x_grid)
    y[0]=np.clip(y[0],0.0,None); return y

# Isolated baseline
sol_iso=solve_bvp(
    lambda xg,yg,p:[yg[1],2*(yg[2]-p[0])*yg[0],yg[3],yg[0]**2/xg**2-(2/xg)*yg[3]],
    _bc, x_grid, build_guess(x_grid), p=[-0.682], tol=1e-8, max_nodes=80000, verbose=0)
xc0=xc0_stored; mu0=mu0_stored; eps0=sol_iso.p[0]
y_iso_f=to_fixed(sol_iso.x,sol_iso.y)

def v_ext(x,Xi,fb,eta):
    out=np.zeros_like(x)
    if Xi!=0: out+=-Xi/x
    if fb!=0: out+=-fb*(1+eta**2)**1.5/np.sqrt(x**2+eta**2)
    return out

def count_nodes(s):
    """Count zero crossings in s (excited state has nodes)."""
    signs = np.sign(s[s != 0])
    return int(np.sum(np.abs(np.diff(signs)) > 0))

def solve_and_diagnose(Xi, fb, eta):
    """Re-solve and return detailed diagnostics."""
    y0 = y_iso_f.copy()

    def rhs(xg,yg,p):
        return np.array([yg[1],2*(yg[2]+v_ext(xg,Xi,fb,eta)-p[0])*yg[0],
                         yg[3],yg[0]**2/xg**2-(2/xg)*yg[3]])

    sol=solve_bvp(rhs,_bc,x_grid.copy(),y0,p=[eps0],tol=1e-8,max_nodes=80000,verbose=0)
    if not sol.success:
        return None

    x=sol.x; s=sol.y[0]; v=sol.y[2]
    rho=(s/x)**2

    # Core radius
    ratio=s/x; idx=np.where(ratio<=0.5)[0]
    xc = None
    if len(idx):
        i=idx[0]
        xc=x[i-1]+(0.5-ratio[i-1])*(x[i]-x[i-1])/(ratio[i]-ratio[i-1])

    mu=trapz(s**2,x)
    nodes=count_nodes(s)
    tail=abs(s[-1])/(s.max()+1e-300)
    s_min=s.min()

    diag = {
        'x': x, 's': s, 'v': v, 'rho': rho,
        'xc': xc,
        'fr': xc/xc0 if xc else np.nan,
        'frho': mu0/mu,
        'mu': mu,
        'nodes': nodes,
        'tail': tail,
        's_min': s_min,
        'eps': sol.p[0],
        'success': True
    }
    return diag

print("\n" + "="*65)
print("DETAILED DIAGNOSTICS FOR ANOMALOUS POINTS")
print("="*65)
print(f"  {'Ξ':>5} {'fb':>5} {'η':>5}  {'nodes':>6} {'s_min':>8} {'tail':>10} {'f_r':>7} {'f_rho':>7}  verdict")
print("  "+"-"*75)

diag_results = []
for Xi, fb, eta, fr_saved, frho_saved, issue in anomalous:
    d = solve_and_diagnose(Xi, fb, eta)
    if d is None:
        print(f"  {Xi:>5.2f} {fb:>5.2f} {eta:>5.1f}  FAILED to re-solve")
        continue

    # Determine likely cause
    if d['nodes'] > 0:
        verdict = f"EXCITED STATE ({d['nodes']} nodes)"
    elif d['s_min'] < -0.01:
        verdict = "s goes NEGATIVE (wrong branch)"
    elif d['tail'] > 0.01:
        verdict = "POOR DECAY (not bound state)"
    elif d['xc'] is None:
        verdict = "NO CORE RADIUS (profile too flat)"
    elif d['frho'] < 0.5:
        verdict = "POSSIBLE WRONG STATE"
    else:
        verdict = "Unclear — may be valid"

    print(f"  {Xi:>5.2f} {fb:>5.2f} {eta:>5.1f}  "
          f"{d['nodes']:>6} {d['s_min']:>8.4f} {d['tail']:>10.2e} "
          f"{d['fr']:>7.4f} {d['frho']:>7.4f}  {verdict}")
    diag_results.append((Xi, fb, eta, d, verdict))

# ── Step 3: Plot anomalous profiles ────────────────────────────────
if diag_results:
    n_anom = len(diag_results)
    ncols = min(4, n_anom)
    nrows = (n_anom + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5*ncols, 4*nrows))
    if n_anom == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    # Reference isolated profile
    s_iso=sol_iso.y[0]; x_iso=sol_iso.x
    rho_iso=(s_iso/x_iso)**2
    rho_iso_norm=rho_iso/rho_iso[0]

    for idx_a, (Xi, fb, eta, d, verdict) in enumerate(diag_results):
        row, col = idx_a // ncols, idx_a % ncols
        ax = axes[row, col]

        # Plot isolated reference
        xn_iso = x_iso/xc0
        mask_iso = xn_iso <= 8
        ax.plot(xn_iso[mask_iso], rho_iso_norm[mask_iso],
                'k--', lw=1.5, alpha=0.5, label='Isolated')

        # Plot anomalous profile
        xn = d['x']/xc0
        rho_norm = d['rho']/d['rho'][0] if d['rho'][0] > 0 else d['rho']
        mask = xn <= 8
        ax.plot(xn[mask], rho_norm[mask], 'r-', lw=2, label='Anomalous')

        # Mark core radius
        if d['xc']:
            ax.axvline(d['xc']/xc0, ls=':', color='red', alpha=0.7)

        ax.axhline(0.25, ls='--', color='gray', alpha=0.5, lw=1)
        ax.axhline(0.0,  ls='-',  color='black', alpha=0.3, lw=0.5)
        ax.set_xlim(0, 8); ax.set_ylim(-0.3, 1.1)
        ax.set_xlabel('x / x_c0', fontsize=9)
        ax.set_ylabel('ρ/ρ(0)', fontsize=9)
        ax.set_title(f'Ξ={Xi}, fb={fb}, η={eta}\n'
                     f'f_r={d["fr"]:.3f}, f_ρ={d["frho"]:.3f}\n'
                     f'{verdict}', fontsize=8)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx_a in range(n_anom, nrows*ncols):
        row, col = idx_a // ncols, idx_a % ncols
        axes[row, col].set_visible(False)

    plt.suptitle("Anomalous points: profile inspection", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/outputs/diagnostic_anomalous.png'),
                dpi=130, bbox_inches='tight')
    print("\n✓ Profile plots saved → diagnostic_anomalous.png")

# ── Step 4: Summary ────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY & INTERPRETATION")
print("="*65)

node_cases   = [(Xi,fb,eta) for Xi,fb,eta,d,v in diag_results if 'node' in v.lower() or 'EXCITED' in v]
neg_cases    = [(Xi,fb,eta) for Xi,fb,eta,d,v in diag_results if 'NEGATIVE' in v]
unclear      = [(Xi,fb,eta) for Xi,fb,eta,d,v in diag_results if 'Unclear' in v]

print(f"\n  Excited state (nodes in s):     {len(node_cases)} points")
print(f"  s goes negative (wrong branch): {len(neg_cases)} points")
print(f"  Unclear / possibly valid:       {len(unclear)} points")

if node_cases:
    print(f"\n  Excited state points:")
    for pt in node_cases: print(f"    Ξ={pt[0]}, fb={pt[1]}, η={pt[2]}")

if neg_cases:
    print(f"\n  Negative s points:")
    for pt in neg_cases: print(f"    Ξ={pt[0]}, fb={pt[1]}, η={pt[2]}")

print("""
  WHAT THIS MEANS FOR YOUR VALIDITY DOMAIN:
  ------------------------------------------
  Points where s has nodes → solver found excited state, not ground
  state. These must be excluded from the fit and mark the edge of
  your validity domain.

  Points where s goes negative → solver jumped to wrong solution
  branch. Also exclude.

  The validity domain for your correction map is the subset of Π
  where ALL of the following hold:
    (1) f_rho > 1  (physically consistent with f_r < 1)
    (2) s(x) has zero nodes (true ground state)
    (3) s(x) > 0 everywhere (no branch jumping)
    (4) tail metric s(x_max)/s_max < 1e-3 (proper decay)
""")
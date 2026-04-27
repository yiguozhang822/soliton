import numpy as np
from scipy.integrate import solve_bvp, cumulative_trapezoid
from scipy.interpolate import interp1d
from numpy import trapezoid as trapz
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Solver setup (identical to Step 5) ────────────────────────────
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

def get_xc(x,s):
    r=s/x; idx=np.where(r<=0.5)[0]
    if not len(idx): return np.nan
    i=idx[0]
    return x[i-1]+(0.5-r[i-1])*(x[i]-x[i-1])/(r[i]-r[i-1])

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
xc0=get_xc(sol_iso.x,sol_iso.y[0])
mu0=trapz(sol_iso.y[0]**2,sol_iso.x)
eps0=sol_iso.p[0]
y_iso_f=to_fixed(sol_iso.x,sol_iso.y)
print(f"Isolated baseline: xc0={xc0:.4f}, mu0={mu0:.4f}")

def v_ext(x,Xi,fb,eta):
    out=np.zeros_like(x)
    if Xi!=0: out+=-Xi/x
    if fb!=0: out+=-fb*(1+eta**2)**1.5/np.sqrt(x**2+eta**2)
    return out

def solve_pt(Xi,fb,eta,yw=None,eg=None):
    if yw is None: yw=y_iso_f.copy()
    if eg is None: eg=eps0
    def rhs(xg,yg,p):
        return np.array([yg[1],2*(yg[2]+v_ext(xg,Xi,fb,eta)-p[0])*yg[0],
                         yg[3],yg[0]**2/xg**2-(2/xg)*yg[3]])
    sol=solve_bvp(rhs,_bc,x_grid.copy(),yw.copy(),p=[eg],tol=1e-8,max_nodes=80000,verbose=0)
    if not sol.success: return None
    xc=get_xc(sol.x,sol.y[0]); mu=trapz(sol.y[0]**2,sol.x)
    rho=(sol.y[0]/sol.x)**2
    rho_norm=rho/rho[0]
    return {'x':sol.x,'s':sol.y[0],'v':sol.y[2],
            'rho_norm':rho_norm,'xc':xc,
            'fr':xc/xc0,'frho':mu0/mu,'eps':sol.p[0],
            'yf':to_fixed(sol.x,sol.y)}

print("Solving grid for plots (this takes ~2 min)...")

# ── Parameter sets to solve ────────────────────────────────────────
Xi_vals  = [0.00,0.05,0.10,0.20,0.50,1.00]
fb_vals  = [0.00,0.10,0.20,0.50,0.80]
eta_vals = [0.10,0.30,0.50,1.00,2.00,5.00]

# Colours
c_Xi  = plt.cm.plasma(np.linspace(0.05,0.92,len(Xi_vals)))
c_fb  = plt.cm.viridis(np.linspace(0.05,0.92,len(fb_vals)))
c_eta = plt.cm.cool(np.linspace(0.05,0.92,len(eta_vals)))

# Solve SMBH series (fb=0, varying Xi)
print("  SMBH series...")
Xi_sols={}
yw=y_iso_f.copy(); eg=eps0
for Xi in Xi_vals:
    r=solve_pt(Xi,0.0,1.0,yw,eg)
    if r:
        Xi_sols[Xi]=r; yw=r['yf']; eg=r['eps']
    print(f"    Xi={Xi}: fr={r['fr']:.4f}, frho={r['frho']:.4f}")

# Solve baryon series (Xi=0, eta=0.5, varying fb)
print("  Baryon series (eta=0.5)...")
fb_sols={}
yw=y_iso_f.copy(); eg=eps0
for fb in fb_vals:
    r=solve_pt(0.0,fb,0.5,yw,eg)
    if r:
        fb_sols[fb]=r; yw=r['yf']; eg=r['eps']
    print(f"    fb={fb}: fr={r['fr']:.4f}, frho={r['frho']:.4f}")

# Solve eta series (Xi=0, fb=0.3, varying eta)
print("  Eta series (fb=0.3)...")
eta_sols={}
for eta in eta_vals:
    r=solve_pt(0.0,0.3,eta)
    if r:
        eta_sols[eta]=r
    print(f"    eta={eta}: fr={r['fr']:.4f}, frho={r['frho']:.4f}")

# Solve combined series (Xi=fb, eta=0.5)
print("  Combined series...")
comb_vals=[0.0,0.05,0.10,0.20,0.50]
comb_sols={}
c_comb=plt.cm.autumn(np.linspace(0.1,0.9,len(comb_vals)))
yw=y_iso_f.copy(); eg=eps0
for v in comb_vals:
    r=solve_pt(v,v,0.5,yw,eg)
    if r:
        comb_sols[v]=r; yw=r['yf']; eg=r['eps']
    print(f"    Xi=fb={v}: fr={r['fr']:.4f}, frho={r['frho']:.4f}")

print("All solves done. Building plots...")

# ══════════════════════════════════════════════════════════════════
# FIGURE 1: SOLITON PROFILES — how shape changes with each parameter
# ══════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 3, figsize=(18, 11))
fig1.suptitle("Step 5: Soliton density profiles under external potentials",
               fontsize=15, fontweight='bold', y=1.01)

x_plot_max = 6.0
x_plot_min = 0.05   # log axis lower bound (must be > 0)
rho_y_min  = 1e-3   # log axis lower bound for density
rho_y_max  = 1.5

# ── Plot A: profiles vs Ξ ─────────────────────────────────────────
ax = axes[0,0]
for i,(Xi,r) in enumerate(Xi_sols.items()):
    x_n = r['x']/xc0
    mask = (x_n >= x_plot_min) & (x_n <= x_plot_max) & (r['rho_norm'] > 0)
    ax.plot(x_n[mask], r['rho_norm'][mask],
            color=c_Xi[i], lw=2.0,
            label=f'Ξ = {Xi}' + (' (isolated)' if Xi==0 else ''))
ax.axhline(0.25, ls='--', color='gray', alpha=0.6, lw=1, label='ρ/ρ(0) = ¼')
ax.axvline(1.0,  ls=':',  color='gray', alpha=0.4, lw=1, label='r = r_c0')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(x_plot_min, x_plot_max); ax.set_ylim(rho_y_min, rho_y_max)
ax.set_xlabel('$x = r / r_{c0}$', fontsize=12)
ax.set_ylabel('$\\rho(r) / \\rho(0)$', fontsize=12)
ax.set_title('Increasing SMBH strength (fb=0)', fontsize=12)
ax.legend(fontsize=9, loc='lower left'); ax.grid(alpha=0.3, which='both')

# ── Plot B: profiles vs fb ────────────────────────────────────────
ax = axes[0,1]
for i,(fb,r) in enumerate(fb_sols.items()):
    x_n = r['x']/xc0
    mask = (x_n >= x_plot_min) & (x_n <= x_plot_max) & (r['rho_norm'] > 0)
    ax.plot(x_n[mask], r['rho_norm'][mask],
            color=c_fb[i], lw=2.0,
            label=f'$f_{{b}}$ = {fb}' + (' (isolated)' if fb==0 else ''))
ax.axhline(0.25, ls='--', color='gray', alpha=0.6, lw=1)
ax.axvline(1.0,  ls=':',  color='gray', alpha=0.4, lw=1)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(x_plot_min, x_plot_max); ax.set_ylim(rho_y_min, rho_y_max)
ax.set_xlabel('$x = r / r_{c0}$', fontsize=12)
ax.set_ylabel('$\\rho(r) / \\rho(0)$', fontsize=12)
ax.set_title('Increasing baryon fraction (Ξ=0, η=0.5)', fontsize=12)
ax.legend(fontsize=9, loc='lower left'); ax.grid(alpha=0.3, which='both')

# ── Plot C: profiles vs η ─────────────────────────────────────────
ax = axes[0,2]
for i,(eta,r) in enumerate(eta_sols.items()):
    x_n = r['x']/xc0
    mask = (x_n >= x_plot_min) & (x_n <= x_plot_max) & (r['rho_norm'] > 0)
    ax.plot(x_n[mask], r['rho_norm'][mask],
            color=c_eta[i], lw=2.0, label=f'η = {eta}')
ax.axhline(0.25, ls='--', color='gray', alpha=0.6, lw=1)
ax.axvline(1.0,  ls=':',  color='gray', alpha=0.4, lw=1)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(x_plot_min, x_plot_max); ax.set_ylim(rho_y_min, rho_y_max)
ax.set_xlabel('$x = r / r_{c0}$', fontsize=12)
ax.set_ylabel('$\\rho(r) / \\rho(0)$', fontsize=12)
ax.set_title('Varying baryon concentration (Ξ=0, fb=0.3)', fontsize=12)
ax.legend(fontsize=9, loc='lower left'); ax.grid(alpha=0.3, which='both')

# ── Plot D: wavefunction s(x) vs Ξ ───────────────────────────────
ax = axes[1,0]
for i,(Xi,r) in enumerate(Xi_sols.items()):
    x_n = r['x']/xc0
    phi = r['s'] / r['x']
    mask = (x_n >= x_plot_min) & (x_n <= x_plot_max) & (phi > 0)
    ax.plot(x_n[mask], phi[mask],
            color=c_Xi[i], lw=2.0, label=f'Ξ = {Xi}')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(x_plot_min, x_plot_max); ax.set_ylim(rho_y_min, rho_y_max)
ax.set_xlabel('$x = r / r_{c0}$', fontsize=12)
ax.set_ylabel('$s(x)/x = \\hat{\\phi}(x) / \\hat{\\phi}(0)$', fontsize=12)
ax.set_title('Wavefunction $\\hat{\\phi}(x)$ vs SMBH strength', fontsize=12)
ax.legend(fontsize=9, loc='lower left'); ax.grid(alpha=0.3, which='both')

# ── Plot E: external potential shape ─────────────────────────────
# y is negative so only log x is applied
ax = axes[1,1]
x_pot = np.linspace(0.05, 6.0, 500)
for i,Xi in enumerate([0.05,0.10,0.20,0.50,1.00]):
    vx = -Xi/x_pot
    ax.plot(x_pot, vx, color=c_Xi[i+1], lw=2.0, ls='-', label=f'SMBH Ξ={Xi}')
for i,fb in enumerate([0.10,0.30,0.50]):
    vx = -fb*(1+0.5**2)**1.5/np.sqrt(x_pot**2+0.5**2)
    ax.plot(x_pot, vx, lw=2.0, ls='--', alpha=0.8, label=f'Plummer fb={fb},η=0.5')
ax.axvline(1.0, ls=':', color='gray', alpha=0.5, lw=1, label='$r_{c0}$')
ax.set_xscale('log')
ax.set_xlim(0.05, 6); ax.set_ylim(-12, 0.3)
ax.set_xlabel('$x = r / r_{c0}$', fontsize=12)
ax.set_ylabel('$v_{\\rm ext}(x)$', fontsize=12)
ax.set_title('External potential shapes', fontsize=12)
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3, which='both')

# ── Plot F: combined effect ───────────────────────────────────────
ax = axes[1,2]
for i,(v,r) in enumerate(comb_sols.items()):
    x_n = r['x']/xc0
    mask = (x_n >= x_plot_min) & (x_n <= x_plot_max) & (r['rho_norm'] > 0)
    ax.plot(x_n[mask], r['rho_norm'][mask],
            color=c_comb[i], lw=2.0,
            label=f'Ξ=fb={v}' + (' (isolated)' if v==0 else ''))
ax.axhline(0.25, ls='--', color='gray', alpha=0.6, lw=1)
ax.axvline(1.0,  ls=':',  color='gray', alpha=0.4, lw=1)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(x_plot_min, x_plot_max); ax.set_ylim(rho_y_min, rho_y_max)
ax.set_xlabel('$x = r / r_{c0}$', fontsize=12)
ax.set_ylabel('$\\rho(r) / \\rho(0)$', fontsize=12)
ax.set_title('Combined SMBH + baryons (η=0.5)', fontsize=12)
ax.legend(fontsize=9, loc='lower left'); ax.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('/Users/yiguozhang/step5_profiles.png', dpi=140, bbox_inches='tight')
plt.show()
print("Saved: step5_profiles.png")

# ══════════════════════════════════════════════════════════════════
# FIGURE 2: CORRECTION FACTORS f_r and f_ρ vs each parameter
# ══════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 11))
fig2.suptitle("Step 5: Correction factors $f_r$ and $f_\\rho$ across parameter space",
               fontsize=15, fontweight='bold', y=1.01)

# Dense Ξ scan for smooth curves (fb=0)
Xi_dense = np.array([0,0.01,0.03,0.05,0.08,0.10,0.15,0.20,0.30,0.50,0.75,1.00])
fr_Xi=[]; frho_Xi=[]; yw=y_iso_f.copy(); eg=eps0
for Xi in Xi_dense:
    r=solve_pt(Xi,0.0,1.0,yw,eg)
    if r: fr_Xi.append(r['fr']); frho_Xi.append(r['frho']); yw=r['yf']; eg=r['eps']
    else: fr_Xi.append(np.nan); frho_Xi.append(np.nan)
fr_Xi=np.array(fr_Xi); frho_Xi=np.array(frho_Xi)

# Dense fb scan (Xi=0, eta=0.5)
fb_dense = np.array([0,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.80])
fr_fb=[]; frho_fb=[]; yw=y_iso_f.copy(); eg=eps0
for fb in fb_dense:
    r=solve_pt(0.0,fb,0.5,yw,eg)
    if r: fr_fb.append(r['fr']); frho_fb.append(r['frho']); yw=r['yf']; eg=r['eps']
    else: fr_fb.append(np.nan); frho_fb.append(np.nan)
fr_fb=np.array(fr_fb); frho_fb=np.array(frho_fb)

# ── Plot 1: f_r vs Ξ (semilogx — Xi=0 excluded from log axis) ────
ax=axes2[0,0]
mask_Xi = Xi_dense > 0
ax.plot(Xi_dense[mask_Xi], fr_Xi[mask_Xi], 'o-', color='royalblue', lw=2.5, ms=7, zorder=3)
ax.fill_between(Xi_dense[mask_Xi], fr_Xi[mask_Xi], 1.0, alpha=0.12, color='royalblue')
ax.axhline(1.0, ls='--', color='gray', alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel('Ξ  (SMBH strength)', fontsize=12)
ax.set_ylabel('$f_r = r_c / r_{c0}$', fontsize=12)
ax.set_title('Core shrinks with SMBH strength', fontsize=12)
ax.set_ylim(0.2, 1.1); ax.grid(alpha=0.3, which='both')
ax.annotate('core\nshrinks', xy=(0.7,0.55), fontsize=10, color='royalblue',
            ha='center', style='italic')

# ── Plot 2: f_ρ vs Ξ (semilogx) ──────────────────────────────────
ax=axes2[0,1]
ax.plot(Xi_dense[mask_Xi], frho_Xi[mask_Xi], 's-', color='crimson', lw=2.5, ms=7, zorder=3)
ax.fill_between(Xi_dense[mask_Xi], frho_Xi[mask_Xi], 1.0, alpha=0.12, color='crimson')
ax.axhline(1.0, ls='--', color='gray', alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel('Ξ  (SMBH strength)', fontsize=12)
ax.set_ylabel('$f_\\rho = \\rho_c / \\rho_{c0}$', fontsize=12)
ax.set_title('Core gets denser with SMBH strength', fontsize=12)
ax.grid(alpha=0.3, which='both')
ax.annotate('core\ndenser', xy=(0.7, np.nanmin(frho_Xi[mask_Xi])*1.05), fontsize=10,
            color='crimson', ha='center', style='italic')

# ── Plot 3: log-log power law for f_r (unchanged) ────────────────
ax=axes2[0,2]
Xi_pos = np.array(Xi_dense[1:])
fr_pos  = np.array(fr_Xi[1:])
frho_pos= np.array(frho_Xi[1:])
valid = np.isfinite(fr_pos) & (fr_pos>0)
ax.loglog(Xi_pos[valid], fr_pos[valid], 'o-', color='royalblue',
          lw=2.0, ms=7, label='$f_r$')
ax.loglog(Xi_pos[valid], frho_pos[valid], 's-', color='crimson',
          lw=2.0, ms=7, label='$f_\\rho$')
if valid.sum()>=3:
    sl,ic=np.polyfit(np.log(Xi_pos[valid]),np.log(fr_pos[valid]),1)
    xr=np.logspace(np.log10(Xi_pos[valid].min()),np.log10(Xi_pos[valid].max()),80)
    ax.loglog(xr,np.exp(ic)*xr**sl,'b--',alpha=0.6,lw=1.5,
              label=f'fit $\\propto \\Xi^{{{sl:.2f}}}$')
    ax.loglog(xr, 0.9*xr**(-0.5),'k:',alpha=0.4,lw=1.5,label='slope $-0.5$')
ax.set_xlabel('Ξ', fontsize=12)
ax.set_ylabel('Correction factor', fontsize=12)
ax.set_title('Log-log: power-law check (Bar+2018)', fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')

# ── Plot 4: f_r and f_ρ vs fb (semilogx — fb=0 excluded) ─────────
ax=axes2[1,0]
mask_fb = fb_dense > 0
ax.plot(fb_dense[mask_fb], fr_fb[mask_fb],   'o-', color='royalblue', lw=2.5, ms=7, label='$f_r$')
ax.plot(fb_dense[mask_fb], frho_fb[mask_fb], 's-', color='crimson',   lw=2.5, ms=7, label='$f_\\rho$')
ax.axhline(1.0, ls='--', color='gray', alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel('$f_{b,\\rm core}$  (baryon fraction inside core)', fontsize=12)
ax.set_ylabel('Correction factor', fontsize=12)
ax.set_title('Corrections vs baryon fraction (Ξ=0, η=0.5)', fontsize=12)
ax.legend(fontsize=10); ax.grid(alpha=0.3, which='both')

# ── Plot 5: f_r vs η (log-log — eta starts at 0.1, safe) ─────────
ax=axes2[1,1]
eta_dense=np.array([0.10,0.20,0.30,0.50,0.70,1.00,1.50,2.00,3.00,5.00])
for i,fb in enumerate([0.10,0.20,0.30,0.50]):
    fr_eta=[]; frho_eta=[]
    for eta in eta_dense:
        r=solve_pt(0.0,fb,eta)
        fr_eta.append(r['fr'] if r else np.nan)
        frho_eta.append(r['frho'] if r else np.nan)
    col=plt.cm.viridis(i/3)
    ax.plot(eta_dense, fr_eta, 'o-', color=col, lw=2.0, ms=6, label=f'fb={fb}')
ax.axhline(1.0, ls='--', color='gray', alpha=0.5)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('η  (Plummer scale / core radius)', fontsize=12)
ax.set_ylabel('$f_r$', fontsize=12)
ax.set_title('Concentration dependence of $f_r$ (Ξ=0)', fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')

# ── Plot 6: core radius marked on actual profile (log-log) ────────
ax=axes2[1,2]
selected = [0.0, 0.20, 0.50, 1.00]
for i,Xi in enumerate(selected):
    if Xi not in Xi_sols: continue
    r=Xi_sols[Xi]
    x_n=r['x']/xc0
    mask = (x_n >= x_plot_min) & (x_n <= x_plot_max) & (r['rho_norm'] > 0)
    col=c_Xi[Xi_vals.index(Xi)]
    ax.plot(x_n[mask], r['rho_norm'][mask], color=col, lw=2.0, label=f'Ξ={Xi}')
    xc_norm=r['xc']/xc0
    ax.plot(xc_norm, 0.25, 'v', color=col, ms=10, zorder=5)
ax.axhline(0.25, ls='--', color='gray', alpha=0.7, lw=1.2,
           label='$\\rho/\\rho(0) = \\frac{1}{4}$  (core def.)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(x_plot_min, x_plot_max); ax.set_ylim(rho_y_min, rho_y_max)
ax.set_xlabel('$x = r / r_{c0}$', fontsize=12)
ax.set_ylabel('$\\rho(r) / \\rho(0)$', fontsize=12)
ax.set_title('Core radius location (▼) shifts inward', fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('/Users/yiguozhang/step5_corrections.png', dpi=140, bbox_inches='tight')
plt.show()
print("Saved: step5_corrections.png")

# ══════════════════════════════════════════════════════════════════
# FIGURE 3: SELF-GRAVITY POTENTIAL v(x) — how it deforms
# ══════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
fig3.suptitle("Step 5: Self-gravity potential $v(x)$ deformation",
               fontsize=14, fontweight='bold')

# v(x) vs Ξ — y is negative so only log x
ax=axes3[0]
for i,(Xi,r) in enumerate(Xi_sols.items()):
    x_n=r['x']/xc0
    mask=(x_n>=0.05) & (x_n<=8.0)
    ax.plot(x_n[mask], r['v'][mask], color=c_Xi[i], lw=2.0, label=f'Ξ={Xi}')
ax.set_xscale('log')
ax.set_xlim(0.05,8); ax.set_xlabel('$x = r/r_{c0}$',fontsize=12)
ax.set_ylabel('$v(x)$  (dimensionless self-gravity)',fontsize=12)
ax.set_title('Self-gravity $v(x)$ vs SMBH strength',fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')

# v(x) vs fb — y is negative so only log x
ax=axes3[1]
for i,(fb,r) in enumerate(fb_sols.items()):
    x_n=r['x']/xc0
    mask=(x_n>=0.05) & (x_n<=8.0)
    ax.plot(x_n[mask], r['v'][mask], color=c_fb[i], lw=2.0, label=f'fb={fb}')
ax.set_xscale('log')
ax.set_xlim(0.05,8); ax.set_xlabel('$x = r/r_{c0}$',fontsize=12)
ax.set_ylabel('$v(x)$',fontsize=12)
ax.set_title('Self-gravity $v(x)$ vs baryon fraction',fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')

# Total effective potential — y is negative so only log x
ax=axes3[2]
x_p=np.linspace(0.05,8.0,500)
for i,Xi in enumerate([0.0,0.05,0.20,0.50,1.00]):
    if Xi in Xi_sols:
        r=Xi_sols[Xi]
        v_self=np.interp(x_p*xc0, r['x'], r['v'])
    else:
        v_self=np.interp(x_p*xc0, Xi_sols[0.0]['x'], Xi_sols[0.0]['v'])
    v_total = v_self + v_ext(x_p*xc0, Xi, 0.0, 1.0)
    col=c_Xi[Xi_vals.index(Xi)] if Xi in Xi_vals else 'gray'
    ax.plot(x_p, v_total, color=col, lw=2.0, label=f'Ξ={Xi}')
ax.set_xscale('log')
ax.set_xlim(0.05,8); ax.set_ylim(-25,0.5)
ax.set_xlabel('$x = r/r_{c0}$',fontsize=12)
ax.set_ylabel('$v + v_{\\rm ext}$  (total potential)',fontsize=12)
ax.set_title('Total effective potential well',fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('/Users/yiguozhang/step5_potentials.png', dpi=140, bbox_inches='tight')
plt.show()
print("Saved: step5_potentials.png")
print("\nAll three figures done.")
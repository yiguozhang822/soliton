import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 11, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial']})

# ── Load data and fit Chebyshev deg 5 ──
data = np.genfromtxt("data.csv", delimiter=",", skip_header=1)
Xi, fb, eta, f_r, f_rho = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4]
ln_fr = np.log(f_r)
ln_frho = np.log(f_rho)

def cheb(n, x):
    if n == 0: return np.ones_like(x)
    if n == 1: return x.copy()
    a, b = np.ones_like(x), x.copy()
    for _ in range(2, n+1): a, b = b, 2*x*b - a
    return b

def trips(deg):
    return [(i,j,k) for i in range(deg+1) for j in range(deg+1-i) for k in range(deg+1-i-j)]

def design(x1, x2, x3, tr):
    return np.column_stack([cheb(i,x1)*cheb(j,x2)*cheb(k,x3) for i,j,k in tr])

def map_inputs(Xi, fb, eta):
    x1 = 2*Xi - 1
    x2 = 2*fb/0.8 - 1
    x3 = 2*(np.log(eta) - np.log(0.1))/(np.log(5.0) - np.log(0.1)) - 1
    return x1, x2, x3

# Fit
tr5 = trips(5)
x1, x2, x3 = map_inputs(Xi, fb, eta)
A5 = design(x1, x2, x3, tr5)
c_fr = np.linalg.lstsq(A5, ln_fr, rcond=None)[0]
c_frho = np.linalg.lstsq(A5, ln_frho, rcond=None)[0]

def predict(Xi_arr, fb_arr, eta_arr):
    x1, x2, x3 = map_inputs(Xi_arr, fb_arr, eta_arr)
    A = design(x1, x2, x3, tr5)
    return np.exp(A @ c_fr), np.exp(A @ c_frho)

# ── Dense evaluation grid ──
Xi_dense = np.linspace(0, 1, 200)
fb_dense = np.linspace(0, 0.8, 200)

# Fixed eta values for slices
eta_vals = [0.3, 1.0, 3.0]
eta_colors = ['#1b9e77', '#d95f02', '#7570b3']
eta_styles = ['-', '--', ':']

# Fixed fb values for Xi plots
fb_vals = [0.0, 0.1, 0.3, 0.5, 0.8]
fb_colors = ['#333333', '#1b9e77', '#d95f02', '#7570b3', '#e7298a']

# Fixed Xi values for fb plots
Xi_vals = [0.0, 0.1, 0.35, 0.5, 1.0]
Xi_colors = ['#333333', '#1b9e77', '#d95f02', '#7570b3', '#e7298a']


# ════════════════════════════════════════════════════════════════
#  FIGURE 1: Core properties vs. SMBH strength Ξ
# ════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for col, (eta_val, eta_c, eta_ls) in enumerate(zip(eta_vals, eta_colors, eta_styles)):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]

    for fb_val, fb_c in zip(fb_vals, fb_colors):
        Xi_arr = Xi_dense
        fb_arr = np.full_like(Xi_arr, fb_val)
        eta_arr = np.full_like(Xi_arr, eta_val)
        fr_pred, frho_pred = predict(Xi_arr, fb_arr, eta_arr)

        ax_top.plot(Xi_arr, fr_pred, color=fb_c, lw=1.8,
                    label=f'$f_{{b}}$ = {fb_val}')
        ax_bot.plot(Xi_arr, frho_pred, color=fb_c, lw=1.8,
                    label=f'$f_{{b}}$ = {fb_val}')

    ax_top.set_title(f'$\\eta$ = {eta_val}', fontsize=13)
    ax_top.set_ylabel('$f_r = r_c / r_{c0}$', fontsize=12)
    ax_top.set_xlabel('$\\Xi = M_{BH} / M_{sol,0}$', fontsize=11)
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1.05)
    ax_top.grid(True, alpha=0.3)
    if col == 0:
        ax_top.legend(fontsize=8.5, loc='lower left')

    ax_bot.set_ylabel('$f_\\rho = \\rho_c / \\rho_{c0}$', fontsize=12)
    ax_bot.set_xlabel('$\\Xi = M_{BH} / M_{sol,0}$', fontsize=11)
    ax_bot.set_xlim(0, 1)
    ax_bot.set_yscale('log')
    ax_bot.grid(True, alpha=0.3, which='both')
    if col == 0:
        ax_bot.legend(fontsize=8.5, loc='upper left')

fig.suptitle('Core properties vs. SMBH strength $\\Xi$', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('core_vs_smbh.png', dpi=200, bbox_inches='tight')
plt.savefig('core_vs_smbh.pdf', bbox_inches='tight')
print("Saved core_vs_smbh.png and .pdf")


# ════════════════════════════════════════════════════════════════
#  FIGURE 2: Core properties vs. baryon fraction f_b
# ════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for col, (eta_val, eta_c, eta_ls) in enumerate(zip(eta_vals, eta_colors, eta_styles)):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]

    for Xi_val, Xi_c in zip(Xi_vals, Xi_colors):
        fb_arr = fb_dense
        Xi_arr = np.full_like(fb_arr, Xi_val)
        eta_arr = np.full_like(fb_arr, eta_val)
        fr_pred, frho_pred = predict(Xi_arr, fb_arr, eta_arr)

        ax_top.plot(fb_arr, fr_pred, color=Xi_c, lw=1.8,
                    label=f'$\\Xi$ = {Xi_val}')
        ax_bot.plot(fb_arr, frho_pred, color=Xi_c, lw=1.8,
                    label=f'$\\Xi$ = {Xi_val}')

    ax_top.set_title(f'$\\eta$ = {eta_val}', fontsize=13)
    ax_top.set_ylabel('$f_r = r_c / r_{c0}$', fontsize=12)
    ax_top.set_xlabel('$f_{b,core}$', fontsize=11)
    ax_top.set_xlim(0, 0.8)
    ax_top.set_ylim(0, 1.05)
    ax_top.grid(True, alpha=0.3)
    if col == 0:
        ax_top.legend(fontsize=8.5, loc='lower left')

    ax_bot.set_ylabel('$f_\\rho = \\rho_c / \\rho_{c0}$', fontsize=12)
    ax_bot.set_xlabel('$f_{b,core}$', fontsize=11)
    ax_bot.set_xlim(0, 0.8)
    ax_bot.set_yscale('log')
    ax_bot.grid(True, alpha=0.3, which='both')
    if col == 0:
        ax_bot.legend(fontsize=8.5, loc='upper left')

fig.suptitle('Core properties vs. baryon fraction $f_{b,core}$', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('core_vs_baryon.png', dpi=200, bbox_inches='tight')
plt.savefig('core_vs_baryon.pdf', bbox_inches='tight')
print("Saved core_vs_baryon.png and .pdf")
plt.show()
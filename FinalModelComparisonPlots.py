import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 11, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial']})
# ── Load data ──
data = np.genfromtxt("data.csv", delimiter=",", skip_header=1)
Xi, fb, eta, f_r, f_rho = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4]
ln_fr = np.log(f_r)
ln_frho = np.log(f_rho)
N = len(Xi)

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

x1 = 2*Xi - 1
x2 = 2*fb/0.8 - 1
x3 = 2*(np.log(eta) - np.log(0.1))/(np.log(5.0) - np.log(0.1)) - 1

# Method 1: deg 5
A5 = design(x1, x2, x3, trips(5))
pred_fr_d5 = np.exp(A5 @ np.linalg.lstsq(A5, ln_fr, rcond=None)[0])
pred_frho_d5 = np.exp(A5 @ np.linalg.lstsq(A5, ln_frho, rcond=None)[0])

# Method 2: sparse 30
def forward_select(A, y, n):
    sel, rem = [], list(range(A.shape[1]))
    for _ in range(n):
        best_sse, best = np.inf, None
        for c in rem:
            t = sel + [c]
            sse = np.sum((A[:,t] @ np.linalg.lstsq(A[:,t], y, rcond=None)[0] - y)**2)
            if sse < best_sse: best_sse, best = sse, c
        sel.append(best); rem.remove(best)
    return sel

print("Forward selecting...")
s_fr = forward_select(A5, ln_fr, 30)
s_frho = forward_select(A5, ln_frho, 30)
pred_fr_sp = np.exp(A5[:,s_fr] @ np.linalg.lstsq(A5[:,s_fr], ln_fr, rcond=None)[0])
pred_frho_sp = np.exp(A5[:,s_frho] @ np.linalg.lstsq(A5[:,s_frho], ln_frho, rcond=None)[0])

# Sorted errors
def serr(t, p): return np.sort(np.abs((p-t)/t)*100)

err_fr  = [serr(f_r, p) for p in [pred_fr_d5, pred_fr_sp]]
err_frho = [serr(f_rho, p) for p in [pred_frho_d5, pred_frho_sp]]
pct = np.linspace(0, 100, N)

colors = ['#1b9e77', '#7570b3']
styles = ['-', '--']
lws = [2.5, 2.0]
names = ['Chebyshev deg 5  (56 terms)', 'Sparse Chebyshev (30 terms)']

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

fr_offsets  = [(8, -2), (-35, 6)]
frho_offsets = [(8, -2), (-35, -12)]

for panel, (ax, errs_list, title, offsets) in enumerate(zip(
    axes,
    [err_fr, err_frho],
    [r'(a)  Core radius ratio $f_r = r_c\,/\,r_{c0}$',
     r'(b)  Central density ratio $f_\rho = \rho_c\,/\,\rho_{c0}$'],
    [fr_offsets, frho_offsets]
)):
    for errs, name, c, ls, lw, off in zip(errs_list, names, colors, styles, lws, offsets):
        mx = errs[-1]
        label = f'{name}, max {mx:.1f}%'
        ax.plot(pct, errs, color=c, lw=lw, ls=ls, label=label)
        # Endpoint dot
        ax.plot(100, mx, 'o', color=c, ms=8, zorder=5, clip_on=False)
        # Label next to dot
        ax.annotate(f'{mx:.1f}%', xy=(100, mx), xytext=off,
                    textcoords='offset points', fontsize=9, fontweight='bold',
                    color=c, ha='left' if off[0]>0 else 'right',
                    arrowprops=dict(arrowstyle='-', color=c, lw=0.8) if abs(off[1])>8 else None)

    # Reference lines
    if panel == 0:
        for h in [1, 5]:
            ax.axhline(h, color='gray', ls='--', lw=0.7, alpha=0.5)
            ax.text(2, h + 0.05, f'{h}%', fontsize=8, color='gray', alpha=0.6)
        ax.set_ylim(0, 2.5)
    else:
        for h in [5]:
            ax.axhline(h, color='gray', ls='--', lw=0.7, alpha=0.5)
            ax.text(2, h + 0.15, f'{h}%', fontsize=8, color='gray', alpha=0.6)
        ax.set_ylim(0, 7)

    ax.axvline(95, color='gray', ls=':', lw=0.7, alpha=0.4)
    ax.text(95.5, ax.get_ylim()[0] + 0.05, 'p95', fontsize=8, color='gray')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Percentile of 630 grid points', fontsize=12)
    ax.set_ylabel('Relative error  (%)', fontsize=12)
    ax.set_title(title, fontsize=14, pad=10)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.95)

plt.tight_layout(w_pad=3)
plt.savefig('fitting_comparison.png', dpi=200, bbox_inches='tight')
plt.savefig('fitting_comparison.pdf', bbox_inches='tight')
print("Saved fitting_comparison.png and .pdf")
plt.show()
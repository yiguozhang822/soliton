"""
dump_paper_numbers.py
Reads data.csv (columns: Xi, fb_core, eta, f_r, f_rho) and prints every
NUMBER that goes into the paper text and appendix:
  - Table A.1 (degree-5, 56 terms, c_fr and c_frho)
  - Table A.2 (sparse f_r, 30 terms, selection order)
  - Table A.3 (sparse f_rho, 30 terms, selection order)
  - Section 3 accuracy numbers (max and 95th-percentile relative error)
  - Section 2.7 / Figure 3 power-law exponent
Fit logic copied verbatim from fittingformulas.py / FinalModelComparisonPlots.py.
"""
import numpy as np

data = np.genfromtxt("data.csv", delimiter=",", skip_header=1)
Xi, fb, eta, f_r, f_rho = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4]
ln_fr, ln_frho = np.log(f_r), np.log(f_rho)

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
tr5 = trips(5)
A5 = design(x1, x2, x3, tr5)

c_fr   = np.linalg.lstsq(A5, ln_fr,   rcond=None)[0]
c_frho = np.linalg.lstsq(A5, ln_frho, rcond=None)[0]

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

s_fr   = forward_select(A5, ln_fr,   30)
s_frho = forward_select(A5, ln_frho, 30)
csp_fr   = np.linalg.lstsq(A5[:,s_fr],   ln_fr,   rcond=None)[0]
csp_frho = np.linalg.lstsq(A5[:,s_frho], ln_frho, rcond=None)[0]

print("="*64)
print("TABLE A.1  (degree-5, 56 terms)")
print(f"{'n':>3} {'i':>2} {'j':>2} {'k':>2} {'Coefficient c_fr':>18} {'Coefficient c_frho':>18}")
for n,(i,j,k) in enumerate(tr5, 1):
    print(f"{n:>3} {i:>2} {j:>2} {k:>2} {c_fr[n-1]:>+18.6e} {c_frho[n-1]:>+18.6e}")

print("\n"+"="*64)
print("TABLE A.2  (sparse f_r, 30 terms, selection order)")
print(f"{'n':>3} {'i':>2} {'j':>2} {'k':>2} {'Coefficient c_fr':>18}")
for n,(col,coef) in enumerate(zip(s_fr, csp_fr), 1):
    i,j,k = tr5[col]
    print(f"{n:>3} {i:>2} {j:>2} {k:>2} {coef:>+18.6e}")

print("\n"+"="*64)
print("TABLE A.3  (sparse f_rho, 30 terms, selection order)")
print(f"{'n':>3} {'i':>2} {'j':>2} {'k':>2} {'Coefficient c_frho':>18}")
for n,(col,coef) in enumerate(zip(s_frho, csp_frho), 1):
    i,j,k = tr5[col]
    print(f"{n:>3} {i:>2} {j:>2} {k:>2} {coef:>+18.6e}")

print("\n"+"="*64)
print("SECTION 3 ACCURACY NUMBERS  (max %, 95th-percentile %)")
def err(true, pred): return np.abs((pred-true)/true)*100.0
for name, t, p in [
    ("f_r   deg-5",  f_r,   np.exp(A5 @ c_fr)),
    ("f_r   sparse", f_r,   np.exp(A5[:,s_fr]   @ csp_fr)),
    ("f_rho deg-5",  f_rho, np.exp(A5 @ c_frho)),
    ("f_rho sparse", f_rho, np.exp(A5[:,s_frho] @ csp_frho)),
]:
    e = err(t,p)
    print(f"  {name:<13}: max {np.max(e):6.3f}%   p95 {np.percentile(e,95):6.3f}%")

print("\n"+"="*64)
print("SECTION 2.7 / FIGURE 3 POWER LAW")
mask = (f_r > 0.2) & (f_r < 1.0) & (f_rho > 1.0)
slope, intc = np.polyfit(np.log(f_r[mask]), np.log(f_rho[mask]), 1)
print(f"  f_rho proportional to f_r^{slope:.2f}   (paper currently says -2.35)")
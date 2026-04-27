"""
Numerical Operation Counter
============================
Computes FLOP counts for:
  1. Full Chebyshev expansion (degree 5, 3 variables)
  2. Sparse Chebyshev expansion (degree 5, 3 variables, N retained terms)
  3. BVP solve via scipy.solve_bvp
  
All counts are computed numerically, not hardcoded.
"""
import numpy as np
from itertools import product as iter_product


# ============================================================
# CONFIGURABLE PARAMETERS
# ============================================================
N_VARS = 3              # input variables: Xi, f_b_core, eta
DEGREE = 5              # max Chebyshev degree per variable
N_SPARSE_TERMS = 30     # retained terms in sparse expansion (per output)
N_OUTPUTS = 2           # f_r and f_rho

# BVP parameters
N_GRID = 2500           # collocation grid points
N_EQ = 4                # number of first-order ODEs
N_ITER_LOW = 5          # best-case iteration count
N_ITER_MID = 10         # typical iteration count
N_ITER_HIGH = 15        # worst-case iteration count


# ============================================================
# HELPER: count operations in each stage
# ============================================================
def count_affine_mapping(n_vars):
    """Map each input to [-1, 1]: x_mapped = a*x + b"""
    mults = n_vars       # one multiply per variable
    adds = n_vars        # one addition per variable
    return {"mults": mults, "adds": adds, "total": mults + adds}


def count_chebyshev_recurrence(n_vars, degree):
    """
    Build T_0(x), T_1(x), ..., T_d(x) for each variable.
    T_0 = 1 (free), T_1 = x (free).
    T_k = 2x * T_{k-1} - T_{k-2} for k >= 2.
    Precompute 2x once per variable.
    """
    precompute_2x = n_vars                          # one mult per var
    n_steps = max(degree - 1, 0)                    # T_2 through T_degree
    recurrence_mults = n_steps * n_vars             # (2x)*T_{k-1}
    recurrence_subs = n_steps * n_vars              # - T_{k-2}
    
    total_mults = precompute_2x + recurrence_mults
    total_adds = recurrence_subs                    # subtraction = addition
    return {
        "mults": total_mults,
        "adds": total_adds,
        "total": total_mults + total_adds,
        "n_basis_per_var": degree + 1               # T_0 ... T_degree
    }


def count_full_expansion(n_vars, degree, n_outputs):
    """
    Full tensor-product Chebyshev: sum over ALL (d+1)^n_vars terms.
    Each term: c_{ijk} * T_i(x1) * T_j(x2) * T_k(x3)
      = (n_vars - 1) mults for the T product + 1 mult for coefficient
      = n_vars multiplications per term
    Accumulation: (n_terms - 1) additions per output.
    """
    n_terms = (degree + 1) ** n_vars
    mults_per_term = n_vars                         # T_i*T_j, *T_k, *c
    adds_per_output = n_terms - 1                   # accumulate sum
    
    total_mults = n_terms * mults_per_term * n_outputs
    total_adds = adds_per_output * n_outputs
    return {
        "n_terms": n_terms,
        "mults": total_mults,
        "adds": total_adds,
        "total": total_mults + total_adds
    }


def count_sparse_expansion(n_vars, n_sparse_terms, n_outputs):
    """
    Sparse Chebyshev: only N retained terms per output.
    Same cost per term as full, just fewer terms.
    Basis values T_i(x_k) are precomputed and reused.
    """
    mults_per_term = n_vars                         # T_i*T_j, *T_k, *c
    adds_per_output = n_sparse_terms - 1
    
    total_mults = n_sparse_terms * mults_per_term * n_outputs
    total_adds = adds_per_output * n_outputs
    return {
        "n_terms": n_sparse_terms,
        "mults": total_mults,
        "adds": total_adds,
        "total": total_mults + total_adds
    }


def count_rhs_eval(n_grid):
    """
    Count FLOPs for one full RHS evaluation of the 4-ODE system
    at all grid points.
    
    System (dimensionless):
      y1' = y2                             -> 0 ops
      y2' = 2*(y3 + v_ext - eps) * y1      -> see below
      y3' = y4                             -> 0 ops
      y4' = y1^2/x^2 - (2/x)*y4           -> see below
      
    v_ext(x) = -Xi/x - f_b/sqrt(x^2 + eta^2):
      Xi/x          : 1 div
      x^2           : 1 mult
      x^2 + eta^2   : 1 add
      sqrt(...)      : 1 op (count as 1 FLOP)
      f_b / sqrt    : 1 div
      sum + negate   : 1 add
      v_ext total    : 6 ops
      
    y2' = 2*(y3 + v_ext - eps)*y1:
      y3 + v_ext    : 1 add
      - eps         : 1 sub
      * 2           : 1 mult
      * y1          : 1 mult
      y2' total     : 4 ops
      
    y4' = y1^2/x^2 - (2/x)*y4:
      y1^2          : 1 mult
      / x^2         : 1 div  (reuse x^2 from v_ext)
      2/x           : 1 div  (reuse 1/x from v_ext)
      * y4          : 1 mult
      subtraction   : 1 sub
      y4' total     : 5 ops (with reuse of cached x^2 and 1/x)
    """
    v_ext_per_point = 6
    y2_prime_per_point = 4
    y4_prime_per_point = 5  # with caching of x^2 and 1/x
    
    ops_per_point = v_ext_per_point + y2_prime_per_point + y4_prime_per_point
    total = n_grid * ops_per_point
    return {
        "ops_per_point": ops_per_point,
        "v_ext": v_ext_per_point,
        "y2_prime": y2_prime_per_point,
        "y4_prime": y4_prime_per_point,
        "total": total
    }


def count_jacobian_construction(n_eq, rhs_total):
    """
    scipy.solve_bvp computes the Jacobian via finite differences.
    For each of the n_eq variables, it perturbs and re-evaluates the RHS.
    Cost: n_eq full RHS evaluations + the difference divisions.
    """
    # n_eq perturbation evaluations
    perturbation_evals = n_eq * rhs_total
    # Finite difference: (f(x+h) - f(x)) / h for each component at each point
    # n_eq * n_eq * n_grid divisions + subtractions, but this is smaller than
    # the RHS evals, so approximate as 2 * n_eq * n_eq * n_grid
    n_grid = rhs_total // 15  # recover n_grid from total
    fd_overhead = 2 * n_eq * n_eq * n_grid
    return {
        "perturbation_evals": perturbation_evals,
        "fd_overhead": fd_overhead,
        "total": perturbation_evals + fd_overhead
    }


def count_banded_linear_solve(n_eq, n_grid, bandwidth):
    """
    Banded LU factorization + back-substitution.
    System size n = n_eq * n_grid, bandwidth b.
    LU: O(n * b^2), back-sub: O(n * b).
    """
    n = n_eq * n_grid
    lu = n * bandwidth ** 2
    backsub = n * bandwidth
    return {
        "system_size": n,
        "bandwidth": bandwidth,
        "lu_factorization": lu,
        "back_substitution": backsub,
        "total": lu + backsub
    }


# ============================================================
# COMPUTE ALL COUNTS
# ============================================================

# --- Shared: affine mapping + recurrence ---
mapping = count_affine_mapping(N_VARS)
recurrence = count_chebyshev_recurrence(N_VARS, DEGREE)
shared_ops = mapping["total"] + recurrence["total"]

# --- Full Chebyshev expansion ---
full_eval = count_full_expansion(N_VARS, DEGREE, N_OUTPUTS)
full_total = shared_ops + full_eval["total"]

# --- Sparse Chebyshev expansion ---
sparse_eval = count_sparse_expansion(N_VARS, N_SPARSE_TERMS, N_OUTPUTS)
sparse_total = shared_ops + sparse_eval["total"]

# --- BVP solve ---
rhs = count_rhs_eval(N_GRID)
bandwidth = 2 * N_EQ  # collocation stencil for 4-eq system
jacobian = count_jacobian_construction(N_EQ, rhs["total"])
linear = count_banded_linear_solve(N_EQ, N_GRID, bandwidth)

per_iteration = rhs["total"] + jacobian["total"] + linear["total"]
bvp_low = N_ITER_LOW * per_iteration
bvp_mid = N_ITER_MID * per_iteration
bvp_high = N_ITER_HIGH * per_iteration


# ============================================================
# PRINT RESULTS
# ============================================================
print("=" * 70)
print("  OPERATION COUNT RESULTS")
print("=" * 70)

print(f"\n{'PARAMETERS':}")
print(f"  Variables: {N_VARS}   Degree: {DEGREE}   Sparse terms: {N_SPARSE_TERMS}")
print(f"  BVP grid: {N_GRID}   Equations: {N_EQ}   Bandwidth: {bandwidth}")
print(f"  Iterations: {N_ITER_LOW}-{N_ITER_HIGH} (typical {N_ITER_MID})")

# --- Chebyshev breakdown ---
print(f"\n{'─' * 70}")
print(f"  FULL CHEBYSHEV EXPANSION (degree {DEGREE}, {N_VARS} vars)")
print(f"{'─' * 70}")
print(f"  Affine mapping:       {mapping['total']:>8d} ops  ({N_VARS} mults + {N_VARS} adds)")
print(f"  Recurrence T_0..T_5:  {recurrence['total']:>8d} ops  ({recurrence['mults']} mults + {recurrence['adds']} subs)")
n_full = full_eval['n_terms']
print(f"  Term evaluation:      {full_eval['total']:>8d} ops  ({n_full} terms x {N_VARS} mults x {N_OUTPUTS} outputs + {full_eval['adds']} adds)")
print(f"  ────────────────────────────────")
print(f"  TOTAL:                {full_total:>8d} ops")

print(f"\n{'─' * 70}")
print(f"  SPARSE CHEBYSHEV EXPANSION ({N_SPARSE_TERMS} terms per output)")
print(f"{'─' * 70}")
print(f"  Affine mapping:       {mapping['total']:>8d} ops")
print(f"  Recurrence T_0..T_5:  {recurrence['total']:>8d} ops")
print(f"  Term evaluation:      {sparse_eval['total']:>8d} ops  ({N_SPARSE_TERMS} terms x {N_VARS} mults x {N_OUTPUTS} outputs + {sparse_eval['adds']} adds)")
print(f"  ────────────────────────────────")
print(f"  TOTAL:                {sparse_total:>8d} ops")

print(f"\n  Full vs Sparse savings: {full_total} -> {sparse_total}"
      f"  ({full_total / sparse_total:.1f}x reduction)")

# --- BVP breakdown ---
print(f"\n{'─' * 70}")
print(f"  BVP SOLVE (scipy.solve_bvp)")
print(f"{'─' * 70}")
print(f"  Per grid point RHS:   {rhs['ops_per_point']:>8d} ops  (v_ext:{rhs['v_ext']} + y2':{rhs['y2_prime']} + y4':{rhs['y4_prime']})")
print(f"  Full RHS evaluation:  {rhs['total']:>8,d} ops  ({N_GRID} x {rhs['ops_per_point']})")
print(f"  Jacobian (finite diff):{jacobian['total']:>8,d} ops  ({N_EQ} perturbed RHS evals + overhead)")
print(f"  Banded linear solve:  {linear['total']:>8,d} ops  (LU:{linear['lu_factorization']:,} + backsub:{linear['back_substitution']:,})")
print(f"  ────────────────────────────────")
print(f"  Per iteration:        {per_iteration:>10,d} ops")
print(f"  Total ({N_ITER_LOW} iters):      {bvp_low:>10,d} ops")
print(f"  Total ({N_ITER_MID} iters):     {bvp_mid:>10,d} ops")
print(f"  Total ({N_ITER_HIGH} iters):     {bvp_high:>10,d} ops")

# --- Speedup ratios ---
print(f"\n{'─' * 70}")
print(f"  SPEEDUP RATIOS (BVP / Chebyshev)")
print(f"{'─' * 70}")
print(f"  {'':30s} {'Full Cheb':>12s} {'Sparse Cheb':>12s}")
print(f"  {'BVP (' + str(N_ITER_LOW) + ' iter)':30s} {bvp_low/full_total:>12,.0f}x {bvp_low/sparse_total:>12,.0f}x")
print(f"  {'BVP (' + str(N_ITER_MID) + ' iter, typical)':30s} {bvp_mid/full_total:>12,.0f}x {bvp_mid/sparse_total:>12,.0f}x")
print(f"  {'BVP (' + str(N_ITER_HIGH) + ' iter)':30s} {bvp_high/full_total:>12,.0f}x {bvp_high/sparse_total:>12,.0f}x")

print(f"\n{'─' * 70}")
print(f"  ORDER OF MAGNITUDE SUMMARY")
print(f"{'─' * 70}")
print(f"  Full Chebyshev:    ~{full_total} ops     (O(10^{np.log10(full_total):.1f}))")
print(f"  Sparse Chebyshev:  ~{sparse_total} ops      (O(10^{np.log10(sparse_total):.1f}))")
print(f"  BVP solve:         ~{bvp_mid:,} ops  (O(10^{np.log10(bvp_mid):.1f}))")
print(f"  Safe claim: ~{int(round(np.log10(bvp_mid/sparse_total)))} orders of magnitude speedup (sparse)")
print(f"              ~{int(round(np.log10(bvp_mid/full_total)))} orders of magnitude speedup (full)")

# --- Sonnet comparison ---
print(f"\n{'─' * 70}")
print(f"  SONNET 4.6 COMPARISON")
print(f"{'─' * 70}")
sonnet_cheb = 290
sonnet_bvp = 6_600_000
print(f"  {'':25s} {'Sonnet':>12s} {'This script':>12s} {'Difference':>12s}")
print(f"  {'Chebyshev (sparse)':25s} {sonnet_cheb:>12,} {sparse_total:>12,} {(sparse_total-sonnet_cheb)/sonnet_cheb*100:>+11.1f}%")
print(f"  {'BVP (10 iter)':25s} {sonnet_bvp:>12,} {bvp_mid:>12,} {(bvp_mid-sonnet_bvp)/sonnet_bvp*100:>+11.1f}%")
print(f"  {'Ratio':25s} {sonnet_bvp//sonnet_cheb:>12,}x {bvp_mid//sparse_total:>12,}x")
print(f"\n  Sonnet undercounted BVP by ~{(bvp_mid-sonnet_bvp)/sonnet_bvp*100:.0f}%:")
print(f"    - Missed v_ext evaluation (~6 ops/point)")
print(f"    - Missed Jacobian construction via finite differences")
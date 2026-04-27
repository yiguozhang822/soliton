import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# ============================================================
# Raw isolated Schrödinger-Poisson solver in the SAME display
# style as Chan, Sibiryakov, Xue (standard soliton style):
#
#   chi(0) = 1
#   chi(r -> infinity) = 0
#   Phi(r -> infinity) = 0
#
# with:
#   s'' = 2 (v - eps) s
#   v'' + (2/x) v' = s^2 / x^2
#
# and chi(x) = s(x) / x
#
# IMPORTANT:
# This script does NOT rescale by half-density radius.
# It uses the RAW solution, because that is the correct
# comparison for the paper-style "standard soliton" plot.
# ============================================================

X_EPS = 1e-6
X_MAX = 200.0


def odes(x, y, p):
    eps = p[0]
    s, sp, v, vp = y

    ds = sp
    dsp = 2.0 * (v - eps) * s
    dv = vp
    dvp = (s * s) / (x * x) - (2.0 / x) * vp

    return np.vstack((ds, dsp, dv, dvp))


def bc(ya, yb, p):
    return np.array([
        ya[0] - X_EPS,   # s ~ x near origin
        ya[1] - 1.0,     # fixes normalization so chi(0)=1
        ya[3] - 0.0,     # v'(0)=0
        yb[0] - 0.0,     # s(infty)=0
        yb[2] - 0.0      # v(infty)=0 gauge choice
    ])


def compute_chi(x, s):
    # chi = psi = s/x
    return s / x


def compute_mass_mu0(x, chi):
    # In these dimensionless units:
    # mu0 = 4*pi * integral r^2 chi(r)^2 dr
    return 4.0 * np.pi * np.trapezoid(x * x * chi * chi, x)


def find_half_density_radius(x, chi):
    rho = chi * chi
    rho0 = rho[0]
    target = 0.5 * rho0

    idx = np.where(rho <= target)[0]
    if len(idx) == 0:
        return None

    i = int(idx[0])
    if i == 0:
        return x[0]

    x1, x2 = x[i - 1], x[i]
    r1, r2 = rho[i - 1], rho[i]

    if r2 == r1:
        return x2

    t = (target - r1) / (r2 - r1)
    return x1 + t * (x2 - x1)


def main():
    # geometric grid helps near the origin
    x = np.geomspace(X_EPS, X_MAX, 2500)

    # initial guess
    s_guess = x * np.exp(-x)
    sp_guess = (1.0 - x) * np.exp(-x)

    v_guess = -1.0 / (1.0 + x)
    vp_guess = 1.0 / (1.0 + x) ** 2

    y_guess = np.vstack((s_guess, sp_guess, v_guess, vp_guess))
    p_guess = np.array([-1.0])

    sol = solve_bvp(
        odes,
        bc,
        x,
        y_guess,
        p=p_guess,
        tol=1e-6,
        max_nodes=200000
    )

    if sol.status != 0:
        raise RuntimeError(
            "solve_bvp did not converge. Try larger X_MAX or more nodes."
        )

    eps_star = float(sol.p[0])
    x_sol = sol.x
    s_sol = sol.y[0]
    v_sol = sol.y[2]

    chi_sol = compute_chi(x_sol, s_sol)

    # Diagnostics
    mu0 = compute_mass_mu0(x_sol, chi_sol)
    r_half = find_half_density_radius(x_sol, chi_sol)
    tail_metric = abs(chi_sol[-1]) / (np.max(np.abs(chi_sol)) + 1e-300)

    print("Converged.")
    print("eps_star =", eps_star)
    print("mu0 =", mu0)
    print("Phi(0) =", v_sol[0])
    print("half-density radius =", r_half)
    print("tail metric =", tail_metric)

    # prepend exact origin values for prettier plotting
    r_chi_plot = np.insert(x_sol, 0, 0.0)
    chi_plot = np.insert(chi_sol, 0, 1.0)

    r_phi_plot = np.insert(x_sol, 0, 0.0)
    phi_plot = np.insert(v_sol, 0, v_sol[0])

    # ------------------------------------------------------------
    # Figure A: paper-style reproduction of Chan et al. Figure 1
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax1, ax2 = axes

    # Left panel: chi_0(r)
    ax1.plot(r_chi_plot, chi_plot, linewidth=2)
    ax1.set_xlim(0, 8)
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel(r"$r$")
    ax1.set_ylabel(r"$\chi_0$")
    ax1.tick_params(direction="in", top=True, right=True)

    # Right panel: Phi_0(r)
    ax2.plot(r_phi_plot, phi_plot, linewidth=2)
    ax2.set_xlim(0, 25)
    ax2.set_ylim(-1.4, 0.0)
    ax2.set_xlabel(r"$r$")
    ax2.set_ylabel(r"$\Phi_0$")
    ax2.tick_params(direction="in", top=True, right=True)

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Figure B: same raw solution, but with teacher-friendly labels
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1, ax2 = axes

    ax1.plot(r_chi_plot, chi_plot, linewidth=2, label="Our numerical standard soliton")
    ax1.set_xlim(0, 8)
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel("r")
    ax1.set_ylabel(r"$\chi(r)$")
    ax1.set_title(r"Wavefunction-like profile $\chi(r)=s(r)/r$")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend()

    ax2.plot(r_phi_plot, phi_plot, linewidth=2, label="Our numerical potential")
    ax2.set_xlim(0, 25)
    ax2.set_ylim(-1.4, 0.0)
    ax2.set_xlabel("r")
    ax2.set_ylabel(r"$\Phi(r)$")
    ax2.set_title("Self-gravitational potential")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Figure C: optional density plot for your own records
    # ------------------------------------------------------------
    rho_norm = (chi_sol * chi_sol) / (chi_sol[0] * chi_sol[0])

    plt.figure(figsize=(7, 5))
    plt.plot(x_sol, rho_norm, linewidth=2)
    plt.axhline(0.5, linestyle=":", label="half-density")
    if r_half is not None:
        plt.axvline(r_half, linestyle=":", label="r_half")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("r")
    plt.ylabel(r"$\rho(r)/\rho(0)$")
    plt.title("Same raw solution shown as normalized density")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
# Overview
Below are numerical code accompanying the paper *"Portable Correction Framework for ULDM Soliton Cores Deformed by Black Holes and Baryonic Gravitational Potentials."*

Solves the dimensionless Schrödinger-Poisson system for a ULDM soliton under a combined SMBH + Plummer baryon external potential, builds a 630-point deformation library over (Ξ, f_{b,core}, η), and fits a Chebyshev correction map (f_r, f_ρ) accurate to ≤0.66% and ≤3.35% respectively at ~5,600× lower cost than a full BVP solve.

## Requirements

```bash
pip install numpy scipy matplotlib pandas
```

## Files

| Script | Purpose |
|---|---|
| `IsolatedSolitonBaseline.py` | Solve isolated SP system; report ε, x_core, ρ₀ |
| `soliton1.py` | Alternative isolated solver in the Chan–Sibiryakov–Xue convention |
| `SolitonValidationSchive2014.py` | Validate numerical profile against Schive+2014 analytic formula |
| `SolitonModified.py` | SP solver with SMBH + Plummer external potential |
| `ModifiedSolitonPlots.py` | Density profile and correction-factor figures for Step 5 |
| `step6_generate_library.py` | Initial parameter sweep on a coarser grid |
| `3Dplot630points.py` | Full 630-point sweep with path continuation |
| `3DplotDiagnostic.py` | Diagnose anomalous points by profile shape and node count |
| `VisualizeLibrary.py` | Five library figures: 3D scatter, surface slices, heatmaps, power-law, f_r×f_ρ scatter |
| `fittingformulas.py` | Compare four fitting strategies (Chebyshev deg 4, deg 5, sparse-30, log-polynomial) |
| `FinalModelComparisonPlots.py` | Final deg-5 vs sparse-30 comparison used in the paper |
| `out_of_sample_validation.py` | Held-out validation: solve the BVP at off-grid (Ξ, f_{b,core}, η) triples and compare the deg-5 and sparse-30 maps against ground-state truth |
| `dump_paper_numbers.py` | Regenerate every paper number from `data.csv`: Table A.1 to A.3 coefficients, Section 3 accuracy figures, and the Figure 3 power-law exponent |
| `resultplots.py` | Prediction-surface plots for f_r and f_ρ |
| `operationCounts.py` | FLOP counter reproducing the speedup analysis |

The library is saved as `step6_continuation_library.csv` / `.npz`. Fitting scripts expect a file named `data.csv` with columns `Xi, fb_core, eta, f_r, f_rho`. `out_of_sample_validation.py` writes `out_of_sample_library.csv` (the held-out truth set) along with the residual figures `out_of_sample_residuals.png` and `out_of_sample_vs_in_sample.png`.

## Acknowledgements

Claude (Anthropic) was used to assist in developing and debugging the numerical code. All results were verified by the author.

## License

MIT, see [LICENSE](LICENSE).

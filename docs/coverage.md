# Coverage against SciPy and MATLAB

Inventory of what `go-statistics/stats` (main) implements against
`scipy.stats` v1.18.1 (distributions, tests, descriptive statistics, QMC,
resampling, contingency, sampling) and the statistical part of MATLAB's
Statistics and Machine Learning Toolbox (machine-learning functions are out of
scope and not listed). Status: **done** (same semantics and options that
matter), **partial** (what is missing is named), **missing**. "Dep" is what a
missing item needs beyond this package.

Dep vocabulary. *closed* is a closed-form density, cdf and quantile over
functions that exist in `go-numerics/special` today (`Ndtr`, `GammaP`,
`BetaInc`, `Erfcx`, `Kv`...): no new dependency, only work. A named function
(`hyp2f1`, `owens_t`, `expn`...) is one `go-numerics/special` lacks (see its
`docs/coverage.md`). *integrate* is numerical quadrature from
`go-numerics/integrate` (what SciPy's generic `rv_continuous` falls back to);
*optimize* is a root finder or fitter from `go-numerics/optimize`.

Sources. SciPy: `scipy/stats/__init__.py`, `qmc.py`, `contingency.py`,
`sampling.py` and `mstats.py` at tag v1.18.1 (every name in their autosummary
lists appears below, in order). MATLAB: the MathWorks function lists could not
be read by the tool used (the pages are rendered client-side), so the MATLAB
column is built from the toolbox's documented function families and
distribution names and must be re-checked against the MathWorks reference
lists before a slice is started. Go side: README, `go doc`, `docs/design.md`
and the sources on main.

Row numbers are identifiers, not counts. A row groups SciPy names only where
they are one algorithm (for instance the `binned_statistic` trio); every
distribution has its own row.

## Totals

Counted by `go run` over the tables below (rows, not names).

| | done | partial | missing | rows |
|---|---|---|---|---|
| SciPy v1.18.1 (sections 1-12) | 69 | 8 | 171 | 248 |
| MATLAB-only (section 13) | 0 | 0 | 13 | 13 |

## 1. Continuous distributions

Done means pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf, moments, entropy,
`rvs` and `fit` (closed form or maximum likelihood), measured against mpmath.

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 1 | `alpha` | | | missing | closed |
| 2 | `anglit` | | | missing | closed |
| 3 | `arcsine` | `makedist('Beta',0.5,0.5)` | | missing | closed (beta special case) |
| 4 | `argus` | | | missing | closed (`GammaP`) |
| 5 | `beta` | `betapdf`, `Beta` | `Beta`, `FitBeta` | done | |
| 6 | `betaprime` | | | missing | closed (`BetaInc`) |
| 7 | `bradford` | | | missing | closed |
| 8 | `burr` | | | missing | closed |
| 9 | `burr12` | `Burr` | | missing | closed |
| 10 | `cauchy` | | `Cauchy`, `FitCauchy` | done | |
| 11 | `chi` | | `Chi`, `FitChi` | done | |
| 12 | `chi2` | `chi2pdf` | `ChiSquared`, `FitChiSquared` | done | |
| 13 | `cosine` | | | missing | closed |
| 14 | `crystalball` | | | missing | closed (`Ndtr`, `Erfc`) |
| 15 | `dgamma` | | | missing | closed (`GammaP`) |
| 16 | `dpareto_lognorm` | | | missing | closed (`Ndtr`, `Erfcx`) |
| 17 | `dweibull` | | | missing | closed |
| 18 | `erlang` | `gampdf` (integer shape) | `Gamma` with integer A | done | alias; no constructor that enforces integer shape |
| 19 | `expon` | `exppdf`, `Exponential` | `Exponential`, `FitExponential` | done | |
| 20 | `exponnorm` | | | missing | closed (`Erfcx`) |
| 21 | `exponweib` | | | missing | closed |
| 22 | `exponpow` | | | missing | closed (`GammaQ`) |
| 23 | `f` | `fpdf` | `F`, `FitF` | done | |
| 24 | `fatiguelife` | `BirnbaumSaunders` | | missing | closed |
| 25 | `fisk` | `Loglogistic` | | missing | closed |
| 26 | `foldcauchy` | | | missing | closed |
| 27 | `foldnorm` | | | missing | closed |
| 28 | `genlogistic` | | | missing | closed |
| 29 | `gennorm` | | | missing | closed (`GammaP`) |
| 30 | `genpareto` | `GeneralizedPareto`, `gppdf` | | missing | closed |
| 31 | `genexpon` | | | missing | closed |
| 32 | `genextreme` | `GeneralizedExtremeValue`, `gevpdf` | | missing | closed (sign convention of shape differs) |
| 33 | `gausshyper` | | | missing | `hyp2f1`; integrate for cdf |
| 34 | `gamma` | `gampdf`, `Gamma` | `Gamma`, `FitGamma` | done | |
| 35 | `gengamma` | | | missing | closed (`GammaP`) |
| 36 | `genhalflogistic` | | | missing | closed |
| 37 | `genhyperbolic` | | | missing | `Kv` (have); integrate for cdf |
| 38 | `geninvgauss` | | | missing | `Kve` (have); integrate for cdf |
| 39 | `gibrat` | | | missing | closed |
| 40 | `gompertz` | | | missing | `exp1` (special) |
| 41 | `gumbel_r` | `evpdf` mirrored | `Gumbel`, `FitGumbel` | done | |
| 42 | `gumbel_l` | `ExtremeValue`, `evpdf` | | missing | closed (mirror of `Gumbel`) |
| 43 | `halfcauchy` | | | missing | closed |
| 44 | `halflogistic` | | | missing | closed |
| 45 | `halfnorm` | `HalfNormal` | `HalfNormal`, `FitHalfNormal` | done | |
| 46 | `halfgennorm` | | | missing | closed (`GammaP`) |
| 47 | `hypsecant` | | | missing | closed |
| 48 | `invgamma` | | `InverseGamma`, `FitInverseGamma` | done | |
| 49 | `invgauss` | `InverseGaussian` | | missing | closed (`Ndtr`) |
| 50 | `invweibull` | | | missing | closed |
| 51 | `irwinhall` | | | missing | closed (finite alternating sums) |
| 52 | `jf_skew_t` | | | missing | closed (`BetaInc`) |
| 53 | `johnsonsb` | `johnsrnd` (Johnson system) | | missing | closed |
| 54 | `johnsonsu` | `johnsrnd` | | missing | closed |
| 55 | `kappa4` | | | missing | closed |
| 56 | `kappa3` | | | missing | closed |
| 57 | `ksone` | | | missing | Smirnov functions (`special` `smirnov`; code exists privately in `ksdist.go`) |
| 58 | `kstwo` | | | missing | same |
| 59 | `kstwobign` | | | missing | `kolmogorov` (same file) |
| 60 | `landau` | | | missing | rational approximations (SciPy 1.15 port) |
| 61 | `laplace` | | `Laplace`, `FitLaplace` | done | |
| 62 | `laplace_asymmetric` | | | missing | closed |
| 63 | `levy` | | | missing | closed (`Erfc`) |
| 64 | `levy_l` | | | missing | closed |
| 65 | `levy_stable` | `Stable` | | missing | integrate (Nolan's parameterisations); `Gamma`, `Cos` helpers |
| 66 | `logistic` | `Logistic` | `Logistic`, `FitLogistic` | done | |
| 67 | `loggamma` | | | missing | closed |
| 68 | `loglaplace` | | | missing | closed |
| 69 | `lognorm` | `lognpdf`, `Lognormal` | `LogNormal`, `FitLogNormal` | done | |
| 70 | `loguniform` | `Loguniform` | `Reciprocal` (alias `LogUniform`) | done | |
| 71 | `lomax` | | `Lomax`, `FitLomax` | done | |
| 72 | `maxwell` | | | missing | closed (`GammaP`) |
| 73 | `mielke` | | | missing | closed |
| 74 | `moyal` | | | missing | closed (`Erfc`) |
| 75 | `nakagami` | `Nakagami` | | missing | closed (`GammaP`) |
| 76 | `ncx2` | `ncx2pdf` | `NoncentralChiSquared` | partial | no `FitNoncentralChiSquared` |
| 77 | `ncf` | `ncfpdf` | | missing | non-central beta (`ncfdtr`; special slice) |
| 78 | `nct` | `nctpdf` | | missing | `nctdtr` (special slice) |
| 79 | `norm` | `normpdf`, `Normal` | `Normal`, `FitNormal` | done | |
| 80 | `norminvgauss` | | | missing | `Kve`; integrate for cdf |
| 81 | `pareto` | | `Pareto`, `FitPareto` | done | |
| 82 | `pearson3` | `pearsrnd` (Pearson system) | | missing | closed (`GammaP`) |
| 83 | `powerlaw` | | | missing | closed |
| 84 | `powerlognorm` | | | missing | `Ndtr`; integrate for ppf |
| 85 | `powernorm` | | | missing | closed |
| 86 | `rdist` | | | missing | closed (`BetaInc`) |
| 87 | `rayleigh` | `Rayleigh` | `Rayleigh`, `FitRayleigh` | done | |
| 88 | `rel_breitwigner` | | | missing | integrate (cdf by quadrature in SciPy) |
| 89 | `rice` | `Rician` | | missing | Marcum Q (`NoncentralChiSquared` cdf exists) |
| 90 | `recipinvgauss` | | | missing | closed |
| 91 | `semicircular` | | | missing | closed |
| 92 | `skewcauchy` | | | missing | closed |
| 93 | `skewnorm` | | | missing | `owens_t` (special slice) |
| 94 | `studentized_range` | | | missing | integrate (Copenhaver-Holland); `qsturng` |
| 95 | `t` | `tpdf`, `tLocationScale` | `StudentT`, `FitStudentT` | done | |
| 96 | `trapezoid` | | | missing | closed |
| 97 | `triang` | `Triangular` | `Triangular`, `FitTriangular` | done | |
| 98 | `truncexpon` | | | missing | closed |
| 99 | `truncnorm` | | | missing | closed (`LogNdtr`) |
| 100 | `truncpareto` | | | missing | closed |
| 101 | `truncweibull_min` | | | missing | closed |
| 102 | `tukeylambda` | | | missing | `tklmbda` (special slice, root finding on the quantile): optimize |
| 103 | `uniform` | `unifpdf`, `Uniform` | `Uniform`, `FitUniform` | done | |
| 104 | `vonmises` | | `VonMises`, `FitVonMises` | done | |
| 105 | `vonmises_line` | | | missing | closed over `VonMises` pieces |
| 106 | `wald` | | | missing | closed |
| 107 | `weibull_min` | `wblpdf`, `Weibull` | `Weibull`, `FitWeibull` | done | |
| 108 | `weibull_max` | | | missing | closed (mirror of `Weibull`) |
| 109 | `wrapcauchy` | | | missing | closed |

## 2. Multivariate distributions

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 110 | `multivariate_normal` | `mvnpdf`, `mvncdf`, `mvnrnd` | `MultivariateNormal`, `FitMultivariateNormal` | done | |
| 111 | `matrix_normal` | | | missing | Kronecker structure on `MultivariateNormal` |
| 112 | `dirichlet` | | `Dirichlet` | done | |
| 113 | `dirichlet_multinomial` | | | missing | closed (`LogGamma`) |
| 114 | `wishart`, `invwishart` | `wishrnd`, `iwishrnd` | | missing | `multigammaln` (special slice 1); Bartlett decomposition, Cholesky (go-ndarray) |
| 115 | `multinomial` | `mnpdf`, `mnrnd` | `Multinomial` | done | |
| 116 | `special_ortho_group`, `ortho_group`, `unitary_group` | | | missing | QR / Householder (go-ndarray linalg); complex for `unitary_group` |
| 117 | `random_correlation` | `randcorr` | | missing | Davies-Higham, Givens rotations |
| 118 | `multivariate_t` | `mvtpdf`, `mvtcdf`, `mvtrnd` | | missing | Genz-Bretz cdf (reuse the `mvn_cdf.go` lattice rule) |
| 119 | `multivariate_hypergeom` | | | missing | closed |
| 120 | `normal_inverse_gamma` | | | missing | closed |
| 121 | `random_table` | | | missing | Patefield algorithm |
| 122 | `uniform_direction` | | | missing | normal draws, normalisation |
| 123 | `vonmises_fisher` | | | missing | `Iv`/`Ive` (have); rejection sampler (Wood) |
| 124 | `matrix_t` | | | missing | `multigammaln` |
| 125 | `Covariance` | | | missing | factorisation classes (`from_diagonal`, `from_precision`, `from_cholesky`, `from_eigendecomposition`) |

## 3. Discrete distributions

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 126 | `bernoulli` | | `Bernoulli` | done | |
| 127 | `betabinom` | | | missing | closed (`LogBeta`) |
| 128 | `betanbinom` | | | missing | closed |
| 129 | `binom` | `binopdf`, `Binomial` | `Binomial` | done | |
| 130 | `boltzmann` | | | missing | closed |
| 131 | `dlaplace` | | | missing | closed |
| 132 | `geom` | `geopdf` | `Geometric` | done | |
| 133 | `hypergeom` | `hygepdf` | `Hypergeometric` | done | |
| 134 | `logser` | | | missing | closed (series, `BetaInc`) |
| 135 | `nbinom` | `nbinpdf`, `NegativeBinomial` | `NegativeBinomial` | done | |
| 136 | `nchypergeom_fisher`, `nchypergeom_wallenius` | | | missing | port of SciPy's `biasedurn` C++; integrate for Wallenius |
| 137 | `nhypergeom` | | | missing | closed |
| 138 | `planck` | | | missing | closed |
| 139 | `poisson` | `poisspdf`, `Poisson` | `Poisson` | done | |
| 140 | `poisson_binom` | | | missing | closed (DP or FFT of the characteristic function) |
| 141 | `randint` | `unidpdf` | `DiscreteUniform` | done | |
| 142 | `skellam` | | | missing | `Iv` (have) or `NoncentralChiSquared` cdf |
| 143 | `yulesimon` | | | missing | closed |
| 144 | `zipf` | | `Zipf` | done | |
| 145 | `zipfian` | | | missing | closed (`HurwitzZeta`, have) |

## 4. Distribution infrastructure

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 146 | `rv_continuous` (user subclass with only `_pdf`/`_cdf`; generic ppf, moments, fit, rvs) | `ProbabilityDistribution`, `makedist` | `Continuous` interface (no generic fallbacks) | partial | generic ppf by root finding (optimize), moments and entropy by quadrature (integrate), rvs by inversion; this one layer would make every `integrate`-dep row above work |
| 147 | `rv_discrete` (user-defined pmf values) | | `Discrete` interface | partial | same, with summation |
| 148 | `rv_histogram` | `PiecewiseLinear`, `Kernel` | | missing | closed over `Histogram` |
| 149 | `expect`, `nnlf`, `fit_loc_scale`, `freeze`, `stats(moments=)` methods | `mean`, `var`, `icdf`, `cdf`, `truncate` (distribution objects) | `Moment`, `Mean`, `Var`... | partial | `Expect` (integrate), `NLLF`, `FitLocScale`, `Truncate` |
| 150 | `fit` as a method with `bounds`, `method=` ("mle", "mm"), `CensoredData` | `fitdist` with `Censoring`, `Frequency` | `FitXxx` + `FitOptions.Fixed` | partial | `bounds`, method of moments, optimizer choice; dep: optimize |
| 151 | `CensoredData` | `fitdist(..., 'Censoring')` | | missing | likelihood terms for left/right/interval censoring |
| 152 | `stats.fit`, `goodness_of_fit` | `fitdist`, `allfitdist` | | missing | generic fit over distribution objects; Monte Carlo (row 187) |
| 153 | `make_distribution`, `Normal`, `Logistic`, `Uniform`, `Binomial`, `Mixture`, `order_statistic`, `truncate`, `abs`, `exp`, `log` (random-variable infrastructure) | `truncate`, `gmdistribution` (mixtures) | | missing | design decision first; integrate for generic transforms |

## 5. Summary statistics

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 154 | `describe` | `summary`, `describe` (table) | `Describe` | done | |
| 155 | `gmean`, `hmean`, `pmean` | `geomean`, `harmmean` | `GMean`, `HMean`, `PMean` | done | |
| 156 | `kurtosis`, `skew`, `moment` | `kurtosis`, `skewness`, `moment` | `Kurtosis`, `Skew`, `Moment` | done | |
| 157 | `mode` | `mode` | `Mode` | done | |
| 158 | `lmoment` | | | missing | none (probability-weighted moments) |
| 159 | `expectile` | | | missing | none (root finding on the weighted mean; optimize) |
| 160 | `kstat`, `kstatvar` | | | missing | none |
| 161 | `tmean`, `tvar`, `tmin`, `tmax`, `tstd`, `tsem` | `trimmean` (different trimming) | `TMean`, `TVar`, `TMin`, `TMax`, `TStd`, `TSem` | done | |
| 162 | `variation` | `std ./ mean` | `Variation` | done | |
| 163 | `rankdata` | `tiedrank` | `RankData` | done | |
| 164 | `tiecorrect` | | | missing | none |
| 165 | `trim_mean` | `trimmean` | `TrimMean` | done | |
| 166 | `gstd` | | | missing | none |
| 167 | `iqr`, `sem`, `median_abs_deviation` | `iqr`, `mad` | `IQR`, `SEM`, `MedianAbsDeviation` | done | |
| 168 | `bayes_mvs`, `mvsdist` | | | missing | `StudentT`, `ChiSquared`, `Gamma` (have) |
| 169 | `entropy` | `entropy` | `Entropy` | done | |
| 170 | `differential_entropy` | | `DifferentialEntropy` | done | |

## 6. Frequency statistics

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 171 | `quantile` | `quantile`, `prctile` | `Quantile`, `Quantiles`, `Percentile`, `Percentiles` | done | |
| 172 | `estimated_cdf` | `ecdf` | | missing | none |
| 173 | `cumfreq`, `relfreq` | `histcounts` (`cumcount`) | `CumFreq`, `RelFreq` | done | |
| 174 | `percentileofscore` | | | missing | none |
| 175 | `scoreatpercentile` | `prctile` | `Percentile` | done | |
| 176 | `binned_statistic` | `histcounts`, `accumarray` | `BinnedStatistic` | done | |
| 177 | `binned_statistic_2d`, `binned_statistic_dd` | `histcounts2` | | missing | none (extend `BinnedStatistic`) |

## 7. Hypothesis tests

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 178 | `ttest_1samp` | `ttest`, `ztest` | `TTest1Samp` | done | |
| 179 | `binomtest` | `binocdf`-based; `proportion` CI | | missing | none (`Binomial`, Clopper-Pearson via `BetaIncInv`) |
| 180 | `quantile_test` | | | missing | `Binomial` |
| 181 | `skewtest`, `kurtosistest`, `normaltest`, `jarque_bera` | `jbtest`, `skewness` | `SkewTest`, `KurtosisTest`, `Normaltest`, `JarqueBera` | done | |
| 182 | `shapiro` | | `ShapiroWilk` | done | |
| 183 | `anderson` | `adtest` | `AndersonDarling` | partial | `weibull_min` family; critical-value tables via Monte Carlo (`method=`) |
| 184 | `cramervonmises` | | | missing | none (series for the exact cdf) |
| 185 | `ks_1samp`, `kstest` | `kstest` | `KSTest1Samp` | done | `kstest` by distribution name: pass the cdf |
| 186 | `chisquare`, `power_divergence` | `chi2gof` | `ChiSquare`, `PowerDivergence` | done | |
| 187 | `goodness_of_fit` | | | missing | Monte Carlo (row 191) + row 152 |
| 188 | `ttest_rel`, `ttest_ind` | `ttest`, `ttest2` | `TTestRel`, `TTestInd` | done | |
| 189 | `ttest_ind_from_stats` | | | missing | none |
| 190 | `wilcoxon` | `signrank` | `Wilcoxon` | done | |
| 191 | `linregress` | `fitlm`, `regress`, `polyfit` | `Linregress` | done | |
| 192 | `pearsonr`, `spearmanr`, `kendalltau` | `corr`, `corrcoef` | `Pearsonr`, `Spearmanr`, `Kendalltau` | done | |
| 193 | `spearmanrho` | `corr('Type','Spearman')` | | missing | none (new in 1.18; `Spearmanr` covers the statistic) |
| 194 | `pointbiserialr` | | | missing | none (`Pearsonr`) |
| 195 | `chatterjeexi` | | | missing | none |
| 196 | `weightedtau` | | | missing | none (merge-sort based) |
| 197 | `somersd` | | | missing | none |
| 198 | `siegelslopes`, `theilslopes` | | | missing | none |
| 199 | `page_trend_test` | | | missing | none (exact null by enumeration) |
| 200 | `multiscale_graphcorr` | | | missing | permutation machinery (row 202) |
| 201 | `chi2_contingency`, `fisher_exact` | `crosstab`, `fishertest` | `Chi2Contingency`, `FisherExact` | done | |
| 202 | `barnard_exact`, `boschloo_exact` | | | missing | optimize (`shgo` in SciPy, a 1-D maximisation suffices) |
| 203 | `poisson_means_test` | | | missing | none |
| 204 | `mannwhitneyu` | `ranksum` | `MannWhitneyU` | done | |
| 205 | `bws_test` | | | missing | none |
| 206 | `ranksums`, `brunnermunzel`, `mood`, `ansari` | `ansaribradley` | | missing | none |
| 207 | `cramervonmises_2samp`, `epps_singleton_2samp` | | | missing | none |
| 208 | `ks_2samp` | `kstest2` | `KSTest2Samp` | done | |
| 209 | `f_oneway`, `kruskal`, `levene`, `bartlett` | `anova1`, `kruskalwallis`, `vartestn` | `FOneway`, `KruskalWallis`, `Levene`, `Bartlett` | done | |
| 210 | `tukey_hsd` | `multcompare` | | missing | `studentized_range` cdf (row 94) |
| 211 | `dunnett` | `multcompare('CriticalValueType','dunnett')` | | missing | `multivariate_t` cdf (row 118) |
| 212 | `alexandergovern`, `fligner`, `median_test`, `friedmanchisquare`, `anderson_ksamp` | `friedman` | | missing | none |

## 8. Resampling and multiple testing

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 213 | `bootstrap` | `bootci`, `bootstrp` | `Bootstrap` | done | percentile, basic, BCa |
| 214 | `monte_carlo_test` | | | missing | none |
| 215 | `permutation_test` | | | missing | none (exact and randomised) |
| 216 | `power` | `sampsizepwr` | | missing | `monte_carlo_test` |
| 217 | `MonteCarloMethod`, `PermutationMethod`, `BootstrapMethod` as the `method=` of the tests | | `Method` (auto/exact/asymptotic) and `BootstrapMethod` (interval kind only) | partial | resampling `method=` objects accepted by each test |
| 218 | `combine_pvalues` | | | missing | none (Fisher, Pearson, Stouffer, Tippett, Mudholkar-George) |
| 219 | `false_discovery_control` | `mafdr` | `FalseDiscoveryControl` (bh, by); also statsmodels `multipletests`: `MultipleTests` (bonferroni, sidak, holm-sidak, holm, simes-hochberg, hommel, fdr_bh, fdr_by, fdr_tsbh, fdr_tsbky) | done | `mafdr` (storey/bootstrap) is not covered |

## 9. Transformations, distances, survival, directional

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 220 | `boxcox`, `boxcox_normmax`, `boxcox_llf` | `boxcox` | | missing | optimize (Brent); `special` `boxcox` kernels (slice 1) |
| 221 | `yeojohnson`, `yeojohnson_normmax`, `yeojohnson_llf` | | | missing | optimize |
| 222 | `obrientransform`, `sigmaclip`, `gzscore` | `rmoutliers` | | missing | none |
| 223 | `trimboth`, `trim1`, `zmap`, `zscore` | `zscore`, `trimmean` | `TrimBoth`, `Trim1`, `ZMap`, `ZScore` | done | |
| 224 | `wasserstein_distance`, `energy_distance` | | | missing | none |
| 225 | `wasserstein_distance_nd` | | | missing | linear assignment (go-ndarray or own Hungarian) |
| 226 | `ecdf`, `logrank` | `ecdf` | | missing | none (Kaplan-Meier, Nelson-Aalen, Greenwood CI) |
| 227 | `directional_stats`, `circmean`, `circvar`, `circstd` | `circ_*` (CircStat, not core) | | missing | none |
| 228 | `sobol_indices` | | | missing | QMC (section 10) |
| 229 | `ppcc_max`, `ppcc_plot`, `probplot`, `boxcox_normplot`, `yeojohnson_normplot` | `probplot`, `qqplot` | | missing | numbers only (no plotting): optimize + distributions |
| 230 | `gaussian_kde` | `ksdensity` | `GaussianKDE` | done | |
| 231 | warning and error classes (`DegenerateDataWarning`, `ConstantInputWarning`, `NearConstantInputWarning`, `FitError`) | | `ErrParam`, `ErrData`, `ErrFit` | done | errors, not warnings |

## 10. Quasi-Monte Carlo (`scipy.stats.qmc`)

The go-ndarray `random` package is NumPy's `Generator`. The engines will
live in a `qmc` sub-package of this module. The API, the exact-path decision
for each engine, validation, performance and the task order are in
[design-qmc.md](design-qmc.md).

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 232 | `Sobol` (with scrambling, `fast_forward`, `random_base2`) | `sobolset`, `scramble` | | missing | direction numbers (Joe-Kuo, embed), Owen scrambling |
| 233 | `Halton` | `haltonset` | | missing | primes, Owen scrambling |
| 234 | `LatinHypercube` | `lhsdesign`, `lhsnorm` | | missing | `random.Generator` |
| 235 | `PoissonDisk` | | | missing | `random.Generator` |
| 236 | `MultinomialQMC`, `MultivariateNormalQMC` | | | missing | engines + `Multinomial`, `MultivariateNormal` |
| 237 | `QMCEngine` (base: `random`, `reset`, `fast_forward`) | `qrandstream` | | missing | engines |
| 238 | `discrepancy`, `geometric_discrepancy`, `update_discrepancy`, `scale` | | | missing | none |

## 11. Contingency tables (`scipy.stats.contingency`)

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 239 | `chi2_contingency` | `crosstab` (chi2 output) | `Chi2Contingency` | done | |
| 240 | `expected_freq`, `margins` | | | missing | none |
| 241 | `association` (Cramer's V, Tschuprow, Pearson) | | | missing | none |
| 242 | `crosstab` | `crosstab` | | missing | none |
| 243 | `odds_ratio`, `relative_risk` (with confidence intervals) | | | missing | `NoncentralHypergeometric` for the conditional MLE (row 136) |

## 12. Random-variate generators (`scipy.stats.sampling`) and `mstats`

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 244 | `TransformedDensityRejection`, `SimpleRatioUniforms`, `RatioUniforms` | `slicesample`, `mhsample` | | missing | UNU.RAN algorithms ported |
| 245 | `NumericalInverseHermite`, `NumericalInversePolynomial`, `FastGeneratorInversion` | | | missing | UNU.RAN; integrate for the cdf tables |
| 246 | `DiscreteAliasUrn`, `DiscreteGuideTable` | `randsample` (weighted) | | missing | none (Walker alias) |
| 247 | `UNURANError` | | | missing | with the above |
| 248 | `stats.mstats` (masked-array twins of the functions above) | `omitnan` flags | `NaNOptions` (omit / propagate / raise) | partial | masks are NaN policy here; no masked-array type, so the `mstats`-only extras (`mquantiles`, `hdquantiles`, `hdmedian`, `trimmed_mean`, `winsorize`, `rsh`, `plotting_positions`, `idealfourths`, `msign`) are missing |

## 13. MATLAB-only statistics (no `scipy.stats` counterpart)

Statistics (not machine-learning) functions of the toolbox without a SciPy
equivalent. Regression and design-of-experiments are `statsmodels` territory
in the Python world; they are listed because the goal is the broadest
supported mathematics, and may be moved to a separate module.

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| M1 | | `ztest`, `vartest`, `vartest2`, `signtest`, `runstest`, `lillietest`, `chi2gof` (variants) | | missing | none; `lillietest` needs Monte Carlo tables |
| M2 | | `anova1` (table), `anova2`, `anovan`, `manova1`, `ranova`, `multcompare` | | missing | `F`; linear-model machinery |
| M3 | | `fitlm`, `fitglm`, `stepwiselm`, `regress`, `robustfit`, `nlinfit`, `fitnlm`, `plsregress`, `mnrfit` | | missing | go-ndarray linalg; optimize (`least_squares`) |
| M4 | | `copulacdf`, `copulapdf`, `copulafit`, `copularnd`, `copulastat` | | missing | `multivariate_t`, `MultivariateNormal` |
| M5 | | `johnsrnd`, `pearsrnd`, `mvksdensity`, `ksdensity` options | | missing | rows 53, 54, 82 |
| M6 | | `mle`, `likelihood ratio` helpers `lratiotest`, `negloglik`, `proflik` | | missing | optimize |
| M7 | | `jackknife`, `bootstrp` | | missing | none |
| M8 | | `partialcorr`, `corrcoef` with p-values and CIs, `corrtype` | | missing | `StudentT` (have) |
| M9 | | `mahal`, `cov`, `cov` shrinkage (`robustcov`) | | missing | go-ndarray linalg |
| M10 | | descriptive extras: `range`, `bounds`, `rms`, `grpstats`, `groupsummary` | | missing | none |
| M11 | | Distribution families with no SciPy name: `Loglogistic` (fisk), `BirnbaumSaunders` (fatiguelife), `Rician` (rice), `Stable` (levy_stable), `Kernel`, `PiecewiseLinear`, `Multinomial` object | | missing | rows 25, 24, 89, 65, 148 |
| M12 | | `hmmgenerate`, `hmmestimate` (Markov models) | | missing | out of the first scope |
| M13 | | `binofit`, `poissfit`, `unifit`, `ecdfhist`, `histfit` (the other `*fit` functions are the `FitXxx` of section 1) | | missing | closed forms with confidence intervals |

## Proposed slices

Ordered by cost per row gained. Each is one PR with mpmath fixtures and SciPy
bit-parity of `rvs` where the algorithm allows.

1. **Closed-form continuous distributions.** The rows marked *closed* in
   section 1 (`alpha` ... `wrapcauchy`), in batches of ten by family
   (extreme-value and mirror images first: `gumbel_l`, `weibull_max`,
   `genextreme`, `genpareto`; then exponentials/Weibull relatives; Burr, fisk,
   Pearson, Johnson). Dep: none beyond today's `special`.
2. **Closed-form discrete and multivariate.** Rows 113, 119, 120, 122, 127,
   128, 130, 131, 134, 137, 138, 140, 143, 145; matrix_normal (111). Dep: none.
3. **Descriptive and sample tests with no dependency.** Rows 158, 160, 164,
   166, 172, 174, 177, 179-180, 184, 189, 193-199, 203, 205-207, 212, 218-219,
   222, 224, 227, 240-242. About 50 names. Dep: none.
4. **Resampling.** Rows 214-217 (`monte_carlo_test`, `permutation_test`,
   `power`, the `method=` objects), then 187 and 200. Dep: slice 3's tests
   as the consumers; `random.Generator`.
5. **Generic distribution layer (row 146-153).** `rv_continuous` fallbacks,
   `Expect`, `CensoredData`, `fit` with bounds. Dep: `go-numerics/integrate`
   and `optimize`; unlocks the *integrate*/*optimize* rows (33, 37, 38, 65, 80,
   84, 88, 94, 102) at generic speed.
6. **Special-function-bound distributions.** `ksone`, `kstwo`, `kstwobign`
   (move `ksdist.go` into `special`), `ncf`, `nct` (non-central beta/t),
   `skewnorm` (`owens_t`), `gausshyper` (`hyp2f1`), `gompertz` (`exp1`),
   `rice`. Dep: the matching `go-numerics/special` slices.
7. **Wishart family and orthogonal groups.** Rows 114, 116, 117, 124. Dep:
   `multigammaln`, go-ndarray linalg (QR, Cholesky).
8. **Multivariate t, Tukey, Dunnett.** Rows 118, 94, 210, 211. Dep: Genz-Bretz
   (have the lattice rule), studentized-range integration.
9. **Transformations and survival.** Rows 220, 221, 226, 229. Dep: optimize
   (Brent).
10. **QMC engines.** Rows 232-238, then 228 (`sobol_indices`) and the `qmc_quad`
    row of `go-numerics/integrate`. Dep: Joe-Kuo direction numbers (embedded
    data), Owen scrambling.
11. **UNU.RAN samplers.** Rows 244-247. Dep: slice 5 for the numerical
    inversions.
12. **MATLAB-only (section 13), regression and ANOVA.** After slices 3, 5 and
    4; M2/M3 depend on a linear-model layer that probably belongs in its own
    module over go-ndarray linalg.

| case | n | before: speed-up, one core | after: speed-up, one core | gain | slower because |
|---|--:|--:|--:|--:|---|
| f.fit | 10000 | 0.00037x | 0.00101x | 2.72x | own: Nelder-Mead polished to ~1e-10 (SciPy's fmin: 4 digits), DD log-density per point |
| t.ppf | 100000 | 0.00548x | 0.00539x | 0.985x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| t.isf | 20000 | 0.00577x | 0.00601x | 1.04x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| chi.fit | 10000 | 0.0031x | 0.00618x | 1.99x | own: Nelder-Mead polished to ~1e-10 (SciPy's fmin: 4 digits), DD log-density per point |
| ncx2.isf | 20000 | 0.0102x | 0.00996x | 0.981x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| ncx2.ppf | 20000 | 0.01x | 0.0102x | 1.02x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| ncx2.cdf | 100000 | 0.0142x | 0.0143x | 1.01x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| ncx2.sf | 100000 | 0.0136x | 0.0144x | 1.06x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| invgamma.fit | 10000 | 0.0141x | 0.0251x | 1.78x | own: Nelder-Mead polished to ~1e-10 (SciPy's fmin: 4 digits), DD log-density per point |
| lomax.fit | 10000 | 0.0234x | 0.0254x | 1.09x | own: Nelder-Mead polished to ~1e-10 (SciPy's fmin: 4 digits), DD log-density per point |
| f.isf | 20000 | 0.0314x | 0.0319x | 1.02x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| vonmises.cdf | 100000 | 0.033x | 0.033x | 0.999x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| f.ppf | 20000 | 0.0299x | 0.0333x | 1.11x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| hypergeom.pmf | 100000 | 0.0303x | 0.0336x | 1.11x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| beta.ppf | 100000 | 0.0348x | 0.0346x | 0.993x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| nbinom.cdf | 100000 | 0.0349x | 0.0348x | 0.996x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| nbinom.sf | 100000 | 0.0346x | 0.0348x | 1x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| beta.isf | 20000 | 0.0408x | 0.0353x | 0.867x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| binom.sf | 100000 | 0.035x | 0.0358x | 1.02x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| binom.cdf | 100000 | 0.0364x | 0.0361x | 0.993x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| beta.fit | 10000 | 0.00829x | 0.0403x | 4.86x | own: Nelder-Mead polished to ~1e-10 (SciPy's fmin: 4 digits), DD log-density per point |
| hypergeom.sf | 100000 | 0.0451x | 0.0451x | 0.999x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| hypergeom.cdf | 100000 | 0.0384x | 0.0455x | 1.19x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| logistic.fit | 10000 | 0.0441x | 0.0478x | 1.08x | own: Nelder-Mead polished to ~1e-10 (SciPy's fmin: 4 digits), DD log-density per point |
| gamma.fit | 10000 | 0.0237x | 0.0486x | 2.05x | own: Nelder-Mead polished to ~1e-10 (SciPy's fmin: 4 digits), DD log-density per point |
| t.cdf | 1000000 | 0.041x | 0.0539x | 1.32x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| t.sf | 100000 | 0.0531x | 0.0564x | 1.06x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| f.cdf | 100000 | 0.0542x | 0.0567x | 1.05x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| f.sf | 100000 | 0.0535x | 0.0568x | 1.06x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| beta.cdf | 1000000 | 0.0579x | 0.0575x | 0.994x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| beta.sf | 100000 | 0.0588x | 0.0588x | 1x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| ncx2.pdf | 100000 | 0.0593x | 0.0657x | 1.11x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| vonmises.sf | 100000 | 0.0333x | 0.0687x | 2.07x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| weibull_min.fit | 10000 | 0.0462x | 0.0746x | 1.62x | own: Nelder-Mead polished to ~1e-10 (SciPy's fmin: 4 digits), DD log-density per point |
| mvn.logpdf | 100000 | 0.0678x | 0.0749x | 1.1x | own: double-double solve per point |
| chi2.fit | 10000 | 0.0423x | 0.0827x | 1.95x | own: Nelder-Mead polished to ~1e-10 (SciPy's fmin: 4 digits), DD log-density per point |
| invgamma.cdf | 100000 | 0.088x | 0.0881x | 1x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| invgamma.sf | 100000 | 0.0864x | 0.0932x | 1.08x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| chi.ppf | 20000 | 0.0973x | 0.0992x | 1.02x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| binom.ppf | 20000 | 0.101x | 0.102x | 1.02x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| binom.isf | 20000 | 0.102x | 0.102x | 1x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| chi.isf | 20000 | 0.096x | 0.106x | 1.1x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| gamma.isf | 20000 | 0.119x | 0.113x | 0.951x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| invgamma.ppf | 20000 | 0.115x | 0.114x | 0.987x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| invgamma.rvs | 1000000 | 0.116x | 0.114x | 0.988x | special: inverse incomplete gamma per draw (SciPy's generic ppf sampling) |
| invgamma.isf | 20000 | 0.111x | 0.115x | 1.03x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| poisson.logpmf | 100000 | 0.119x | 0.118x | 0.996x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| gamma.ppf | 100000 | 0.118x | 0.119x | 1.01x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| chi2.isf | 20000 | 0.121x | 0.122x | 1.01x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| nbinom.logpmf | 100000 | 0.125x | 0.122x | 0.981x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| geom.rvs | 1000000 | 0.134x | 0.126x | 0.942x | ndarray: the generator's draws |
| chi2.ppf | 20000 | 0.123x | 0.126x | 1.03x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| cauchy.fit | 10000 | 0.131x | 0.13x | 0.991x | own: Nelder-Mead polished to ~1e-10 (SciPy's fmin: 4 digits), DD log-density per point |
| lognorm.cdf | 100000 | 0.142x | 0.132x | 0.931x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| poisson.sf | 100000 | 0.136x | 0.135x | 0.996x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| poisson.cdf | 100000 | 0.134x | 0.137x | 1.02x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| reciprocal.fit | 10000 | 0.151x | 0.15x | 0.993x | own: Nelder-Mead polished to ~1e-10 (SciPy's fmin: 4 digits), DD log-density per point |
| reciprocal.cdf | 100000 | 0.152x | 0.16x | 1.05x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| reciprocal.sf | 100000 | 0.108x | 0.16x | 1.49x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| chi.sf | 100000 | 0.148x | 0.161x | 1.09x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| poisson.pmf | 100000 | 0.153x | 0.161x | 1.05x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| chi.cdf | 100000 | 0.151x | 0.164x | 1.09x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| triang.fit | 10000 | 0.152x | 0.166x | 1.1x | own: Nelder-Mead polished to ~1e-10 (SciPy's fmin: 4 digits), DD log-density per point |
| binom.logpmf | 100000 | 0.168x | 0.168x | 1x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| lognorm.sf | 100000 | 0.196x | 0.168x | 0.861x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| nbinom.pmf | 100000 | 0.18x | 0.177x | 0.984x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| expon.fit | 10000 | 0.0865x | 0.179x | 2.07x | own: Nelder-Mead polished to ~1e-10 (SciPy's fmin: 4 digits), DD log-density per point |
| gamma.cdf | 1000000 | 0.145x | 0.18x | 1.24x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| gamma.sf | 100000 | 0.202x | 0.182x | 0.901x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| lognorm.isf | 20000 | 0.19x | 0.184x | 0.973x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| lognorm.ppf | 20000 | 0.232x | 0.187x | 0.803x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| cauchy.sf | 100000 | 0.192x | 0.192x | 1x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| norm.ppf | 1000000 | 0.198x | 0.192x | 0.97x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| uniform.fit | 10000 | 0.121x | 0.195x | 1.61x | own: Nelder-Mead polished to ~1e-10 (SciPy's fmin: 4 digits), DD log-density per point |
| t.fit | 10000 | 0.0198x | 0.197x | 9.95x | own: Nelder-Mead polished to ~1e-10 (SciPy's fmin: 4 digits), DD log-density per point |
| cauchy.cdf | 100000 | 0.189x | 0.206x | 1.09x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| invgamma.pdf | 100000 | 0.16x | 0.207x | 1.29x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| entropy | 1000000 | 0.0404x | 0.208x | 5.14x | own: correctly rounded log/pow per element |
| geom.pmf | 100000 | 0.255x | 0.213x | 0.835x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| chi2.sf | 1000000 | 0.221x | 0.219x | 0.99x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| invgamma.logpdf | 100000 | 0.162x | 0.228x | 1.4x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| weibull_min.ppf | 20000 | 0.214x | 0.228x | 1.06x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| chi2.cdf | 100000 | 0.223x | 0.233x | 1.04x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| pmean | 1000000 | 0.064x | 0.233x | 3.64x | own: correctly rounded log/pow per element |
| logistic.sf | 100000 | 0.22x | 0.235x | 1.07x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| lognorm.pdf | 100000 | 0.163x | 0.236x | 1.44x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| logistic.cdf | 100000 | 0.219x | 0.239x | 1.09x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| lomax.ppf | 20000 | 0.252x | 0.24x | 0.956x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| geom.logpmf | 100000 | 0.23x | 0.242x | 1.05x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| norm.isf | 20000 | 0.15x | 0.242x | 1.61x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| power_divergence | 1000000 | 0.062x | 0.243x | 3.92x | own: correctly rounded log/pow per element |
| geom.cdf | 100000 | 0.214x | 0.249x | 1.17x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| gumbel_r.rvs | 1000000 | 0.0513x | 0.251x | 4.88x | own: SciPy's transform with correctly rounded elementary functions |
| binom.pmf | 100000 | 0.248x | 0.251x | 1.01x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| nbinom.isf | 20000 | 0.249x | 0.252x | 1.01x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| geom.sf | 100000 | 0.251x | 0.253x | 1.01x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| nbinom.ppf | 20000 | 0.255x | 0.255x | 1x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| reciprocal.logpdf | 100000 | 0.549x | 0.26x | 0.473x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| weibull_min.isf | 20000 | 0.213x | 0.26x | 1.22x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| lognorm.fit | 10000 | 0.06x | 0.265x | 4.42x | own: Nelder-Mead polished to ~1e-10 (SciPy's fmin: 4 digits), DD log-density per point |
| gumbel_r.sf | 100000 | 0.266x | 0.269x | 1.01x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| weibull_min.rvs | 1000000 | 0.0737x | 0.269x | 3.65x | own: SciPy's transform with correctly rounded elementary functions |
| lognorm.logpdf | 100000 | 0.276x | 0.273x | 0.992x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| multinomial.rvs | 100000 | 0.302x | 0.288x | 0.954x | ndarray: the generator's draws |
| pareto.ppf | 20000 | 0.308x | 0.292x | 0.949x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| pareto.isf | 20000 | 0.341x | 0.297x | 0.87x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| weibull_min.pdf | 100000 | 0.323x | 0.299x | 0.925x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| cauchy.logpdf | 100000 | 0.298x | 0.302x | 1.02x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| cauchy.rvs | 1000000 | 0.307x | 0.308x | 1x | own: SciPy's transform with correctly rounded elementary functions |
| mvn.cdf | 1 | 0.226x | 0.31x | 1.37x | own: randomised lattice rule in double-double |
| zipf.rvs | 1000000 | 0.309x | 0.315x | 1.02x | ndarray: the generator's draws |
| pareto.rvs | 1000000 | 0.094x | 0.315x | 3.36x | own: SciPy's transform with correctly rounded elementary functions |
| geom.ppf | 20000 | 0.239x | 0.319x | 1.33x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| weibull_min.logpdf | 100000 | 0.336x | 0.32x | 0.951x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| dirichlet.logpdf | 100000 | 0.0788x | 0.321x | 4.07x | own: double-double sum of logs (Ziv fast path) |
| kde.integrate_box_1d | 10000 | 0.407x | 0.321x | 0.789x | own: compensated sums of exponentials per point |
| weibull_min.sf | 100000 | 0.324x | 0.323x | 0.996x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| geom.isf | 20000 | 0.321x | 0.328x | 1.02x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| logistic.rvs | 1000000 | 0.353x | 0.333x | 0.944x | ndarray: the generator's draws |
| pareto.pdf | 100000 | 0.348x | 0.334x | 0.96x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| f.logpdf | 100000 | 0.136x | 0.334x | 2.45x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| logistic.pdf | 100000 | 0.326x | 0.335x | 1.03x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| weibull_min.cdf | 100000 | 0.314x | 0.337x | 1.08x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| cauchy.pdf | 100000 | 0.332x | 0.341x | 1.03x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| lomax.isf | 20000 | 0.269x | 0.35x | 1.31x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| linregress | 1000000 | 0.0559x | 0.351x | 6.27x | own: exact (double-double-equivalent) sums, compensated on two lanes; NumPy uses SIMD/BLAS |
| vonmises.pdf | 100000 | 0.342x | 0.352x | 1.03x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| hypergeom.logpmf | 100000 | 0.351x | 0.354x | 1.01x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| halfnorm.isf | 20000 | 0.373x | 0.36x | 0.965x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| lomax.cdf | 100000 | 0.604x | 0.362x | 0.598x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| gumbel_r.pdf | 100000 | 0.333x | 0.363x | 1.09x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| lomax.logpdf | 100000 | 0.363x | 0.365x | 1.01x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| lomax.rvs | 1000000 | 0.159x | 0.366x | 2.3x | own: SciPy's transform with correctly rounded elementary functions |
| pareto.cdf | 100000 | 0.473x | 0.371x | 0.784x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| cauchy.isf | 20000 | 0.388x | 0.379x | 0.977x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| cauchy.ppf | 20000 | 0.35x | 0.38x | 1.08x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| norm.fit | 10000 | 0.234x | 0.387x | 1.65x | own: Nelder-Mead polished to ~1e-10 (SciPy's fmin: 4 digits), DD log-density per point |
| logistic.logpdf | 100000 | 0.372x | 0.392x | 1.05x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| gmean | 1000000 | 0.0606x | 0.395x | 6.51x | own: correctly rounded log/pow per element |
| pareto.fit | 10000 | 0.0682x | 0.395x | 5.79x | own: Nelder-Mead polished to ~1e-10 (SciPy's fmin: 4 digits), DD log-density per point |
| kde.evaluate2d | 10000 | 0.239x | 0.396x | 1.66x | own: compensated sums of exponentials per point |
| halfnorm.ppf | 20000 | 0.34x | 0.401x | 1.18x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| gumbel_r.cdf | 100000 | 0.408x | 0.404x | 0.989x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| lomax.sf | 100000 | 0.377x | 0.406x | 1.08x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| f.pdf | 100000 | 0.176x | 0.411x | 2.34x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| chi2.pdf | 100000 | 0.365x | 0.412x | 1.13x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| norm.sf | 100000 | 0.245x | 0.416x | 1.7x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| chi.logpdf | 100000 | 0.293x | 0.417x | 1.42x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| mode | 1000000 | 0.441x | 0.422x | 0.958x | own: comparison sort (data holding both -0 and +0 cannot take the radix path bit-identically) |
| chi.pdf | 100000 | 0.307x | 0.423x | 1.38x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| gamma.pdf | 1000000 | 0.242x | 0.425x | 1.75x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| logistic.isf | 20000 | 0.424x | 0.428x | 1.01x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| norm.cdf | 1000000 | 0.427x | 0.431x | 1.01x | special: incomplete gamma/beta, their inverses, Ndtr/Ndtri |
| pareto.sf | 100000 | 0.445x | 0.435x | 0.978x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| reciprocal.isf | 20000 | 0.378x | 0.437x | 1.16x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| reciprocal.pdf | 100000 | 1.99x | 0.437x | 0.22x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| dirichlet.rvs | 100000 | 0.455x | 0.45x | 0.99x | ndarray: the generator's draws |
| gumbel_r.ppf | 20000 | 0.46x | 0.467x | 1.01x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| gumbel_r.isf | 20000 | 0.407x | 0.474x | 1.16x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| logistic.ppf | 20000 | 0.406x | 0.477x | 1.18x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| chisquare | 1000000 | 0.183x | 0.478x | 2.61x | own: exact (double-double-equivalent) sums, compensated on two lanes; NumPy uses SIMD/BLAS |
| lomax.pdf | 100000 | 0.297x | 0.479x | 1.61x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| chi2.logpdf | 100000 | 0.414x | 0.479x | 1.16x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| triang.isf | 20000 | 0.42x | 0.496x | 1.18x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| pareto.logpdf | 100000 | 0.516x | 0.506x | 0.98x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| f_oneway | 1000000 | 0.164x | 0.509x | 3.11x | own: exact (double-double-equivalent) sums, compensated on two lanes; NumPy uses SIMD/BLAS |
| reciprocal.ppf | 20000 | 1.87x | 0.51x | 0.274x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| t.pdf | 100000 | 0.0817x | 0.511x | 6.25x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| rayleigh.ppf | 20000 | 0.552x | 0.513x | 0.929x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| triang.ppf | 20000 | 0.408x | 0.521x | 1.28x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| triang.logpdf | 100000 | 0.504x | 0.524x | 1.04x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| rayleigh.logpdf | 100000 | 0.469x | 0.525x | 1.12x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| gamma.logpdf | 100000 | 0.279x | 0.526x | 1.88x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| tmax | 1000000 | 0.409x | 0.528x | 1.29x | own: scalar loops where NumPy has SIMD kernels |
| rayleigh.isf | 20000 | 0.884x | 0.529x | 0.598x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| laplace.pdf | 100000 | 0.513x | 0.53x | 1.03x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| halfnorm.rvs | 1000000 | 0.704x | 0.559x | 0.795x | ndarray: the generator's draws |
| hmean | 1000000 | 0.663x | 0.572x | 0.863x | own: correctly rounded log/pow per element |
| halfnorm.cdf | 100000 | 0.593x | 0.575x | 0.969x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| halfnorm.pdf | 100000 | 0.534x | 0.582x | 1.09x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| t.logpdf | 100000 | 0.0707x | 0.591x | 8.36x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| pearsonr | 1000000 | 0.106x | 0.599x | 5.64x | own: exact (double-double-equivalent) sums, compensated on two lanes; NumPy uses SIMD/BLAS |
| expon.pdf | 100000 | 0.633x | 0.604x | 0.954x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| norm.pdf | 1000000 | 0.649x | 0.614x | 0.946x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| tmean | 1000000 | 0.452x | 0.626x | 1.38x | own: scalar loops where NumPy has SIMD kernels |
| shapiro | 5000 | 0.659x | 0.641x | 0.972x | own |
| tmin | 1000000 | 0.397x | 0.65x | 1.64x | own: scalar loops where NumPy has SIMD kernels |
| anderson | 100000 | 0.399x | 0.655x | 1.64x | own: scalar loops where NumPy has SIMD kernels |
| reciprocal.rvs | 1000000 | 1.13x | 0.668x | 0.59x | own: SciPy's transform with correctly rounded elementary functions |
| sem | 1000000 | 0.589x | 0.676x | 1.15x | own |
| laplace.sf | 100000 | 0.652x | 0.684x | 1.05x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| laplace.rvs | 1000000 | 0.688x | 0.688x | 0.999x | ndarray: the generator's draws |
| laplace.cdf | 100000 | 0.676x | 0.696x | 1.03x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| halfnorm.sf | 100000 | 0.658x | 0.71x | 1.08x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| mvn.rvs | 100000 | 0.699x | 0.717x | 1.03x | own: NumPy's SVD factor, then ndarray's normals |
| beta.logpdf | 100000 | 0.149x | 0.724x | 4.86x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| nbinom.rvs | 1000000 | 0.713x | 0.736x | 1.03x | ndarray: the generator's draws |
| beta.pdf | 1000000 | 0.201x | 0.747x | 3.72x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| vonmises.logpdf | 100000 | 0.419x | 0.764x | 1.82x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| rayleigh.cdf | 100000 | 0.737x | 0.764x | 1.04x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| gamma.rvs | 1000000 | 0.72x | 0.781x | 1.09x | ndarray: the generator's draws |
| expon.ppf | 20000 | 0.878x | 0.794x | 0.904x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| f.rvs | 1000000 | 0.866x | 0.801x | 0.925x | ndarray: the generator's draws |
| moment | 1000000 | 0.803x | 0.818x | 1.02x | own: scalar loops where NumPy has SIMD kernels |
| beta.rvs | 1000000 | 0.842x | 0.826x | 0.981x | ndarray: the generator's draws |
| t.rvs | 1000000 | 0.817x | 0.832x | 1.02x | ndarray: the generator's draws |
| laplace.fit | 10000 | 0.234x | 0.835x | 3.57x | own: Nelder-Mead polished to ~1e-10 (SciPy's fmin: 4 digits), DD log-density per point |
| expon.cdf | 100000 | 0.791x | 0.836x | 1.06x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| chi.rvs | 1000000 | 0.803x | 0.836x | 1.04x | ndarray: the generator's draws |
| ncx2.rvs | 1000000 | 0.819x | 0.836x | 1.02x | ndarray: the generator's draws |
| mvn.fit | 100000 | 0.689x | 0.837x | 1.22x | own: Nelder-Mead polished to ~1e-10 (SciPy's fmin: 4 digits), DD log-density per point |
| chi2.rvs | 1000000 | 0.819x | 0.839x | 1.02x | ndarray: the generator's draws |
| triang.rvs | 1000000 | 0.824x | 0.844x | 1.03x | ndarray: the generator's draws |
| expon.rvs | 1000000 | 0.824x | 0.845x | 1.03x | ndarray: the generator's draws |
| randint.logpmf | 100000 | 0.84x | 0.85x | 1.01x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| average | 1000000 | 0.312x | 0.851x | 2.73x | own: scalar loops where NumPy has SIMD kernels |
| poisson.rvs | 1000000 | 0.769x | 0.863x | 1.12x | ndarray: the generator's draws |
| lognorm.rvs | 1000000 | 0.713x | 0.877x | 1.23x | own: SciPy's transform with correctly rounded elementary functions |
| binom.rvs | 1000000 | 0.875x | 0.879x | 1.01x | ndarray: the generator's draws |
| norm.rvs | 1000000 | 0.841x | 0.881x | 1.05x | ndarray: the generator's draws |
| norm.logpdf | 100000 | 0.678x | 0.896x | 1.32x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| vonmises.rvs | 1000000 | 0.895x | 0.897x | 1x | ndarray: the generator's draws |
| bootstrap | 1000 | 0.845x | 0.898x | 1.06x | own: scalar loops where NumPy has SIMD kernels |
| rayleigh.sf | 100000 | 0.87x | 0.901x | 1.04x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| uniform.rvs | 1000000 | 0.893x | 0.905x | 1.01x | ndarray: the generator's draws |
| rayleigh.pdf | 100000 | 0.923x | 0.922x | 0.999x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| halfnorm.logpdf | 100000 | 0.926x | 0.944x | 1.02x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| expon.sf | 100000 | 0.945x | 0.954x | 1.01x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| expon.isf | 20000 | 1.09x | 0.956x | 0.874x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| rayleigh.rvs | 1000000 | 0.971x | 0.959x | 0.988x | ndarray: the generator's draws |
| mean | 1000000 | 0.348x | 0.966x | 2.77x | own: scalar loops where NumPy has SIMD kernels |
| bernoulli.rvs | 1000000 | 0.949x | 0.971x | 1.02x | ndarray: the generator's draws |
| hypergeom.rvs | 1000000 | 0.998x | 0.975x | 0.977x | ndarray: the generator's draws |
| expon.logpdf | 100000 | 1.05x | 0.986x | 0.936x | own: double-double evaluation (≤ 0.5–2 ulp results) |
| kendalltau | 100000 | 0.657x | 0.993x | 1.51x | own |
| kde.resample | 100000 | 0.936x | 0.994x | 1.06x | own: compensated sums of exponentials per point |
| gumbel_r.logpdf | 100000 | 0.555x | 1x | 1.81x |  |
| chi2_contingency | 10000 | 0.963x | 1x | 1.04x |  |
| randint.rvs | 1000000 | 0.899x | 1.01x | 1.13x |  |
| ncx2.logpdf | 100000 | 0.976x | 1.02x | 1.05x |  |
| rayleigh.fit | 10000 | 0.306x | 1.06x | 3.46x |  |
| ttest_rel | 1000000 | 0.933x | 1.11x | 1.19x |  |
| gumbel_r.fit | 10000 | 0.488x | 1.16x | 2.39x |  |
| multinomial.logpmf | 100000 | 1.11x | 1.17x | 1.06x |  |
| vonmises.fit | 10000 | 1.07x | 1.19x | 1.11x |  |
| levene | 1000000 | 1.13x | 1.19x | 1.06x |  |
| zipf.pmf | 100000 | 0.0269x | 1.21x | 44.8x |  |
| triang.sf | 100000 | 1.19x | 1.21x | 1.01x |  |
| median | 1000000 | 1.2x | 1.21x | 1.01x |  |
| percentile | 1000000 | 1.22x | 1.22x | 1x |  |
| laplace.ppf | 20000 | 1.3x | 1.22x | 0.942x |  |
| triang.pdf | 100000 | 1.14x | 1.23x | 1.08x |  |
| laplace.isf | 20000 | 1.2x | 1.24x | 1.04x |  |
| jarque_bera | 1000000 | 0.739x | 1.25x | 1.69x |  |
| ttest_ind | 1000000 | 0.931x | 1.27x | 1.36x |  |
| normaltest | 1000000 | 0.75x | 1.27x | 1.69x |  |
| kurtosistest | 1000000 | 0.588x | 1.28x | 2.17x |  |
| kde.evaluate | 10000 | 1.22x | 1.28x | 1.05x |  |
| kurtosis | 1000000 | 0.597x | 1.29x | 2.17x |  |
| mad | 1000000 | 1.34x | 1.3x | 0.97x |  |
| differential_entropy | 100000 | 0.468x | 1.32x | 2.81x |  |
| skew | 1000000 | 1.36x | 1.33x | 0.974x |  |
| bartlett | 1000000 | 0.58x | 1.33x | 2.29x |  |
| ttest_1samp | 1000000 | 0.986x | 1.33x | 1.35x |  |
| tstd | 1000000 | 0.826x | 1.34x | 1.62x |  |
| histogram_bin_edges | 1000000 | 1.3x | 1.35x | 1.04x |  |
| tvar | 1000000 | 0.811x | 1.36x | 1.68x |  |
| skewtest | 1000000 | 1.05x | 1.43x | 1.36x |  |
| describe | 1000000 | 1.26x | 1.45x | 1.15x |  |
| zipf.logpmf | 100000 | 0.0274x | 1.45x | 53x |  |
| tsem | 1000000 | 0.955x | 1.45x | 1.52x |  |
| laplace.logpdf | 100000 | 1.46x | 1.48x | 1.02x |  |
| var | 1000000 | 0.351x | 1.52x | 4.33x |  |
| std | 1000000 | 0.467x | 1.53x | 3.28x |  |
| zscore | 1000000 | 1.31x | 1.55x | 1.18x |  |
| bernoulli.logpmf | 100000 | 1.62x | 1.6x | 0.99x |  |
| uniform.logpdf | 100000 | 1.54x | 1.6x | 1.04x |  |
| uniform.pdf | 100000 | 1.51x | 1.62x | 1.07x |  |
| binned_statistic | 1000000 | 1.66x | 1.65x | 0.992x |  |
| poisson.ppf | 20000 | 1.62x | 1.65x | 1.02x |  |
| poisson.isf | 20000 | 1.64x | 1.66x | 1.02x |  |
| zmap | 1000000 | 0.992x | 1.78x | 1.79x |  |
| randint.ppf | 20000 | 0.928x | 1.8x | 1.94x |  |
| mannwhitneyu | 100000 | 0.881x | 1.84x | 2.09x |  |
| kruskal | 100000 | 0.901x | 1.88x | 2.09x |  |
| quantile | 1000000 | 1.94x | 1.92x | 0.991x |  |
| wilcoxon | 100000 | 0.853x | 2.05x | 2.41x |  |
| bernoulli.sf | 100000 | 2.06x | 2.06x | 0.998x |  |
| randint.isf | 20000 | 0.977x | 2.07x | 2.11x |  |
| ks_1samp | 100000 | 0.844x | 2.1x | 2.48x |  |
| rankdata | 1000000 | 0.871x | 2.15x | 2.46x |  |
| variation | 1000000 | 0.607x | 2.18x | 3.58x |  |
| bernoulli.cdf | 100000 | 2.22x | 2.2x | 0.989x |  |
| histogram | 1000000 | 2.2x | 2.21x | 1.01x |  |
| spearmanr | 1000000 | 0.879x | 2.22x | 2.53x |  |
| trim_mean | 1000000 | 2.12x | 2.22x | 1.05x |  |
| trimboth | 1000000 | 2.24x | 2.25x | 1x |  |
| randint.pmf | 100000 | 2.36x | 2.32x | 0.984x |  |
| trim1 | 1000000 | 2.28x | 2.33x | 1.02x |  |
| randint.sf | 100000 | 2.33x | 2.46x | 1.06x |  |
| randint.cdf | 100000 | 2.49x | 2.56x | 1.03x |  |
| uniform.cdf | 100000 | 2.44x | 2.58x | 1.06x |  |
| uniform.sf | 100000 | 2.4x | 2.6x | 1.09x |  |
| uniform.isf | 20000 | 2.52x | 2.62x | 1.04x |  |
| uniform.ppf | 20000 | 2.4x | 2.81x | 1.17x |  |
| triang.cdf | 100000 | 1.13x | 2.85x | 2.53x |  |
| relfreq | 1000000 | 0.602x | 3.89x | 6.46x |  |
| halfnorm.fit | 10000 | 1.12x | 3.96x | 3.54x |  |
| ks_2samp | 100000 | 1.16x | 4.02x | 3.46x |  |
| kde.logpdf | 10000 | 4.05x | 4.08x | 1.01x |  |
| iqr | 1000000 | 4.44x | 4.3x | 0.968x |  |
| cumfreq | 1000000 | 0.626x | 4.86x | 7.76x |  |
| bernoulli.pmf | 100000 | 4.94x | 5.18x | 1.05x |  |
| vonmises.entropy | 1 | 6.19x | 6.28x | 1.01x |  |
| fisher_exact | 1 | 9.22x | 7.48x | 0.811x |  |
| hypergeom.ppf | 20000 | 7.51x | 7.87x | 1.05x |  |
| zipf.cdf | 100000 | 0.877x | 8x | 9.13x |  |
| hypergeom.isf | 20000 | 7.65x | 8.15x | 1.07x |  |
| nbinom.entropy | 1 | 8.46x | 8.18x | 0.966x |  |
| zipf.sf | 100000 | 0.872x | 8.96x | 10.3x |  |
| bernoulli.ppf | 20000 | 9.6x | 9.38x | 0.977x |  |
| bernoulli.isf | 20000 | 10.1x | 10.7x | 1.06x |  |
| zipf.isf | 20000 | 8.1x | 18.7x | 2.31x |  |
| hypergeom.entropy | 1 | 10.7x | 20.1x | 1.89x |  |
| ncx2.entropy | 1 | 24.2x | 25.4x | 1.05x |  |
| binom.entropy | 1 | 26x | 25.5x | 0.981x |  |
| poisson.entropy | 1 | 33.6x | 33.4x | 0.995x |  |
| vonmises.ppf | 20000 | 70.8x | 61.7x | 0.871x |  |
| vonmises.isf | 20000 | 73.2x | 69x | 0.942x |  |
| invgamma.entropy | 1 | 220x | 142x | 0.643x |  |
| triang.entropy | 1 | 360x | 150x | 0.417x |  |
| t.entropy | 1 | 303x | 169x | 0.555x |  |
| lognorm.entropy | 1 | 326x | 182x | 0.558x |  |
| f.entropy | 1 | 194x | 208x | 1.07x |  |
| reciprocal.entropy | 1 | 281x | 214x | 0.763x |  |
| halfnorm.entropy | 1 | 343x | 217x | 0.632x |  |
| weibull_min.entropy | 1 | 295x | 219x | 0.743x |  |
| zipf.entropy | 1 | 145x | 223x | 1.54x |  |
| pareto.entropy | 1 | 175x | 232x | 1.32x |  |
| bernoulli.entropy | 1 | 231x | 245x | 1.06x |  |
| lomax.entropy | 1 | 138x | 267x | 1.93x |  |
| geom.entropy | 1 | 262x | 277x | 1.06x |  |
| randint.entropy | 1 | 351x | 299x | 0.852x |  |
| cauchy.entropy | 1 | 360x | 303x | 0.841x |  |
| chi.entropy | 1 | 328x | 306x | 0.932x |  |
| gumbel_r.entropy | 1 | 232x | 310x | 1.34x |  |
| chi2.entropy | 1 | 314x | 315x | 1x |  |
| norm.entropy | 1 | 261x | 325x | 1.24x |  |
| expon.entropy | 1 | 273x | 326x | 1.2x |  |
| beta.entropy | 1 | 375x | 332x | 0.886x |  |
| gamma.entropy | 1 | 256x | 335x | 1.31x |  |
| laplace.entropy | 1 | 312x | 340x | 1.09x |  |
| rayleigh.entropy | 1 | 282x | 363x | 1.29x |  |
| uniform.entropy | 1 | 286x | 364x | 1.27x |  |
| logistic.entropy | 1 | 332x | 372x | 1.12x |  |
| zipf.ppf | 20000 | — | — | — | SciPy raises |

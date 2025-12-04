# Aurora-GLM: Bibliography and Mathematical References

This document provides comprehensive mathematical foundations and bibliographic references for all methods implemented in Aurora-GLM.

---

## Table of Contents

1. [Generalized Linear Models (GLM)](#generalized-linear-models-glm)
2. [Generalized Additive Models (GAM)](#generalized-additive-models-gam)
3. [Generalized Additive Mixed Models (GAMM)](#generalized-additive-mixed-models-gamm)
4. [Optimization Algorithms](#optimization-algorithms)
5. [Smoothing and Splines](#smoothing-and-splines)
6. [Distribution Families](#distribution-families)
7. [Statistical Inference](#statistical-inference)

---

## Generalized Linear Models (GLM)

### Theory

**Core Reference**:
- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman and Hall/CRC.

### Mathematical Framework

The GLM relates the mean $\mu_i = E[Y_i]$ to a linear predictor $\eta_i = \mathbf{x}_i^T \boldsymbol{\beta}$ via a link function $g(\cdot)$:

$$
g(\mu_i) = \eta_i = \mathbf{x}_i^T \boldsymbol{\beta}
$$

The response $Y_i$ follows an exponential family distribution:

$$
f(y_i; \theta_i, \phi) = \exp\left\{\frac{y_i\theta_i - b(\theta_i)}{a(\phi)} + c(y_i, \phi)\right\}
$$

where:
- $\theta_i$ is the canonical parameter (related to $\mu_i$ via $\mu_i = b'(\theta_i)$)
- $\phi$ is the dispersion parameter
- $b(\cdot)$ is the cumulant function
- $a(\phi)$ and $c(\cdot, \cdot)$ are known functions

**Variance function**: $\text{Var}(Y_i) = a(\phi) b''(\theta_i) = a(\phi) V(\mu_i)$

### Iteratively Reweighted Least Squares (IRLS)

**References**:
- Green, P. J. (1984). "Iteratively reweighted least squares for maximum likelihood estimation, and some robust and resistant alternatives." *Journal of the Royal Statistical Society: Series B*, 46(2), 149-192.
- Nelder, J. A., & Wedderburn, R. W. M. (1972). "Generalized linear models." *Journal of the Royal Statistical Society: Series A*, 135(3), 370-384.

**Algorithm**:

At iteration $t$, update coefficients via weighted least squares:

$$
\boldsymbol{\beta}^{(t+1)} = (\mathbf{X}^T \mathbf{W}^{(t)} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W}^{(t)} \mathbf{z}^{(t)}
$$

where:
- Working response: $z_i^{(t)} = \eta_i^{(t)} + (y_i - \mu_i^{(t)}) g'(\mu_i^{(t)})$
- Working weights: $w_i^{(t)} = \frac{1}{[g'(\mu_i^{(t)})]^2 V(\mu_i^{(t)})}$

**Convergence**: Iterate until $\|\boldsymbol{\beta}^{(t+1)} - \boldsymbol{\beta}^{(t)}\| < \epsilon$

### Deviance and Model Fit

**Deviance**:
$$
D = 2[\ell(\mathbf{y}; \mathbf{y}) - \ell(\boldsymbol{\mu}; \mathbf{y})]
$$

where $\ell(\mathbf{y}; \mathbf{y})$ is the saturated model log-likelihood.

**Information Criteria**:
- Akaike Information Criterion (AIC): $-2\ell(\hat{\boldsymbol{\beta}}) + 2p$
- Bayesian Information Criterion (BIC): $-2\ell(\hat{\boldsymbol{\beta}}) + p \log(n)$

**References**:
- Akaike, H. (1974). "A new look at the statistical model identification." *IEEE Transactions on Automatic Control*, 19(6), 716-723.

**Model Selection and Statistical Learning**:
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-84858-7
  (Comprehensive coverage of model selection, cross-validation, and information criteria)

---

## Generalized Additive Models (GAM)

### Theory

**Core References**:
- Hastie, T., & Tibshirani, R. (1990). *Generalized Additive Models*. Chapman and Hall/CRC.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). CRC Press.

### Mathematical Framework

GAM extends GLM by allowing smooth functions of predictors:

$$
g(E[Y_i]) = \beta_0 + f_1(x_{i1}) + f_2(x_{i2}) + \cdots + f_p(x_{ip})
$$

where $f_j(\cdot)$ are smooth functions represented using basis expansions:

$$
f_j(x) = \sum_{k=1}^{K_j} \beta_{jk} \phi_{jk}(x)
$$

**Penalized Likelihood**:

$$
\ell_p(\boldsymbol{\beta}) = \ell(\boldsymbol{\beta}) - \frac{1}{2}\sum_{j=1}^p \lambda_j \boldsymbol{\beta}_j^T \mathbf{S}_j \boldsymbol{\beta}_j
$$

where:
- $\lambda_j$ is the smoothing parameter for $f_j$
- $\mathbf{S}_j$ is the penalty matrix (e.g., second derivative penalty)

### B-Spline Basis Functions

**References**:
- de Boor, C. (2001). *A Practical Guide to Splines* (Revised ed.). Springer.
- Eilers, P. H. C., & Marx, B. D. (1996). "Flexible smoothing with B-splines and penalties." *Statistical Science*, 11(2), 89-121.

**Cox-de Boor Recursion**:

$$
B_{i,0}(x) = \begin{cases} 1 & \text{if } t_i \leq x < t_{i+1} \\ 0 & \text{otherwise} \end{cases}
$$

$$
B_{i,k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i,k-1}(x) + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1,k-1}(x)
$$

### Natural Cubic Splines

**Reference**: Green, P. J., & Silverman, B. W. (1993). *Nonparametric Regression and Generalized Linear Models: A Roughness Penalty Approach*. Chapman and Hall/CRC.

**Penalty**: Integrated squared second derivative:

$$
J(f) = \int [f''(x)]^2 dx
$$

For cubic splines with knots $\{t_j\}$:

$$
f(x) = \sum_{j=0}^3 \beta_j x^j + \sum_{k=1}^{K-2} \gamma_k (x - t_k)_+^3
$$

### Thin Plate Splines

**Reference**: Duchon, J. (1977). "Splines minimizing rotation-invariant semi-norms in Sobolev spaces." In *Constructive Theory of Functions of Several Variables* (pp. 85-100). Springer.

**Mathematical Formulation** (for $d=2$ dimensions):

Minimize:
$$
\iint \left[\left(\frac{\partial^2 f}{\partial x^2}\right)^2 + 2\left(\frac{\partial^2 f}{\partial x \partial y}\right)^2 + \left(\frac{\partial^2 f}{\partial y^2}\right)^2\right] dx\, dy
$$

**Basis functions**: Radial basis functions $\eta(\|\mathbf{x} - \mathbf{x}_k\|)$ where:

$$
\eta(r) = \begin{cases}
r^2 \log(r) & d = 2 \\
r & d = 3
\end{cases}
$$

### Smoothing Parameter Selection

#### Generalized Cross-Validation (GCV)

**Reference**: Craven, P., & Wahba, G. (1978). "Smoothing noisy data with spline functions." *Numerische Mathematik*, 31(4), 377-403.

$$
\text{GCV}(\lambda) = \frac{n \sum_{i=1}^n (y_i - \hat{f}_\lambda(x_i))^2}{[n - \text{tr}(\mathbf{H}_\lambda)]^2}
$$

where $\mathbf{H}_\lambda$ is the smoother (hat) matrix.

#### Restricted Maximum Likelihood (REML)

**Reference**: Patterson, H. D., & Thompson, R. (1971). "Recovery of inter-block information when block sizes are unequal." *Biometrika*, 58(3), 545-554.

**Wood's REML for GAM**:
- Wood, S. N. (2011). "Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models." *Journal of the Royal Statistical Society: Series B*, 73(1), 3-36.

$$
-2\ell_R(\lambda) = \log|\mathbf{X}^T\mathbf{X}| + \log|\mathbf{S} + \lambda\mathbf{I}| + n\log(\|\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}\|^2)
$$

---

## Generalized Additive Mixed Models (GAMM)

### Gaussian GAMM with REML

**Core Reference**:
- Wood, S. N. (2006). *Generalized Additive Models: An Introduction with R*. Chapman and Hall/CRC. Chapter 6.

### Mathematical Framework

The linear mixed model representation of GAM:

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{Z}\mathbf{b} + \boldsymbol{\epsilon}
$$

where:
- $\mathbf{X}$ is the design matrix for fixed effects (parametric + smooth)
- $\mathbf{Z}$ is the design matrix for random effects
- $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Psi})$ are random effects
- $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})$ is residual error

**Mixed Model Equations** (Henderson's equations):

$$
\begin{bmatrix}
\mathbf{X}^T\mathbf{X} & \mathbf{X}^T\mathbf{Z} \\
\mathbf{Z}^T\mathbf{X} & \mathbf{Z}^T\mathbf{Z} + \boldsymbol{\Psi}^{-1}
\end{bmatrix}
\begin{bmatrix}
\hat{\boldsymbol{\beta}} \\
\hat{\mathbf{b}}
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{X}^T\mathbf{y} \\
\mathbf{Z}^T\mathbf{y}
\end{bmatrix}
$$

**Reference**: Henderson, C. R. (1975). "Best linear unbiased estimation and prediction under a selection model." *Biometrics*, 31(2), 423-447.

**REML Estimation**:

Maximize the restricted log-likelihood:

$$
\ell_R(\boldsymbol{\theta}) = -\frac{1}{2}\left[\log|\mathbf{V}| + \log|\mathbf{X}^T\mathbf{V}^{-1}\mathbf{X}| + \mathbf{r}^T\mathbf{V}^{-1}\mathbf{r}\right]
$$

where $\mathbf{V} = \mathbf{Z}\boldsymbol{\Psi}\mathbf{Z}^T + \sigma^2\mathbf{I}$ and $\mathbf{r} = \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}$.

### Non-Gaussian GAMM with Penalized Quasi-Likelihood (PQL)

**Core References**:
- Breslow, N. E., & Clayton, D. G. (1993). "Approximate inference in generalized linear mixed models." *Journal of the American Statistical Association*, 88(421), 9-25.
- Schall, R. (1991). "Estimation in generalized linear models with random effects." *Biometrika*, 78(4), 719-727.

### PQL Algorithm

**Mathematical Formulation**:

For non-Gaussian response $Y_{ij}$ with link $g(\cdot)$:

$$
g(E[Y_{ij} | \mathbf{b}_i]) = \mathbf{x}_{ij}^T\boldsymbol{\beta} + \mathbf{z}_{ij}^T\mathbf{b}_i
$$

where $\mathbf{b}_i \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Psi})$.

**Iterative Procedure**:

1. **Initialize**: $\boldsymbol{\beta}^{(0)} = \mathbf{0}$, $\mathbf{b}^{(0)} = \mathbf{0}$, $\boldsymbol{\Psi}^{(0)} = \mathbf{I}$

2. **Inner loop** (fix $\boldsymbol{\Psi}$, update $\boldsymbol{\beta}, \mathbf{b}$):

   a. Compute linear predictor: $\eta_{ij} = \mathbf{x}_{ij}^T\boldsymbol{\beta} + \mathbf{z}_{ij}^T\mathbf{b}_i$

   b. Compute mean: $\mu_{ij} = g^{-1}(\eta_{ij})$

   c. **Working response**:
   $$
   z_{ij} = \eta_{ij} + (y_{ij} - \mu_{ij}) \left(\frac{d\mu}{d\eta}\right)_{\mu=\mu_{ij}}^{-1}
   $$

   d. **Working weights**:
   $$
   w_{ij} = \left[\left(\frac{d\mu}{d\eta}\right)_{\mu=\mu_{ij}}\right]^2 \frac{1}{V(\mu_{ij})}
   $$

   e. Solve weighted mixed model equations:

$$\begin{bmatrix} \mathbf{X}^T\mathbf{W}\mathbf{X} & \mathbf{X}^T\mathbf{W}\mathbf{Z} \\ \mathbf{Z}^T\mathbf{W}\mathbf{X} & \mathbf{Z}^T\mathbf{W}\mathbf{Z} + \boldsymbol{\Psi}^{-1} \end{bmatrix} \begin{bmatrix} \hat{\boldsymbol{\beta}} \\ \hat{\mathbf{b}} \end{bmatrix} = \begin{bmatrix} \mathbf{X}^T\mathbf{W}\mathbf{z} \\ \mathbf{Z}^T\mathbf{W}\mathbf{z} \end{bmatrix}$$

3. **Outer loop** (update $\boldsymbol{\Psi}$):

   a. Compute empirical covariance:
   $$
   \hat{\boldsymbol{\Psi}} = \frac{1}{m}\sum_{i=1}^m \hat{\mathbf{b}}_i \hat{\mathbf{b}}_i^T
   $$

   b. Apply shrinkage for numerical stability

4. **Convergence**: Stop when $\|\boldsymbol{\Psi}^{(t+1)} - \boldsymbol{\Psi}^{(t)}\| / \|\boldsymbol{\Psi}^{(t)}\| < \epsilon$

**Properties**:
- PQL provides asymptotically unbiased estimates when cluster sizes are large
- Bias can be substantial for small cluster sizes or binary data with rare events
- Computationally efficient compared to Laplace approximation or quadrature

**Limitations** (see Lin & Breslow, 1996):
- Lin, X., & Breslow, N. E. (1996). "Bias correction in generalized linear mixed models with multiple components of dispersion." *Journal of the American Statistical Association*, 91(435), 1007-1016.

### Laplace Approximation (Planned)

**Reference**: Breslow, N. E., & Lin, X. (1995). "Bias correction in generalised linear mixed models with a single component of dispersion." *Biometrika*, 82(1), 81-91.

**Marginal likelihood approximation**:

$$
\ell(\boldsymbol{\beta}, \boldsymbol{\theta}) \approx \log p(\mathbf{y} | \hat{\mathbf{b}}, \boldsymbol{\beta}) + \log p(\hat{\mathbf{b}} | \boldsymbol{\theta}) - \frac{1}{2}\log|\mathbf{H}(\hat{\mathbf{b}})|
$$

where $\mathbf{H}(\hat{\mathbf{b}})$ is the Hessian of the log-posterior at the mode.

---

## Optimization Algorithms

### Newton-Raphson Method

**Reference**: Dennis, J. E., & Schnabel, R. B. (1996). *Numerical Methods for Unconstrained Optimization and Nonlinear Equations*. SIAM.

**Update**:
$$
\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - [\nabla^2 f(\boldsymbol{\theta}^{(t)})]^{-1} \nabla f(\boldsymbol{\theta}^{(t)})
$$

**Fisher Scoring** (for GLM):

$$
\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} + [\mathcal{I}(\boldsymbol{\beta}^{(t)})]^{-1} \nabla \ell(\boldsymbol{\beta}^{(t)})
$$

where $\mathcal{I}(\boldsymbol{\beta}) = E[-\nabla^2 \ell(\boldsymbol{\beta})]$ is the Fisher information matrix.

### L-BFGS Algorithm

**Reference**: Liu, D. C., & Nocedal, J. (1989). "On the limited memory BFGS method for large scale optimization." *Mathematical Programming*, 45(1-3), 503-528.

Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm:
- Approximates Hessian using $m$ recent gradient vectors
- Memory requirement: $O(nm)$ instead of $O(n^2)$
- Suitable for high-dimensional problems

**Implementation reference**: Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer.

---

## Distribution Families

### Exponential Family Theory

**Reference**: Barndorff-Nielsen, O. (2014). *Information and Exponential Families in Statistical Theory*. John Wiley & Sons.

### Gaussian (Normal) Family

**Density**:
$$
f(y; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left\{-\frac{(y-\mu)^2}{2\sigma^2}\right\}
$$

- Canonical link: $g(\mu) = \mu$ (identity)
- Variance function: $V(\mu) = 1$

### Poisson Family

**Reference**: Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.). Cambridge University Press.

**Density**:
$$
f(y; \mu) = \frac{\mu^y e^{-\mu}}{y!}, \quad y = 0, 1, 2, \ldots
$$

- Canonical link: $g(\mu) = \log(\mu)$
- Variance function: $V(\mu) = \mu$

### Binomial Family

**Density**:
$$
f(y; n, p) = \binom{n}{y} p^y (1-p)^{n-y}, \quad y = 0, 1, \ldots, n
$$

- Mean: $\mu = np$
- Canonical link: $g(\mu) = \log\left(\frac{\mu}{n-\mu}\right) = \text{logit}(p)$
- Variance function: $V(\mu) = \mu(1 - \mu/n)$

### Gamma Family

**Reference**: Dunn, P. K., & Smyth, G. K. (2018). *Generalized Linear Models with Examples in R*. Springer.

**Density**:
$$
f(y; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} y^{\alpha-1} e^{-\beta y}, \quad y > 0
$$

- Mean: $\mu = \alpha/\beta$
- Canonical link: $g(\mu) = -1/\mu$ (inverse)
- Variance function: $V(\mu) = \mu^2$

### Negative Binomial Family (Planned)

**Reference**: Hilbe, J. M. (2011). *Negative Binomial Regression* (2nd ed.). Cambridge University Press.

**Density** (NB2 parameterization):
$$
f(y; \mu, \theta) = \frac{\Gamma(y + \theta)}{\Gamma(\theta) y!} \left(\frac{\theta}{\theta + \mu}\right)^\theta \left(\frac{\mu}{\theta + \mu}\right)^y
$$

- Mean: $E[Y] = \mu$
- Variance: $\text{Var}(Y) = \mu + \mu^2/\theta$
- Used for overdispersed count data

---

## Statistical Inference

### Wald Tests

**Reference**: Wald, A. (1943). "Tests of statistical hypotheses concerning several parameters when the number of observations is large." *Transactions of the American Mathematical Society*, 54(3), 426-482.

**Test statistic** for $H_0: \mathbf{C}\boldsymbol{\beta} = \mathbf{d}$:

$$
W = (\mathbf{C}\hat{\boldsymbol{\beta}} - \mathbf{d})^T [\mathbf{C} \text{Cov}(\hat{\boldsymbol{\beta}}) \mathbf{C}^T]^{-1} (\mathbf{C}\hat{\boldsymbol{\beta}} - \mathbf{d}) \sim \chi^2_q
$$

where $q = \text{rank}(\mathbf{C})$.

### Likelihood Ratio Tests

**Reference**: Wilks, S. S. (1938). "The large-sample distribution of the likelihood ratio for testing composite hypotheses." *The Annals of Mathematical Statistics*, 9(1), 60-62.

**Test statistic**:
$$
\Lambda = -2[\ell(\boldsymbol{\beta}_0) - \ell(\hat{\boldsymbol{\beta}})] \sim \chi^2_q
$$

### Confidence Intervals

**Wald-type intervals**:
$$
\hat{\beta}_j \pm z_{\alpha/2} \sqrt{\text{Var}(\hat{\beta}_j)}
$$

**Reference**: Cox, D. R., & Hinkley, D. V. (1979). *Theoretical Statistics*. CRC Press.

---

## Software Implementation References

### R Packages (for validation)

1. **mgcv** (GAM/GAMM):
   - Wood, S. N. (2011). "Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models." *Journal of the Royal Statistical Society: Series B*, 73(1), 3-36.

2. **lme4** (GLMM):
   - Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). "Fitting linear mixed-effects models using lme4." *Journal of Statistical Software*, 67(1), 1-48.

3. **statsmodels** (Python GLM):
   - Seabold, S., & Perktold, J. (2010). "statsmodels: Econometric and statistical modeling with python." In *9th Python in Science Conference*.

### Numerical Stability

**References**:
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.

**Key techniques used in Aurora-GLM**:
- Cholesky decomposition for symmetric positive-definite systems
- QR decomposition for rank-deficient cases
- Log-sum-exp trick for numerical stability in exponentials
- Iterative refinement for ill-conditioned problems

---

## Citation for Aurora-GLM

If you use Aurora-GLM in your research, please cite:

```bibtex
@software{aurora_glm_2025,
  author = {Arias, Lucy E.},
  title = {Aurora-GLM: A Python Framework for Generalized Linear and Additive Mixed Models},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Matcraft94/Aurora-GLM},
  version = {0.5.0}
}
```

---

## Additional Reading

### Books

1. **Faraway, J. J. (2016).** *Extending the Linear Model with R: Generalized Linear, Mixed Effects and Nonparametric Regression Models* (2nd ed.). CRC Press.

2. **Ruppert, D., Wand, M. P., & Carroll, R. J. (2003).** *Semiparametric Regression*. Cambridge University Press.

3. **Pinheiro, J. C., & Bates, D. M. (2000).** *Mixed-Effects Models in S and S-PLUS*. Springer. https://doi.org/10.1007/b98882

4. **Schumaker, L. L. (2007).** *Spline Functions: Basic Theory* (3rd ed.). Cambridge University Press. (Comprehensive mathematical treatment of spline theory)

5. **Prautzsch, H., Boehm, W., & Paluszny, M. (2002).** *Bézier and B-Spline Techniques*. Springer. (Computer graphics perspective on splines)

6. **Gelman, A., & Hill, J. (2007).** *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press. (Multilevel modeling and visualization)

### Survey Articles

1. **Diggle, P., Heagerty, P., Liang, K. Y., & Zeger, S. (2002).** *Analysis of Longitudinal Data* (2nd ed.). Oxford University Press.

2. **Verbeke, G., & Molenberghs, G. (2009).** *Linear Mixed Models for Longitudinal Data*. Springer.

3. **Wood, S. N. (2020).** "Inference and computation with generalized additive models and their extensions." *TEST*, 29(2), 307-339.

### Additional Technical References

1. **Cox, M. G. (1972).** "The numerical evaluation of B-splines." *IMA Journal of Applied Mathematics*, 10(2), 134-149. https://doi.org/10.1093/imamat/10.2.134 (Cox-de Boor recursion)

2. **de Boor, C. (1972).** "On calculating with B-splines." *Journal of Approximation Theory*, 6(1), 50-62. https://doi.org/10.1016/0021-9045(72)90080-9

3. **Griewank, A., & Walther, A. (2008).** *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation* (2nd ed.). SIAM. (Automatic differentiation theory)

4. **Dennis, J. E., & Moré, J. J. (1977).** "Quasi-Newton methods, motivation and theory." *SIAM Review*, 19(1), 46-89. https://doi.org/10.1137/1019005 (Line search methods)

5. **Wood, S. N. (2003).** "Thin plate regression splines." *Journal of the Royal Statistical Society: Series B*, 65(1), 95-114. (Thin plate splines for GAM)

6. **Hodges, J. S., & Sargent, D. J. (2001).** "Counting degrees of freedom in hierarchical and other richly-parameterized models." *Biometrika*, 88(2), 367-379. https://doi.org/10.1093/biomet/88.2.367 (Effective degrees of freedom)

7. **Nakagawa, S., & Schielzeth, H. (2013).** "A general and simple method for obtaining R² from generalized linear mixed-effects models." *Methods in Ecology and Evolution*, 4(2), 133-142. (R² for mixed models)

### Robust Standard Errors

1. **White, H. (1980).** "A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity." *Econometrica*, 48(4), 817-838. (HC0 estimator)

2. **MacKinnon, J. G., & White, H. (1985).** "Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties." *Journal of Econometrics*, 29(3), 305-325. (HC1, HC2, HC3 estimators)

3. **Cribari-Neto, F. (2004).** "Asymptotic inference under heteroskedasticity of unknown form." *Computational Statistics & Data Analysis*, 45(2), 215-233. (HC4, HC4m, HC5 estimators)

---

*Last updated: 2025-12-03*
*Aurora-GLM Version: 0.6.1*

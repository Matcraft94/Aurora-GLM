# Aurora-GLM: Contexto del Proyecto y Objetivos de Desarrollo

## 📋 Información General

**Proyecto**: Aurora-GLM  
**Repositorio**: https://github.com/Matcraft94/Aurora-GLM  
**Autor**: Lucy E. Arias (@matcraf94)  
**Versión actual**: 0.1.0-dev  
**Estado**: Fase 2 en progreso (Core completado, GLM en desarrollo)  
**Python**: 3.10+  

---

## 🎯 Visión del Proyecto

### Objetivo Principal

Crear un framework de modelado estadístico en Python que sea:

1. **Científicamente riguroso**: Implementaciones correctas validadas contra R (mgcv) y statsmodels
2. **Alto rendimiento**: Competitivo o superior a alternativas existentes, con soporte GPU
3. **Extensible**: Usuarios pueden agregar distribuciones, funciones de enlace, y algoritmos propios
4. **Multi-backend**: Soporte transparente para NumPy, PyTorch, y JAX
5. **Modular y funcional**: Diseño limpio que favorece composición sobre herencia pesada

### Alcance del Framework

**Modelos soportados** (objetivo final):
- **GLM**: Modelos Lineales Generalizados
- **GAM**: Modelos Aditivos Generalizados (con splines)
- **GAMM**: Modelos Aditivos Generalizados Mixtos (con efectos aleatorios)

**Casos de uso**:
- Investigación académica (ecología, epidemiología, ciencias sociales)
- Industria farmacéutica (ensayos clínicos)
- Análisis financiero (credit scoring, risk modeling)
- Machine learning con fundamentos estadísticos sólidos

---

## ✅ Estado Actual de Implementación

### Fase 1: Core Numérico - COMPLETADO ✅

**Backend System** (100% completo):
```
aurora/core/backends/
├── __init__.py           ✅ Backend registry
├── jax_backend.py        ✅ JAX implementation
└── pytorch_backend.py    ✅ PyTorch implementation
```

**Características**:
- ✅ Abstracción de backends con protocolo unificado
- ✅ Soporte JAX (JIT, grad, vmap, device_put)
- ✅ Soporte PyTorch con compatibilidad completa
- ✅ Registro de backends custom
- ✅ Detección automática de arrays (NumPy/PyTorch/JAX)

**Type System** (100% completo):
```
aurora/core/types.py      ✅ Protocols y type aliases
```

**Características**:
- ✅ Protocolos para Distribution, Link, Optimizer
- ✅ Type aliases para Array, Scalar, Shape
- ✅ ArrayLike protocol
- ✅ OptimizationResult protocol

**Optimization Algorithms** (100% completo):
```
aurora/core/optimization/
├── __init__.py           ✅ Unified optimize() interface
├── result.py             ✅ OptimizationResult dataclass
├── newton.py             ✅ Newton-Raphson
├── irls.py               ✅ IRLS (Iteratively Reweighted Least Squares)
└── lbfgs.py              ✅ L-BFGS with line search
```

**Características**:
- ✅ Newton-Raphson con Hessian automático
- ✅ IRLS para GLM
- ✅ L-BFGS con two-loop recursion
- ✅ Line search (Armijo backtracking)
- ✅ Callbacks para monitoreo
- ✅ Convergencia robusta

**Distribution Families** (80% completo):
```
aurora/distributions/families/
├── __init__.py           ✅ Exports
├── gaussian.py           ✅ Normal/Gaussian
├── poisson.py            ✅ Poisson
├── binomial.py           ✅ Binomial
└── gamma.py              ✅ Gamma
```

**Distribuciones implementadas**:
- ✅ Gaussian (Normal) - completa
- ✅ Poisson - completa
- ✅ Binomial - completa
- ✅ Gamma - completa

**Distribuciones pendientes** (Fase 5):
- ⏳ Inverse Gaussian
- ⏳ Negative Binomial
- ⏳ Beta
- ⏳ Tweedie
- ⏳ Exponential
- ⏳ Multinomial

**Link Functions** (70% completo):
```
aurora/distributions/links/
├── __init__.py           ✅ Exports
├── identity.py           ✅ Identity link
├── log.py                ✅ Log link
├── logit.py              ✅ Logit link
├── inverse.py            ✅ Inverse link
└── cloglog.py            ✅ Complementary log-log
```

**Links implementadas**:
- ✅ Identity: g(μ) = μ
- ✅ Log: g(μ) = log(μ)
- ✅ Logit: g(μ) = log(μ/(1-μ))
- ✅ Inverse: g(μ) = 1/μ
- ✅ CLogLog: g(μ) = log(-log(1-μ))

**Links pendientes** (Fase 5):
- ⏳ Probit: g(μ) = Φ⁻¹(μ)
- ⏳ Square root: g(μ) = √μ
- ⏳ Power: g(μ) = μᵖ

**Array Namespace Utilities** (100% completo):
```
aurora/distributions/_utils.py  ✅ Multi-backend support
```

**Características**:
- ✅ namespace() para detectar array type
- ✅ as_namespace_array() para conversiones
- ✅ Soporte transparente NumPy/PyTorch/JAX
- ✅ Operaciones agnósticas al backend

---

## 🚧 Fase 2: GLM Básico - EN PROGRESO (30% completo)

### Objetivo de la Fase 2

Implementar modelos GLM funcionales con:
- Fitting usando IRLS
- Predicción
- Inferencia básica (intervalos de confianza, p-values)
- Diagnósticos del modelo
- Métricas de evaluación

### Estado Actual

**Estructura creada**:
```
aurora/models/
├── base/
│   ├── __init__.py           ✅ Exports
│   └── result.py             🚧 ModelResult (esqueleto)
├── glm/
│   ├── __init__.py           🚧 fit_glm (NO implementado)
│   └── fitting.py            ❌ Por crear
├── gam/                      ⏳ Fase 3
└── gamm/                     ⏳ Fase 4
```

### Componentes a Implementar (PRIORIDAD ALTA)

#### 1. GLM Fitting Function

**Archivo**: `aurora/models/glm/fitting.py`

**Función principal**:
```python
def fit_glm(
    X: Array,
    y: Array,
    *,
    family: str | Family = "gaussian",
    link: str | LinkFunction | None = None,
    weights: Array | None = None,
    offset: Array | None = None,
    backend: str = "jax",
    max_iter: int = 25,
    tol: float = 1e-8,
    fit_intercept: bool = True,
) -> GLMResult:
    """
    Fit a Generalized Linear Model using IRLS.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Design matrix
    y : array-like, shape (n_samples,)
        Response variable
    family : str or Family
        Distribution family ('gaussian', 'poisson', 'binomial', 'gamma')
    link : str or LinkFunction, optional
        Link function. If None, uses canonical link for family
    weights : array-like, optional
        Observation weights
    offset : array-like, optional
        Offset term
    backend : str
        Backend to use ('jax', 'pytorch', 'numpy')
    max_iter : int
        Maximum IRLS iterations
    tol : float
        Convergence tolerance
    fit_intercept : bool
        Whether to fit intercept
        
    Returns
    -------
    GLMResult
        Fitted model result with parameters, predictions, inference
        
    Examples
    --------
    >>> X = np.random.randn(100, 3)
    >>> y = np.random.poisson(np.exp(X[:, 0] * 0.5))
    >>> result = fit_glm(X, y, family='poisson', link='log')
    >>> result.coef_
    array([0.52, -0.03, 0.01])
    """
```

**Pasos del algoritmo IRLS**:
1. Inicializar μ usando `family.initialize(y)`
2. Para cada iteración:
   - Calcular η = link(μ)
   - Calcular working response: z = η + (y - μ) * link.derivative(μ)
   - Calcular weights: w = 1 / (link.derivative(μ)² * family.variance(μ))
   - Resolver: β = (X'WX)⁻¹ X'Wz
   - Actualizar: η = Xβ, μ = link.inverse(η)
   - Verificar convergencia

**Criterio de convergencia**:
```python
# Convergencia basada en cambio en deviance
dev_change = abs(deviance_new - deviance_old) / (abs(deviance_old) + 0.1)
converged = dev_change < tol
```

#### 2. GLMResult Class

**Archivo**: `aurora/models/base/result.py`

**Estructura requerida**:
```python
@dataclass
class GLMResult:
    """Container for GLM fitting results."""
    
    # Fitted parameters
    coef_: Array              # Coefficients (excluding intercept if fit)
    intercept_: float | None  # Intercept (if fitted)
    
    # Model specification
    family: Family
    link: LinkFunction
    
    # Fitted values
    mu_: Array                # Fitted means
    eta_: Array               # Linear predictor
    
    # Model statistics
    deviance_: float          # Deviance
    null_deviance_: float     # Null model deviance
    aic_: float               # Akaike Information Criterion
    bic_: float               # Bayesian Information Criterion
    
    # Convergence info
    n_iter_: int              # Number of iterations
    converged_: bool          # Whether converged
    
    # Inference (optional, computed on demand)
    _coef_cov: Array | None = None     # Coefficient covariance matrix
    _std_errors: Array | None = None    # Standard errors
    _p_values: Array | None = None      # P-values
    
    # Data info
    _X: Array | None = None
    _y: Array | None = None
    _weights: Array | None = None
    
    @property
    def std_errors_(self) -> Array:
        """Standard errors of coefficients."""
        if self._std_errors is None:
            self._compute_inference()
        return self._std_errors
    
    @property
    def p_values_(self) -> Array:
        """P-values for coefficients."""
        if self._p_values is None:
            self._compute_inference()
        return self._p_values
    
    @property
    def coef_cov_(self) -> Array:
        """Covariance matrix of coefficients."""
        if self._coef_cov is None:
            self._compute_inference()
        return self._coef_cov
    
    def _compute_inference(self):
        """Compute standard errors and p-values."""
        # Fisher information matrix: I = X'WX
        # where W = diag(1 / (link'(μ)² * Var(μ)))
        # Covariance: Cov(β) = I⁻¹
        pass
    
    def predict(
        self,
        X_new: Array,
        type: str = "response",
        interval: str | None = None,
        level: float = 0.95,
    ) -> Array | tuple[Array, Array, Array]:
        """
        Make predictions on new data.
        
        Parameters
        ----------
        X_new : array-like
            New design matrix
        type : str
            Type of prediction: 'response', 'link', or 'terms'
        interval : str, optional
            Type of interval: 'confidence' or 'prediction'
        level : float
            Confidence level for intervals
            
        Returns
        -------
        predictions : Array
            Predictions (and intervals if requested)
        """
        pass
    
    def summary(self) -> str:
        """Print summary of fit."""
        pass
    
    def plot_diagnostics(self):
        """Generate diagnostic plots."""
        pass
```

#### 3. Inference Module

**Archivos a crear**:
```
aurora/inference/
├── __init__.py
├── confidence.py         # Confidence intervals
├── hypothesis.py         # Hypothesis tests
└── diagnostics.py        # Model diagnostics
```

**Funciones clave**:

```python
# confidence.py
def confidence_intervals(
    result: GLMResult,
    level: float = 0.95,
    method: str = "wald"
) -> tuple[Array, Array]:
    """
    Compute confidence intervals for coefficients.
    
    Parameters
    ----------
    result : GLMResult
        Fitted model
    level : float
        Confidence level (default 0.95)
    method : str
        Method: 'wald' or 'profile'
        
    Returns
    -------
    lower, upper : Array
        Lower and upper bounds
    """
    pass

# hypothesis.py
def wald_test(
    result: GLMResult,
    hypothesis: str | Array,
) -> dict:
    """
    Perform Wald test for coefficient hypotheses.
    
    Parameters
    ----------
    result : GLMResult
        Fitted model
    hypothesis : str or Array
        Hypothesis to test (e.g., "x1 = 0" or contrast matrix)
        
    Returns
    -------
    dict with keys: statistic, p_value, df
    """
    pass

def likelihood_ratio_test(
    result_full: GLMResult,
    result_reduced: GLMResult,
) -> dict:
    """
    Likelihood ratio test comparing nested models.
    
    Returns
    -------
    dict with keys: statistic, p_value, df
    """
    pass

# diagnostics.py
def residuals(
    result: GLMResult,
    type: str = "deviance"
) -> Array:
    """
    Compute residuals.
    
    Parameters
    ----------
    type : str
        Type of residuals: 'deviance', 'pearson', 'working', 'response'
    """
    pass

def influential_observations(
    result: GLMResult
) -> dict[str, Array]:
    """
    Compute measures of influence.
    
    Returns
    -------
    dict with keys: cooks_d, leverage, dfbetas
    """
    pass
```

#### 4. Validation Metrics

**Archivo**: `aurora/validation/metrics/__init__.py`

**Métricas a implementar**:
```python
def deviance(y_true: Array, y_pred: Array, family: Family) -> float:
    """Compute deviance."""
    pass

def aic(deviance: float, n_params: int) -> float:
    """Akaike Information Criterion."""
    return deviance + 2 * n_params

def bic(deviance: float, n_params: int, n_samples: int) -> float:
    """Bayesian Information Criterion."""
    return deviance + np.log(n_samples) * n_params

def pseudo_r_squared(
    deviance: float,
    null_deviance: float,
    method: str = "mcfadden"
) -> float:
    """
    Pseudo R² measures.
    
    Methods: 'mcfadden', 'cox_snell', 'nagelkerke'
    """
    pass

def concordance_index(y_true: Array, y_pred: Array) -> float:
    """C-statistic for binary outcomes."""
    pass
```

---

## 📅 Plan de Implementación Detallado

### Semana 1: GLM Fitting Core

**Día 1-2**: Implementar `fit_glm()` con IRLS
- [ ] Crear `aurora/models/glm/fitting.py`
- [ ] Implementar algoritmo IRLS completo
- [ ] Manejo de weights y offset
- [ ] Convergencia robusta
- [ ] Tests con datos sintéticos

**Día 3**: Implementar `GLMResult`
- [ ] Completar dataclass en `result.py`
- [ ] Propiedades lazy para inferencia
- [ ] Método `predict()`
- [ ] Tests de predicción

**Día 4-5**: Inferencia básica
- [ ] Implementar cálculo de covariance matrix
- [ ] Standard errors
- [ ] Wald confidence intervals
- [ ] P-values
- [ ] Tests de inferencia

### Semana 2: Diagnósticos y Validación

**Día 1-2**: Residuales y diagnósticos
- [ ] Implementar tipos de residuales
- [ ] Cook's distance
- [ ] Leverage
- [ ] DFBETAs
- [ ] Tests

**Día 3**: Métricas de evaluación
- [ ] Deviance
- [ ] AIC/BIC
- [ ] Pseudo R²
- [ ] Concordance index
- [ ] Tests

**Día 4-5**: Integración y tests
- [ ] Tests end-to-end completos
- [ ] Validación contra statsmodels
- [ ] Validación contra R (glm)
- [ ] Benchmarks de performance
- [ ] Documentación

### Semana 3: Casos de uso y ejemplos

**Día 1-2**: Ejemplos documentados
- [ ] Regresión Poisson (count data)
- [ ] Regresión logística (binomial)
- [ ] Regresión Gamma (positive continuous)
- [ ] Casos con weights y offset
- [ ] Notebooks Jupyter

**Día 3**: Visualización
- [ ] Diagnostic plots básicos
- [ ] Partial residual plots
- [ ] QQ plots
- [ ] Influence plots

**Día 4-5**: Documentación y cleanup
- [ ] README con ejemplos actualizados
- [ ] Docstrings completos
- [ ] Tutorial básico
- [ ] Preparar release 0.2.0

---

## 🎯 Criterios de Éxito para Fase 2

### Funcionalidad (Debe)
- [ ] `fit_glm()` funciona con todas las familias implementadas
- [ ] Predicción funciona correctamente
- [ ] Intervalos de confianza correctos
- [ ] P-values correctos
- [ ] Residuales implementados

### Validación (Debe)
- [ ] Resultados coinciden con statsmodels (dentro de tolerancia)
- [ ] Resultados coinciden con R glm() (dentro de tolerancia)
- [ ] Tests pasan con NumPy, PyTorch y JAX backends
- [ ] Coverage >90%

### Performance (Debería)
- [ ] Comparable o más rápido que statsmodels
- [ ] Sin memory leaks
- [ ] Escalable a 100K+ observaciones

### Documentación (Debe)
- [ ] Todos los docstrings completos
- [ ] Al menos 3 ejemplos funcionando
- [ ] README actualizado
- [ ] Tutorial básico

---

## 📊 Fase 3: GAM - PLANEADO (No iniciado)

### Objetivo

Implementar Modelos Aditivos Generalizados con:
- Bases de splines (cubic, B-splines, P-splines, thin plate)
- Penalización y selección de smoothing parameter (GCV, REML)
- Parser de fórmulas estilo R
- Visualización de términos suaves

### Componentes principales

```
aurora/smoothing/
├── splines/
│   ├── cubic.py              # Cubic splines
│   ├── bsplines.py           # B-splines
│   ├── psplines.py           # P-splines (penalized B-splines)
│   ├── thinplate.py          # Thin plate splines
│   └── tensor.py             # Tensor product splines
├── penalties/
│   ├── ridge.py              # Ridge penalty
│   └── difference.py         # Difference penalty
└── selection/
    ├── gcv.py                # Generalized Cross-Validation
    ├── reml.py               # Restricted Maximum Likelihood
    └── aic.py                # AIC-based selection

aurora/models/gam/
├── fitting.py                # fit_gam()
├── formula.py                # Formula parser (patsy-like)
└── result.py                 # GAMResult
```

### API objetivo

```python
from aurora.models.gam import fit_gam

# Con fórmula tipo R
result = fit_gam(
    formula="y ~ s(x1, bs='tp', k=10) + s(x2, bs='cr') + x3 + x4",
    data=df,
    family='gaussian',
    method='REML'
)

# Visualizar términos suaves
result.plot_smooth('s(x1)')
result.summary()
```

**Estimación**: 6-8 semanas de desarrollo

---

## 📊 Fase 4: GAMM - PLANEADO (No iniciado)

### Objetivo

Añadir efectos aleatorios a GAM:
- Interceptos y pendientes aleatorias
- Efectos cruzados y anidados
- Modelos jerárquicos multinivel
- Estimación REML/ML/Laplace

### Componentes principales

```
aurora/models/gamm/
├── fitting.py                # fit_gamm()
├── random_effects.py         # Random effects structures
└── result.py                 # GAMMResult

aurora/estimation/
├── reml/
│   └── reml.py               # REML estimation
├── ml/
│   └── ml.py                 # Maximum Likelihood
└── laplace/
    └── laplace.py            # Laplace approximation
```

### API objetivo

```python
from aurora.models.gamm import fit_gamm

# Modelo mixto con efectos aleatorios
result = fit_gamm(
    formula="""
        y ~ s(time, by=treatment, k=10) + 
            s(age, bs='cr') + 
            (1 + time | subject) + 
            (1 | clinic)
    """,
    data=df,
    family='gamma',
    link='log',
    method='REML'
)
```

**Estimación**: 6-8 semanas de desarrollo

---

## 🔧 Aspectos Técnicos Importantes

### 1. Multi-Backend Support

**Pattern actual** (debe mantenerse):
```python
from aurora.distributions._utils import namespace, as_namespace_array

def my_function(x, y):
    # Detectar backend automáticamente
    xp = namespace(x, y)
    
    # Convertir a arrays del backend correcto
    x_arr = as_namespace_array(x, xp, like=y)
    
    # Usar operaciones del namespace
    return xp.sum(x_arr * y)
```

**Ventajas**:
- Código único funciona con NumPy, PyTorch, JAX
- No necesita if/else por tipo de array
- Gradientes automáticos cuando se usa PyTorch/JAX

### 2. Testing Strategy

**Multi-backend tests**:
```python
@pytest.fixture(params=["numpy", "pytorch"])
def array_backend(request):
    """Parametrize tests across backends."""
    if request.param == "pytorch":
        pytest.importorskip("torch")
    return request.param

def test_something(array_backend):
    if array_backend == "numpy":
        x = np.array([1, 2, 3])
    else:
        import torch
        x = torch.tensor([1, 2, 3])
    
    result = my_function(x)
    # Test result...
```

**Validation tests**:
```python
def test_glm_matches_statsmodels():
    """GLM results should match statsmodels."""
    # Crear datos
    # Fit con Aurora
    # Fit con statsmodels
    # Comparar coeficientes (tolerancia 1e-6)
    # Comparar std errors (tolerancia 1e-5)
    # Comparar p-values (tolerancia 1e-4)
```

### 3. Performance Considerations

**Optimizaciones actuales**:
- ✅ Backend abstraction permite JIT compilation
- ✅ Operaciones vectorizadas
- ✅ Evita loops en Python

**Optimizaciones futuras**:
- Cython para loops críticos (si necesario)
- Sparse matrices para datasets grandes
- Chunking para datasets que no caben en memoria

### 4. Documentation Standards

**Docstring format** (NumPy style):
```python
def function(param1: Type1, param2: Type2) -> ReturnType:
    """
    Short one-line summary.
    
    Longer description explaining the function's purpose,
    behavior, and any important notes. Can include LaTeX
    for mathematical notation: :math:`f(x) = x^2`.
    
    Parameters
    ----------
    param1 : Type1
        Description of param1
    param2 : Type2
        Description of param2
        
    Returns
    -------
    ReturnType
        Description of return value
        
    Raises
    ------
    ValueError
        When this error occurs
        
    Examples
    --------
    >>> result = function(arg1, arg2)
    >>> result
    expected_output
    
    Notes
    -----
    Additional implementation or mathematical details.
    
    References
    ----------
    .. [1] Author, "Title", Journal, Year.
    """
```

---

## 🚀 Cómo Contribuir

### Setup de Desarrollo

```bash
# Clonar
git clone https://github.com/Matcraft94/Aurora-GLM.git
cd Aurora-GLM

# Crear entorno
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar en modo desarrollo
pip install -e ".[dev,test]"

# Instalar backends opcionales
pip install torch jax jaxlib
```

### Workflow de Desarrollo

```bash
# Crear branch para feature
git checkout -b feature/implement-glm-fitting

# Hacer cambios
# ... editar archivos ...

# Tests
pytest
pytest --cov=aurora --cov-report=html

# Quality checks
ruff format aurora/ tests/
ruff check aurora/ tests/
mypy aurora/

# Commit
git add .
git commit -m "feat(glm): implement IRLS fitting algorithm"

# Push y crear PR
git push origin feature/implement-glm-fitting
```

### Prioridades de Contribución

**ALTA PRIORIDAD** (necesario para Fase 2):
1. Implementar `fit_glm()` con IRLS
2. Completar `GLMResult` con inferencia
3. Implementar residuales y diagnósticos
4. Validación contra statsmodels y R

**MEDIA PRIORIDAD** (nice to have):
1. Distribuciones adicionales (Inverse Gaussian, etc.)
2. Links adicionales (Probit, etc.)
3. Optimizaciones de performance
4. Más ejemplos y tutoriales

**BAJA PRIORIDAD** (futuro):
1. GAM implementation
2. GAMM implementation
3. Visualización avanzada

---

## 📈 Métricas de Éxito del Proyecto

### A Corto Plazo (3 meses)
- [ ] GLM completamente funcional
- [ ] Validado contra statsmodels y R
- [ ] 100+ tests pasando
- [ ] Coverage >90%
- [ ] 3+ ejemplos documentados

### A Mediano Plazo (6 meses)
- [ ] GAM implementado
- [ ] Parser de fórmulas funcional
- [ ] Benchmarks publicados
- [ ] 10+ GitHub stars
- [ ] 1 usuario externo reportando issues

### A Largo Plazo (12 meses)
- [ ] GAMM implementado
- [ ] 100+ GitHub stars
- [ ] 1000+ descargas PyPI/mes
- [ ] Paper o presentación en conferencia
- [ ] 5+ contribuidores

---

## 📚 Referencias Clave

### Teoría
- McCullagh & Nelder (1989) - "Generalized Linear Models" 2nd Ed.
- Wood (2017) - "Generalized Additive Models: An Introduction with R" 2nd Ed.
- Hastie & Tibshirani (1990) - "Generalized Additive Models"

### Implementaciones de Referencia
- **R glm()**: Base stats package
- **R mgcv**: GAM implementation by Simon Wood
- **statsmodels.genmod**: Python GLM implementation
- **scikit-learn**: Para API design patterns

### Recursos Técnicos
- JAX documentation: https://jax.readthedocs.io
- PyTorch documentation: https://pytorch.org/docs
- Array API standard: https://data-apis.org/array-api

---

## 🎯 Próximos Pasos Inmediatos

### Esta Semana

1. **Implementar estructura básica de fit_glm()**
   - [ ] Crear archivo `aurora/models/glm/fitting.py`
   - [ ] Esqueleto de función `fit_glm()`
   - [ ] Parsing de argumentos
   - [ ] Selección de family y link

2. **Implementar IRLS loop**
   - [ ] Inicialización
   - [ ] Loop de optimización
   - [ ] Verificación de convergencia
   - [ ] Tests básicos

3. **Crear GLMResult básico**
   - [ ] Almacenar coeficientes
   - [ ] Almacenar fitted values
   - [ ] Método predict() simple
   - [ ] Tests

### Próximas 2 Semanas

1. **Completar inferencia**
   - [ ] Covariance matrix
   - [ ] Standard errors
   - [ ] Confidence intervals
   - [ ] P-values

2. **Diagnósticos**
   - [ ] Residuales (deviance, pearson)
   - [ ] Cook's distance
   - [ ] Leverage

3. **Validación**
   - [ ] Tests contra statsmodels
   - [ ] Tests contra R
   - [ ] Benchmarks

---

## ✅ Checklist de Commit

Antes de cada commit, verificar:

- [ ] Código formateado con `ruff format`
- [ ] Pasa `ruff check` sin errores
- [ ] Pasa `mypy` sin errores
- [ ] Tests relevantes añadidos
- [ ] Tests pasan (`pytest`)
- [ ] Docstrings completos
- [ ] Type hints presentes
- [ ] Ejemplos en docstrings funcionan

---

**Estado del documento**: Actualizado - Fase 2 en progreso  
**Última actualización**: Noviembre 2025  
**Próxima revisión**: Al completar Fase 2  

---

*Este documento es el "source of truth" para el desarrollo de Aurora-GLM. Debe actualizarse conforme el proyecto avanza.*
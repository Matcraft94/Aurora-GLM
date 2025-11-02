# Aurora-GLM: Contexto del Proyecto y Objetivos de Desarrollo

## 📋 Información General

**Proyecto**: Aurora-GLM  
**Repositorio**: https://github.com/Matcraft94/Aurora-GLM  
**Autor**: Lucy E. Arias (@matcraf94)  
**Versión actual**: 0.2.0-dev  
**Estado**: Fase 2 en progreso (GLM al 80%, preparativos GAM)  
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
- Herramientas de validación (cross-validation, scoring)

### Estado Actual

**Componentes completados**:
- `aurora/models/glm/fitting.py`: algoritmo IRLS estable con soporte para weights, offset y namespaces NumPy/PyTorch, incluyendo `_matvec` libre de BLAS.
- `aurora/models/base/result.py`: `GLMResult` con inferencia diferida (covarianza, errores estándar, p-values) y caché de diagnósticos.
- `aurora/inference/intervals/confidence.py`: intervalos de confianza tipo Wald empaquetados en `ConfidenceIntervalResult`.
- `aurora/inference/hypothesis/wald.py`: pruebas de Wald para contrastes lineales univariados.
- `aurora/inference/diagnostics/glm.py`: residuales (response, Pearson, deviance, working), leverage y distancia de Cook agrupados en `GLMDiagnosticResult`.
- `aurora/inference/__init__.py`: API pública consolidada (`confidence_intervals`, `wald_test`, `glm_diagnostics`).
- `aurora/validation/metrics`: métricas de regresión (MSE/MAE/RMSE), pseudo R² y métricas de clasificación (accuracy, log-loss, Brier).
- `aurora/validation/cross_val`: `KFold` y `cross_val_score` con barajado reproducible.
- Suites de pruebas en `tests/test_inference/*` y `tests/test_validation/*` cubriendo inferencia, diagnósticos, métricas y validación cruzada.

**Cobertura de pruebas representativa**:
- `pytest tests/test_inference/test_confidence_intervals.py`
- `pytest tests/test_inference/test_hypothesis.py`
- `pytest tests/test_inference/test_diagnostics.py`
- `pytest tests/test_validation/test_metrics.py`
- `pytest tests/test_validation/test_classification_metrics.py`
- `pytest tests/test_validation/test_cross_val.py`
- `pytest tests/test_validation`

**Pendientes inmediatos**:
- Soporte para contrastes multivariados y pruebas chi-cuadrado en `wald_test`.
- Diagnósticos adicionales (DFBETAs, residuos estudentizados, gráficos integrados).
- Métricas avanzadas: deviance generalizada, índices de concordancia y reporting integrado.
- Validación cruzada estratificada y scoring específico para clasificación.
- Documentación end-to-end y comparativas con statsmodels/R.

### Checklist de Avance

- [x] `fit_glm()` con IRLS robusto y pruebas sintéticas.
- [x] `GLMResult` con inferencia diferida y método `predict` operativo.
- [x] Intervalos de confianza y p-values vía aproximación Wald.
- [x] Residuales principales, leverage y distancia de Cook en `glm_diagnostics`.
- [x] Métricas de regresión y clasificación con soporte para pesos de muestra.
- [x] Implementación de `pseudo_r2` para GLM.
- [x] `KFold` y `cross_val_score` con semilla reproducible.
- [ ] Extender `wald_test` a contrastes múltiples y pruebas LRT.
- [ ] Incorporar residuales/influencias adicionales (DFBETAs, leverage bayesiano).
- [ ] Añadir métricas específicas (concordancia, deviance generalizada) y reporting integrado.
- [ ] Documentar ejemplos end-to-end y notebooks.

### Plan de Implementación Ajustado

**Semana 1 (completada)**
- [x] Implementar `fit_glm()` y pruebas de convergencia.
- [x] Completar `GLMResult` con propiedades lazy y predicción.
- [x] Calcular covariance, errores estándar y p-values.

**Semana 2 (en curso)**
- [x] Residuales básicos y medidas de influencia iniciales.
- [x] Métricas de evaluación (regresión, clasificación, pseudo R²).
- [x] Cross-validation genérico (`KFold`, `cross_val_score`).
- [ ] Métricas avanzadas (deviance específica, concordance index).
- [ ] Integración con benchmarks y validación externa.

**Semana 3 (pendiente)**
- [ ] Ejemplos documentados y notebooks.
- [ ] Visualizaciones y diagnósticos gráficos.
- [ ] Documentación y preparación de release 0.2.0.

### Criterios de Éxito para Fase 2 (estado)

**Funcionalidad (Debe)**
- [x] `fit_glm()` funciona con familias gaussian, poisson, binomial y gamma.
- [x] Predicción (`type="response"`/`"link"`) validada con datos sintéticos.
- [x] Intervalos de confianza y p-values por aproximación Wald.
- [x] Residuales principales (response, Pearson, deviance, working) implementados.

**Validación (Debe)**
- [ ] Resultados validados contra statsmodels.
- [ ] Resultados validados contra R `glm()`.
- [x] Tests automatizados cubren rutas NumPy y (parcialmente) PyTorch.
- [ ] Coverage >90% (medición pendiente).

**Performance (Debería)**
- [ ] Benchmarks frente a statsmodels.
- [ ] Escalabilidad probada en datasets >100K observaciones.

**Documentación (Debe)**
- [ ] Docstrings y ejemplos consolidados.
- [ ] README actualizado con nuevas capacidades.
- [ ] Tutorial básico publicado.

---

## 🗂️ Backlog Operativo Prioritario (Nov 2025)

| Categoría | Objetivo | Definición de terminado |
| --- | --- | --- |
| **Inferencia** | Integrar intervalos y p-values en `GLMResult`, ampliar `wald_test` a contrastes multivariados y LRT. | `predict(interval="confidence")` funcional, rutas nuevas cubiertas en `tests/test_models/test_glm_fitting.py` y `tests/test_inference/test_hypothesis.py`. |
| **Diagnósticos** | Añadir residuales studentizados, DFBETAs y resumen tabular reutilizable. | API `glm_diagnostics` extendida, fixtures sintéticos actualizados, documentación básica en docstrings. |
| **Métricas & Validación** | Incorporar concordance index, deviance específica, pseudo R² avanzados y `StratifiedKFold`. | Funciones disponibles en `aurora/validation`, pruebas en `tests/test_validation`, ejemplos en notebooks. |
| **Documentación & UX** | Actualizar README/guías y preparar notebooks demostrativos. | README sincronizado, guía rápida GLM publicada, dos notebooks revisados. |
| **Validación Externa** | Automatizar comparativas con statsmodels/R y registrar benchmarks. | Scripts en `benchmarks/` con resultados versionados y reporte semanal.

### Tablero y seguimiento

- GitHub Projects en modo Kanban (`Todo → In progress → Review → Done`).
- Issues etiquetados por categoría, tamaño (`S/M/L`) y responsable claro.
- Revisión de backlog los lunes; retro semanal de riesgos y ajustes.

### Ciclo QA semanal

- Pipeline: `pytest`, suites específicas por módulo, comparación automática vs statsmodels/R (`benchmarks/run_glm_checks.py`).
- Indicadores mínimos: cobertura ≥85%, error relativo en coeficientes <1e-6, tiempo de ajuste registrado (NumPy y PyTorch).
- Entrega de hallazgos en `docs/reports/QA_SEMANA.md` con acciones correctivas.

### Hitos Sprint (3 semanas)

1. **Semana 1**: Backlog de inferencia cerrado + script de validación externa operativo.
2. **Semana 2**: Diagnósticos y métricas avanzadas listos; primera versión de `GLMResult.summary()`.
3. **Semana 3**: Documentación y notebooks publicados; prototipo de `plot_diagnostics` con hooks a visualización.

### Preparativos Release 0.2.0

- Checklist funcional (inferencias, diagnósticos, métricas, documentación) completado.
- Changelog redactado y versiones sincronizadas.
- Validaciones cruzadas firmadas (NumPy, PyTorch); publicar resultados comparativos.


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

> Referencia: ver `aurora/smoothing/DESIGN.md` para el borrador de arquitectura, entregables incrementales y riesgos identificados.

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

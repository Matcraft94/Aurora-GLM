# Aurora-GLM: Contexto del Proyecto y Objetivos de Desarrollo

## 📋 Información General

**Proyecto**: Aurora-GLM  
**Repositorio**: https://github.com/Matcraft94/Aurora-GLM  
**Autor**: Lucy E. Arias (@matcraf94)  
**Versión actual**: 0.3.0
**Estado**: Fase 3 COMPLETADA (GAM 100% - splines, REML, formulas, tensor products, thin plate splines)  
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
- `aurora/models/base/result.py`: `GLMResult` con inferencia diferida (covarianza, errores estándar, p-values), caché de diagnósticos e intervalos de confianza en `predict()`.
- `aurora/inference/intervals/confidence.py`: intervalos de confianza tipo Wald empaquetados en `ConfidenceIntervalResult`.
- `aurora/inference/hypothesis/wald.py`: pruebas de Wald para contrastes lineales univariados **y multivariados** (chi-cuadrado).
- `aurora/inference/diagnostics/glm.py`: residuales (response, Pearson, deviance, working, **studentized**), leverage, distancia de Cook y **DFBETAs** agrupados en `GLMDiagnosticResult`.
- `aurora/inference/__init__.py`: API pública consolidada (`confidence_intervals`, `wald_test`, `glm_diagnostics`).
- `aurora/validation/metrics`: métricas de regresión (MSE/MAE/RMSE), pseudo R² y métricas de clasificación (accuracy, log-loss, Brier).
- `aurora/validation/cross_val`: `KFold`, `StratifiedKFold` y `cross_val_score` con barajado reproducible y agregados (`CrossValResult`).
- `benchmarks/run_glm_checks.py`: comparativa automatizada contra statsmodels (Gaussian, Poisson, Binomial, Gamma-log) con reporte JSON.
- `pyproject.toml`: metadatos de empaquetado y soporte para instalación editable vía `pip install -e .`.
- Suites de pruebas en `tests/test_inference/*`, `tests/test_validation/*` y `tests/test_models/*` cubriendo inferencia, diagnósticos, métricas, validación cruzada y fitting GLM (86 tests pasando).

**Cobertura de pruebas representativa**:
- `pytest tests/test_inference/test_confidence_intervals.py`
- `pytest tests/test_inference/test_hypothesis.py`
- `pytest tests/test_inference/test_diagnostics.py`
- `pytest tests/test_validation/test_metrics.py`
- `pytest tests/test_validation/test_classification_metrics.py`
- `pytest tests/test_validation/test_cross_val.py`
- `pytest tests/test_validation`

**Pendientes inmediatos** (para cerrar Fase 2):
- Método `summary()` en `GLMResult` con tabla formateada de coeficientes, estadísticos, y métricas del modelo.
- Método `plot_diagnostics()` con visualizaciones estándar (residuals vs fitted, Q-Q, scale-location, leverage).
- Métricas avanzadas opcionales: concordance index (C-index) para clasificación binaria.
- Validación cruzada con R `glm()` (complementar validación contra statsmodels).
- Documentación end-to-end: notebooks demostrativos con datasets reales.
- Medir y reportar coverage (instalar `pytest-cov`, objetivo ≥90%).

### Checklist de Avance

**Completados:**
- [x] `fit_glm()` con IRLS robusto y pruebas sintéticas.
- [x] `GLMResult` con inferencia diferida, método `predict` con intervalos de confianza.
- [x] Intervalos de confianza y p-values vía aproximación Wald.
- [x] Residuales completos (response, Pearson, deviance, working, studentized) en `glm_diagnostics`.
- [x] Leverage, distancia de Cook y DFBETAs implementados.
- [x] Métricas de regresión y clasificación con soporte para pesos de muestra.
- [x] Implementación de `pseudo_r2` para GLM.
- [x] `KFold`, `StratifiedKFold` y `cross_val_score` con semilla reproducible y resumen estadístico (`CrossValResult`).
- [x] Benchmark automatizado contra statsmodels (`benchmarks/run_glm_checks.py`).
- [x] `wald_test` extendido a contrastes múltiples con estadístico chi-cuadrado.

**En curso (Sprint actual):**
- [ ] Implementar `GLMResult.summary()` con tabla formateada.
- [ ] Implementar `GLMResult.plot_diagnostics()` con matplotlib.
- [ ] Añadir concordance index (C-index) a métricas de clasificación.
- [ ] Script de validación contra R `glm()` (complementar statsmodels).
- [ ] Medir coverage con pytest-cov (instalar en entorno).
- [ ] Crear 2 notebooks demostrativos (uno Poisson, uno Binomial).
- [ ] Actualizar README con ejemplos de uso avanzado.

### Plan de Implementación Ajustado

**Semana 1 (✅ completada)**
- [x] Implementar `fit_glm()` y pruebas de convergencia.
- [x] Completar `GLMResult` con propiedades lazy y predicción.
- [x] Calcular covariance, errores estándar y p-values.

**Semana 2 (✅ completada)**
- [x] Residuales completos (response, Pearson, deviance, working, studentized).
- [x] Medidas de influencia (leverage, Cook's distance, DFBETAs).
- [x] Métricas de evaluación (regresión, clasificación, pseudo R²).
- [x] Cross-validation genérico (`KFold`, `StratifiedKFold`, `cross_val_score`).
- [x] Wald tests multivariados con estadístico chi-cuadrado.
- [x] Integración con benchmarks y validación externa (`benchmarks/run_glm_checks.py`).

**Semana 3 (🚧 en curso - para cerrar Fase 2)**
- [ ] `GLMResult.summary()`: tabla formateada con coeficientes, std errors, z-scores, p-values.
- [ ] `GLMResult.plot_diagnostics()`: 4 gráficos estándar (residuals vs fitted, Q-Q, scale-location, leverage plot).
- [ ] Concordance index (C-index) en `aurora/validation/metrics/classification.py`.
- [ ] Script de validación contra R `glm()` en `benchmarks/run_r_checks.R` + wrapper Python.
- [ ] Instalar pytest-cov y medir coverage (objetivo ≥90%).
- [ ] Notebook demostrativo: Poisson regression con dataset de conteos.
- [ ] Notebook demostrativo: Binomial logistic regression con clasificación.
- [ ] Actualizar README con ejemplos de `summary()` y `plot_diagnostics()`.
- [ ] Preparar CHANGELOG para release 0.2.0.

### Criterios de Éxito para Fase 2 (estado)

**Funcionalidad (Debe)**
- [x] `fit_glm()` funciona con familias gaussian, poisson, binomial y gamma.
- [x] Predicción (`type="response"`/`"link"`) validada con datos sintéticos.
- [x] Intervalos de confianza y p-values por aproximación Wald.
- [x] Residuales principales (response, Pearson, deviance, working) implementados.

-**Validación (Debe)**
- [x] Resultados validados contra statsmodels (`benchmarks/run_glm_checks.py`): max |delta_coef| ≈ 4e-06.
- [ ] Resultados validados contra R `glm()` (pendiente script R).
- [x] Tests automatizados cubren rutas NumPy y PyTorch (86 tests pasando).
- [ ] Coverage >90% (pendiente instalación pytest-cov y medición).

**Performance (Debería)**
- [ ] Benchmarks frente a statsmodels.
- [ ] Escalabilidad probada en datasets >100K observaciones.

**Documentación (Debe)**
- [ ] Docstrings y ejemplos consolidados.
- [ ] README actualizado con nuevas capacidades.
- [ ] Tutorial básico publicado.

---

## 🗂️ Backlog Detallado para Cerrar Fase 2

### 1. Reporting y Visualización (Prioridad ALTA)

#### 1.1. Implementar `GLMResult.summary()`
**Archivo**: `aurora/models/base/result.py`

**Especificación**:
- Método `summary(self, *, detailed: bool = True) -> str`
- Retorna string multilínea con formato tabular
- Secciones:
  1. **Modelo**: familia, link, observaciones, parámetros, deviance, AIC, BIC
  2. **Convergencia**: iteraciones, estado (converged/failed)
  3. **Tabla de coeficientes**: nombre (X0, X1, ..., intercept), coef, std_error, z_score, p_value, significancia (*, **, ***)
  4. **Bondad de ajuste**: null deviance, model deviance, pseudo R²

**Pruebas unitarias** (`tests/test_models/test_glm_summary.py`):
- `test_summary_includes_coefficient_table_with_all_columns()`
- `test_summary_shows_convergence_status()`
- `test_summary_includes_model_metrics_aic_bic_deviance()`
- `test_summary_works_without_intercept()`
- `test_summary_handles_non_converged_models()`

**Criterio de éxito**: ejecutar `result.summary()` y obtener tabla legible similar a R/statsmodels.

---

#### 1.2. Implementar `GLMResult.plot_diagnostics()`
**Archivo**: `aurora/models/base/result.py`

**Especificación**:
- Método `plot_diagnostics(self, *, figsize: tuple = (12, 10)) -> Figure`
- Requiere matplotlib (importación lazy, error claro si no disponible)
- 4 subplots (2x2):
  1. **Residuals vs Fitted**: residuales de respuesta vs valores ajustados
  2. **Q-Q Plot**: cuantiles teóricos normales vs residuales studentizados
  3. **Scale-Location**: √|residuales studentizados| vs valores ajustados
  4. **Residuals vs Leverage**: residuales studentizados vs leverage, resaltar Cook's distance

**Pruebas unitarias** (`tests/test_models/test_glm_plotting.py`):
- `test_plot_diagnostics_creates_figure_with_four_subplots()`
- `test_plot_diagnostics_raises_without_matplotlib()`
- `test_plot_diagnostics_uses_cached_diagnostics()`
- `test_plot_can_be_saved_to_file()`

**Criterio de éxito**: generar visualización sin errores, verificar que usa caché de diagnósticos.

---

### 2. Métricas Avanzadas (Prioridad MEDIA)

#### 2.1. Concordance Index (C-index)
**Archivo**: `aurora/validation/metrics/classification.py`

**Especificación**:
- Función `concordance_index(y_true, y_proba, *, weights=None) -> float`
- Mide discriminación en clasificación binaria (AUC-like)
- C = P(score_pos > score_neg) para pares concordantes
- Soporte para pesos de muestra

**Pruebas unitarias** (`tests/test_validation/test_classification_metrics.py`):
- `test_concordance_index_perfect_separation_returns_one()`
- `test_concordance_index_random_predictions_returns_half()`
- `test_concordance_index_with_sample_weights()`
- `test_concordance_index_handles_ties()`

**Criterio de éxito**: C-index = 1.0 para separación perfecta, ≈0.5 para predicciones aleatorias.

---

### 3. Validación Externa (Prioridad ALTA)

#### 3.1. Validación contra R `glm()`
**Archivo**: `benchmarks/run_r_checks.R` + `benchmarks/compare_with_r.py`

**Especificación R script**:
```r
# run_r_checks.R
# Ajustar GLMs con R y exportar resultados a JSON
# Familias: gaussian, poisson, binomial, gamma
# Exportar: coef, std.error, deviance, aic, fitted.values
```

**Especificación Python wrapper**:
- Lee datos sintéticos, ejecuta R via `subprocess`, compara resultados
- Tolerancias: coef ≤1e-5, deviance ≤1e-4

**Pruebas de integración**:
- Ejecutar script manualmente y verificar que coinciden resultados
- Registrar máximas diferencias en `benchmarks/results/r_comparison.json`

**Criterio de éxito**: diferencias ≤ tolerancias especificadas para 4 familias.

---

### 4. Coverage y QA (Prioridad ALTA)

#### 4.1. Instalar pytest-cov y medir coverage
**Comandos**:
```bash
pip install pytest-cov
pytest --cov=aurora --cov-report=html --cov-report=term-missing
```

**Objetivo**: Coverage ≥90%

**Identificar gaps**:
- Revisar reporte HTML en `htmlcov/index.html`
- Añadir tests para líneas no cubiertas (edge cases, error handling)

**Pruebas adicionales sugeridas**:
- `tests/test_models/test_glm_edge_cases.py`:
  - `test_fit_glm_with_single_feature()`
  - `test_fit_glm_with_collinear_features()`
  - `test_fit_glm_with_all_zero_response()`
  - `test_fit_glm_without_intercept_flag()`
  - `test_predict_with_mismatched_dimensions_raises_error()`

**Criterio de éxito**: cobertura ≥90%, todos los módulos principales cubiertos.

---

### 5. Documentación (Prioridad ALTA)

#### 5.1. Notebook: Poisson Regression
**Archivo**: `examples/notebooks/01_poisson_regression.ipynb`

**Contenido**:
1. Dataset: conteos simulados o reales (ej. número de eventos por unidad de tiempo)
2. Exploración: histograma de y, estadísticas descriptivas
3. Fitting: `fit_glm(X, y, family='poisson', link='log')`
4. Diagnósticos: `result.diagnostics_`, `result.plot_diagnostics()`
5. Inferencia: `result.summary()`, interpretación de coeficientes
6. Predicción: `result.predict()` con intervalos de confianza
7. Cross-validation: `cross_val_score()` con scoring=-deviance

**Criterio de éxito**: notebook ejecutable sin errores, resultados interpretables.

---

#### 5.2. Notebook: Logistic Regression
**Archivo**: `examples/notebooks/02_logistic_regression.ipynb`

**Contenido**:
1. Dataset: clasificación binaria simulada o real
2. Exploración: distribución de clases, correlación features
3. Fitting: `fit_glm(X, y, family='binomial', link='logit')`
4. Diagnósticos: leverage, Cook's distance, DFBETAs
5. Métricas: accuracy, log-loss, Brier score, concordance index
6. ROC curve: usar sklearn para comparación
7. Wald tests: contrastes multivariados para hipótesis específicas

**Criterio de éxito**: notebook ejecutable, métricas comparables con sklearn.

---

#### 5.3. Actualizar README
**Archivo**: `README.md`

**Cambios**:
- Sección "Current Usage" → añadir ejemplo de `result.summary()`
- Sección "Current Usage" → añadir ejemplo de `result.plot_diagnostics()`
- Actualizar estado de Fase 2 a "~95% completo"
- Añadir badges de coverage si se configura CI

**Criterio de éxito**: README sincronizado con capacidades actuales.

---

### 6. Preparación Release 0.2.0

#### 6.1. Crear CHANGELOG.md
**Archivo**: `CHANGELOG.md`

**Formato**:
```markdown
# Changelog

## [0.2.0] - 2025-11-XX

### Added
- GLM fitting with IRLS for Gaussian, Poisson, Binomial, Gamma families
- Confidence intervals and p-values via Wald approximation
- Comprehensive diagnostics: residuals (response, Pearson, deviance, studentized), leverage, Cook's distance, DFBETAs
- `GLMResult.summary()` method with formatted coefficient table
- `GLMResult.plot_diagnostics()` with 4 standard plots
- Cross-validation utilities: KFold, StratifiedKFold, cross_val_score
- Validation metrics: MSE, MAE, RMSE, pseudo R², accuracy, log-loss, Brier, concordance index
- Benchmarking against statsmodels and R glm()

### Fixed
- Numerical stability in IRLS with custom Gaussian elimination
- Multi-backend support for NumPy and PyTorch

### Documentation
- Two demo notebooks (Poisson and Logistic regression)
- CLAUDE.md for AI assistant guidance
```

---

### Resumen de Entregables

| # | Entregable | Tipo | Tests | Archivos |
|---|------------|------|-------|----------|
| 1 | `GLMResult.summary()` | Implementación | 5 unitarios | `result.py`, `test_glm_summary.py` |
| 2 | `GLMResult.plot_diagnostics()` | Implementación | 4 unitarios | `result.py`, `test_glm_plotting.py` |
| 3 | Concordance index | Implementación | 4 unitarios | `classification.py`, test existente |
| 4 | Validación R | Script + tests | 1 integración | `run_r_checks.R`, `compare_with_r.py` |
| 5 | Coverage ≥90% | QA | Tests edge cases | `test_glm_edge_cases.py` |
| 6 | Notebook Poisson | Documentación | - | `01_poisson_regression.ipynb` |
| 7 | Notebook Logistic | Documentación | - | `02_logistic_regression.ipynb` |
| 8 | README actualizado | Documentación | - | `README.md` |
| 9 | CHANGELOG | Documentación | - | `CHANGELOG.md` |

**Total estimado**: 5-7 días de desarrollo para cerrar Fase 2.


## 📊 Fase 3: GAM - COMPLETADA ✅ (100%)

### Objetivo ALCANZADO

Implementación completa de Modelos Aditivos Generalizados con:
- ✅ Bases de splines (cubic, B-splines) - COMPLETADO
- ✅ Penalización y selección de smoothing parameter (GCV, REML) - COMPLETADO
- ✅ Fitting GAM univariado - COMPLETADO
- ✅ Parser de fórmulas estilo R - COMPLETADO
- ✅ GAMs multivariados (additive) - COMPLETADO
- ✅ Visualización de términos suaves - COMPLETADO
- ✅ Tensor product smooths - COMPLETADO
- ✅ Thin plate splines - COMPLETADO

### Componentes Implementados (229 tests nuevos)

```
aurora/smoothing/
├── bspline.py                ✅ B-splines Cox-de Boor (17 tests)
├── cubic.py                  ✅ Natural cubic splines (16 tests)
├── difference.py             ✅ Difference/ridge/combined penalties (20 tests)
├── gcv.py                    ✅ GCV smoothing parameter selection (15 tests)
├── reml.py                   ✅ REML smoothing parameter selection (20 tests)
├── tensor.py                 ✅ Tensor product smooths (13 tests)
└── thinplate.py              ✅ Thin plate splines (24 tests)

aurora/models/gam/
├── fitting.py                ✅ fit_gam() univariado (20 tests)
├── result.py                 ✅ GAMResult con predicciones y summary
├── additive.py               ✅ fit_additive_gam() y fit_gam_formula() (20 tests)
├── formula.py                ✅ Parser de fórmulas estilo R (12 tests)
├── terms.py                  ✅ SmoothTerm, ParametricTerm, TensorTerm (14 tests)
└── plotting.py               ✅ plot_smooth() y plot_all_smooths() (18 tests)
```

### Estado Final de Implementación

**Completado (Fase 3.1-3.7 - TODO)**:

#### Fase 3.1: Spline Basis Functions ✅
- ✅ B-splines con recursión Cox-de Boor, soporte local, partition of unity
- ✅ Splines cúbicos naturales con base de potencias truncadas
- ✅ Colocación de knots (cuantiles, uniforme)
- ✅ Penalties analíticos para splines cúbicos

#### Fase 3.2: Penalty Matrices ✅
- ✅ Matrices de penalización: diferencias de orden 2
- ✅ Ridge penalties para regularización
- ✅ Weighted penalties con ponderaciones custom
- ✅ Block-diagonal penalties para múltiples términos

#### Fase 3.3: Smoothing Parameter Selection ✅
- ✅ Selección automática de λ vía GCV (Generalized Cross-Validation)
- ✅ Selección automática vía REML (Restricted Maximum Likelihood)
- ✅ Optimización en escala logarítmica para estabilidad numérica
- ✅ Selección per-term λ y simultánea para múltiples términos
- ✅ Tracking de EDF (Effective Degrees of Freedom)

#### Fase 3.4: GAM Fitting ✅
- ✅ Ajuste GAM univariado con `fit_gam(x, y, n_basis=10, basis_type='bspline')`
- ✅ Ajuste GAM multivariado con `fit_additive_gam(X, y, smooth_terms=[...])`
- ✅ `GAMResult` con `predict()`, `summary()`, tracking de EDF y λ
- ✅ `AdditiveGAMResult` con per-term λ, EDF, y penalties
- ✅ Soporte para pesos de observaciones
- ✅ Términos paramétricos y smooth mixtos

#### Fase 3.5: Formula Parser ✅
- ✅ Parser de fórmulas tipo R: `y ~ s(x1) + s(x2, bs='cubic') + x3`
- ✅ Soporte para smooth terms con especificación de basis
- ✅ Soporte para términos paramétricos
- ✅ Soporte para tensor products: `te(x1, x2)`
- ✅ API de alto nivel: `fit_gam_formula(formula, data, method='REML')`
- ✅ Validación comprehensiva con mensajes de error informativos

#### Fase 3.6: Visualization ✅
- ✅ `plot_smooth()` para visualizar términos smooth individuales
- ✅ Bandas de confianza (credible intervals bayesianos)
- ✅ Overlay de partial residuals
- ✅ Rug plots para distribución de datos
- ✅ `plot_all_smooths()` para grid de todos los términos
- ✅ Apariencia customizable (colores, estilos, labels)

#### Fase 3.7: Advanced Smoothing ✅
- ✅ Tensor product smooths para interacciones multidimensionales
- ✅ Construcción de basis vía Kronecker products
- ✅ Estructura de penalty dual (λ separado por dimensión)
- ✅ Thin plate splines para smoothing multidimensional
- ✅ Funciones radiales variando por dimensión (d=1: r³, d=2: r²log(r), d=3: r)
- ✅ Null space polinomial (no penalizado)
- ✅ Selección eficiente de knots (uniform, random)

**Tests totales**: 348 pasando (229 nuevos en Fase 3), 2 skipped

> Diseño completo en `aurora/smoothing/DESIGN.md` con arquitectura incremental y riesgos.

### API Implementada

#### Univariado

```python
from aurora.models.gam import fit_gam
import numpy as np

# Datos con relación no-lineal
x = np.linspace(0, 1, 100)
y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(100)

# Ajuste GAM con selección automática de λ (GCV o REML)
result = fit_gam(x, y, n_basis=12, basis_type='bspline', method='REML')

# Summary con λ, EDF, R², diagnósticos
print(result.summary())

# Predicciones
y_pred = result.predict(np.linspace(0, 1, 200))
```

#### Multivariado con Fórmulas

```python
from aurora.models.gam import fit_gam_formula
import pandas as pd

# Con fórmula tipo R (AHORA DISPONIBLE)
result = fit_gam_formula(
    formula="y ~ s(x1, k=10) + s(x2, bs='cubic') + x3",
    data=df,
    method='REML'
)

# Tensor products para interacciones
result = fit_gam_formula(
    formula="y ~ te(x1, x2) + s(x3)",
    data=df,
    method='GCV'
)

# Visualizar términos suaves
from aurora.models.gam import plot_smooth, plot_all_smooths
plot_smooth(result, term='s(x1)')
plot_all_smooths(result)
```

#### Low-Level API con SmoothTerm

```python
from aurora.models.gam import fit_additive_gam, SmoothTerm, ParametricTerm

result = fit_additive_gam(
    X, y,
    smooth_terms=[
        SmoothTerm(variable=0, n_basis=12, basis_type='bspline'),
        SmoothTerm(variable=1, n_basis=10, basis_type='cubic')
    ],
    parametric_terms=[
        ParametricTerm(variable=2)  # Linear term
    ],
    method='REML'
)
```

### Estadísticas Finales Fase 3

- **Líneas de código**: ~3,600 nuevas líneas
- **Módulos nuevos**: 14 archivos (7 en smoothing, 7 en models/gam)
- **Tests nuevos**: 229 tests (348 totales pasando)
- **Commits**: 5 commits principales
- **Duración**: 3 sesiones de desarrollo
- **Cobertura funcional**: 100% de objetivos alcanzados

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

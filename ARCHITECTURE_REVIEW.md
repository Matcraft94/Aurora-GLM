# Aurora-GLM Architecture Review

**Version**: 0.5.0-dev  
**Date**: 2024-12-19  
**Purpose**: Análisis completo de la arquitectura del proyecto con recomendaciones de refactorización.

---

## 📋 Resumen Ejecutivo

Aurora-GLM es un framework modular para modelos lineales generalizados (GLM), modelos aditivos generalizados (GAM) y modelos mixtos (GAMM). La arquitectura actual presenta fortalezas significativas pero también oportunidades de mejora.

### Calificación General: ⭐⭐⭐⭐ (4/5)

**Fortalezas**:
- Diseño modular bien estructurado
- Soporte multi-backend (NumPy, PyTorch, JAX)
- Documentación matemática extensa en docstrings
- Cobertura de tests razonable

**Áreas de Mejora**:
- Módulos vacíos (placeholders)
- Duplicación de patrones Result
- Exportaciones incompletas en `__init__.py` principal
- Inconsistencias en convenciones de nombrado

---

## 📊 Mapa de Arquitectura

```
aurora/
├── core/                    # ⭐⭐⭐⭐⭐ Núcleo bien diseñado
│   ├── backends/            # Multi-backend abstraction
│   ├── linalg/              # ⚠️ VACÍO - solo __init__.py
│   ├── autodiff/            # ⚠️ VACÍO - solo __init__.py
│   ├── optimization/        # IRLS, Newton, L-BFGS
│   └── types.py             # Type aliases
│
├── distributions/           # ⭐⭐⭐⭐⭐ Muy bien estructurado
│   ├── families/            # Gaussian, Poisson, Binomial, Gamma, Student-t, NB, Tweedie
│   ├── links/               # Identity, Log, Logit, Inverse, Sqrt, Power
│   ├── base.py              # ABC: Family, LinkFunction
│   └── _utils.py            # Helpers multi-backend
│
├── models/                  # ⭐⭐⭐⭐ Bien pero con redundancia
│   ├── base/                # ModelResult, GLMResult
│   ├── glm/                 # fit_glm, predict_glm
│   ├── gam/                 # fit_gam, smooth terms, plotting
│   └── gamm/                # fit_gamm, PQL, Laplace, covariance
│
├── estimation/              # ⭐⭐ MAYORMENTE VACÍO
│   ├── laplace/             # ⚠️ VACÍO - funcionalidad en gamm/laplace.py
│   ├── ml/                  # ⚠️ VACÍO
│   └── reml/                # ⚠️ VACÍO - funcionalidad en gamm/estimation.py
│
├── inference/               # ⭐⭐⭐⭐ Bien estructurado
│   ├── anova/               # ⚠️ VACÍO
│   ├── diagnostics/         # GLM diagnostics
│   ├── hypothesis/          # Wald tests
│   ├── intervals/           # Confidence intervals
│   └── robust.py            # Bootstrap, HC covariance
│
├── smoothing/               # ⭐⭐⭐⭐⭐ Completo y bien documentado
│   ├── penalties/           # Difference penalties
│   ├── selection/           # GCV, REML
│   ├── splines/             # B-splines, Cubic splines
│   ├── tensor.py            # Tensor product smooths
│   └── thinplate.py         # Thin plate splines
│
├── validation/              # ⭐⭐⭐⭐ Funcional
│   ├── cross_val/           # K-Fold, cross_val_score
│   ├── metrics/             # MSE, AUC, etc.
│   └── sensitivity/         # ⚠️ VACÍO
│
├── io/                      # ⭐ COMPLETAMENTE VACÍO
│   ├── converters/          # ⚠️ VACÍO
│   ├── readers/             # ⚠️ VACÍO
│   └── writers/             # ⚠️ VACÍO
│
├── visualization/           # ⭐⭐ MAYORMENTE VACÍO
│   ├── model_plots/         # ⚠️ VACÍO - funcionalidad dispersa
│   ├── predictions/         # ⚠️ VACÍO
│   └── residuals/           # ⚠️ VACÍO
│
└── utils/                   # ⭐⭐⭐ Básico pero funcional
    ├── exceptions/          # Custom exceptions
    └── validation/          # Input validation helpers
```

---

## 🔴 Problemas Críticos

### 1. Módulos Vacíos (Technical Debt)

Los siguientes módulos solo contienen docstrings sin funcionalidad:

| Módulo | Descripción | Acción Recomendada |
|--------|-------------|-------------------|
| `estimation/laplace/` | Vacío | Mover `gamm/laplace.py` aquí |
| `estimation/ml/` | Vacío | Implementar o eliminar |
| `estimation/reml/` | Vacío | Mover `gamm/estimation.py` aquí |
| `core/linalg/` | Vacío | Agregar abstracciones QR, SVD, Cholesky |
| `core/autodiff/` | Vacío | Integrar JAX/PyTorch autodiff o eliminar |
| `inference/anova/` | Vacío | Implementar ANOVA para GLM/GAM |
| `validation/sensitivity/` | Vacío | Implementar o eliminar |
| `io/converters/` | Vacío | Implementar conversores pandas/R |
| `io/readers/` | Vacío | Implementar lectores CSV/RDS |
| `io/writers/` | Vacío | Implementar exportadores |
| `visualization/model_plots/` | Vacío | Consolidar plotting aquí |
| `visualization/predictions/` | Vacío | Implementar o eliminar |
| `visualization/residuals/` | Vacío | Consolidar con diagnostics |

**Impacto**: ~35% de la estructura declarada no tiene implementación.

### 2. Exportaciones Incompletas en `aurora/__init__.py`

El archivo principal solo exporta funciones de backend:
```python
# Actual
__all__ = ["available_backends", "get_backend", "register_backend"]

# Debería incluir también:
# - fit_glm, fit_gam, fit_gamm
# - GaussianFamily, PoissonFamily, BinomialFamily
# - IdentityLink, LogLink, LogitLink
```

**Impacto**: Los usuarios deben conocer la estructura interna para imports.

### 3. Duplicación de Clases Result

Existen múltiples clases Result con patrones similares:

| Clase | Ubicación | Atributos Comunes |
|-------|-----------|-------------------|
| `ModelResult` | `models/base/result.py` | params, fitted_values, converged |
| `GLMResult` | `models/base/result.py` | coef_, mu_, deviance_, aic_ |
| `GAMResult` | `models/gam/result.py` | coef, fitted, deviance, edf |
| `GAMMResult` | `models/gamm/fitting.py` | beta, b, fitted_values, variance_components |
| `PQLResult` | `models/gamm/pql.py` | beta, b, psi, sigma2, fitted_values |
| `LaplaceResult` | `models/gamm/laplace.py` | beta, b, fitted_values, log_likelihood |
| `REMLResult` | `models/gamm/estimation.py` | variance_components, log_likelihood |

**Recomendación**: Crear una jerarquía de herencia:
```python
ModelResult (base)
├── LinearModelResult
│   └── GLMResult
│       ├── GAMResult
│       └── GAMMResult (MixedModelResult mixin)
```

---

## 🟡 Problemas Moderados

### 4. Funcionalidad Dispersa de Plotting

Las funciones de visualización están distribuidas en:
- `models/gam/plotting.py` - plot_smooth, plot_all_smooths
- `models/gamm/plotting.py` - plot_caterpillar, plot_random_effects_*, plot_diagnostics_panel
- `models/gamm/diagnostics.py` - plot_diagnostics, plot_random_effects
- `inference/diagnostics/glm.py` - (sin plotting directo)

**Recomendación**: Centralizar en `visualization/`:
```
visualization/
├── __init__.py          # Re-exportar todo
├── base.py              # Estilos, helpers comunes
├── smooth_effects.py    # Curvas suaves (de gam/plotting)
├── random_effects.py    # Caterpillar, QQ (de gamm/plotting)
├── diagnostics.py       # Residuos, leverage (de gamm/diagnostics)
└── model_summary.py     # Tablas de coeficientes
```

### 5. Inconsistencia en Convenciones de Nombrado

| Patrón | Ejemplos | Inconsistencia |
|--------|----------|----------------|
| Funciones | `fit_glm`, `fit_gamm_gaussian`, `fit_pql_with_smooth` | Algunas muy largas |
| Clases | `BSplineBasis`, `CubicSplineBasis` vs `StudentTFamily` | T vs T |
| Atributos | `coef_` vs `beta` vs `coef` | Trailing underscore inconsistente |
| Módulos | `gaussian.py` vs `student_t.py` | Underscore vs no underscore |

**Recomendación**: Adoptar convención scikit-learn:
- Atributos ajustados: siempre con `_` final (`coef_`, `fitted_`)
- Nombres de clases: CamelCase sin underscores (`StudentTFamily` → `StudentTFamily` ✓)

### 6. Acoplamiento entre `models/gamm/` y `smoothing/`

El módulo `gamm` tiene dependencias directas de smoothing:
- `pql_smooth.py` importa `BSplineBasis`
- `smoothing_selection.py` duplica lógica de `smoothing/selection/`

**Recomendación**: Usar interfaces abstractas para desacoplar.

---

## 🟢 Oportunidades de Mejora

### 7. Mejoras en API Pública

**Actual**:
```python
from aurora.models.glm import fit_glm
from aurora.distributions.families import GaussianFamily
from aurora.distributions.links import IdentityLink
```

**Propuesto**:
```python
from aurora import fit_glm, Gaussian, identity
# o
import aurora
aurora.fit_glm(X, y, family=aurora.Gaussian())
```

### 8. Documentación de Tipos

Usar `TypeAlias` y `Protocol` para mejor documentación:
```python
from typing import Protocol, TypeAlias

class FamilyProtocol(Protocol):
    def variance(self, mu: Array) -> Array: ...
    def log_likelihood(self, y: Array, mu: Array) -> float: ...

FamilyLike: TypeAlias = Family | str
```

### 9. Validación de Entrada Centralizada

Crear decoradores reutilizables:
```python
@validate_inputs(X="array_2d", y="array_1d", family="family_or_str")
def fit_glm(X, y, family="gaussian", ...):
    ...
```

---

## 📈 Métricas de Código

### Conteo de Archivos por Módulo

| Módulo | .py files | Líneas (aprox) | Tests |
|--------|-----------|----------------|-------|
| core | 10 | ~800 | 8 |
| distributions | 12 | ~1500 | 12 |
| models | 22 | ~5000 | 25 |
| smoothing | 10 | ~2000 | 6 |
| inference | 6 | ~800 | 5 |
| validation | 6 | ~600 | 3 |
| utils | 4 | ~200 | 2 |
| io | 3 | ~10 | 0 |
| visualization | 3 | ~10 | 0 |

**Total**: ~73 archivos Python, ~11,000 líneas de código.

### Cobertura de Tests Estimada

- `distributions/`: ~90% (todas las familias y links)
- `models/glm/`: ~85%
- `models/gam/`: ~75%
- `models/gamm/`: ~70%
- `smoothing/`: ~80%
- `inference/`: ~60%
- `validation/`: ~50%
- `io/`, `visualization/`: 0%

---

## 🛠️ Plan de Refactorización Propuesto

### Fase 1: Limpieza (1-2 días)
1. [ ] Eliminar o documentar módulos vacíos
2. [ ] Consolidar clases Result en jerarquía
3. [ ] Unificar convenciones de nombrado

### Fase 2: Reorganización (2-3 días)
1. [ ] Mover `gamm/laplace.py` → `estimation/laplace/`
2. [ ] Mover `gamm/estimation.py` → `estimation/reml/`
3. [ ] Consolidar plotting en `visualization/`
4. [ ] Mejorar `aurora/__init__.py` con exports principales

### Fase 3: Mejoras de API (3-5 días)
1. [ ] Implementar aliases cortos (Gaussian, Poisson, etc.)
2. [ ] Agregar validación de entrada con decoradores
3. [ ] Crear typing protocols para interfaces

### Fase 4: Completar Funcionalidad (5-10 días)
1. [ ] Implementar `io/` (pandas, R compatibility)
2. [ ] Implementar ANOVA para GLM/GAM
3. [ ] Agregar sensitivity analysis básico
4. [ ] Documentación con Sphinx

---

## 📝 Conclusiones

La arquitectura de Aurora-GLM es **sólida en su núcleo** pero tiene **deuda técnica significativa** en forma de módulos vacíos y dispersión de funcionalidad. La refactorización propuesta mejoraría:

1. **Usabilidad**: API más simple y consistente
2. **Mantenibilidad**: Menos duplicación, mejor organización
3. **Extensibilidad**: Interfaces claras para nuevas familias/links

**Prioridad recomendada**: 
1. Primero consolidar clases Result (impacto alto, riesgo bajo)
2. Luego mejorar exports en `__init__.py` (impacto alto, riesgo bajo)
3. Finalmente reorganizar módulos (impacto medio, riesgo medio)

---

*Generado automáticamente durante revisión de arquitectura - Aurora-GLM v0.5.0-dev*

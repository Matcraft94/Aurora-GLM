# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Validation of Aurora-GLM against R glm() via subprocess (no rpy2 needed).

This test wraps benchmarks/compare_with_r.py and benchmarks/run_r_checks.R.
It auto-skips when R / Rscript is not installed, so it runs on developer
machines with R but stays inert in environments without R.

To run locally:
    1. Install R: https://www.r-project.org/
    2. Install required R packages:
       Rscript -e 'install.packages(c("jsonlite", "mgcv", "lme4"))'
    3. Run: pytest tests/test_models/test_glm_vs_r.py -v
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARE_SCRIPT = REPO_ROOT / "benchmarks" / "compare_with_r.py"
R_SCRIPT = REPO_ROOT / "benchmarks" / "run_r_checks.R"

rscript_available = shutil.which("Rscript") is not None
compare_script_exists = COMPARE_SCRIPT.is_file()
r_script_exists = R_SCRIPT.is_file()

skip_reasons = []
if not rscript_available:
    skip_reasons.append("Rscript not on PATH")
if not compare_script_exists:
    skip_reasons.append(f"{COMPARE_SCRIPT} missing")
if not r_script_exists:
    skip_reasons.append(f"{R_SCRIPT} missing")

R_READY = not skip_reasons
SKIP_REASON = "; ".join(skip_reasons) if skip_reasons else ""


@pytest.mark.skipif(not R_READY, reason=f"R validation unavailable: {SKIP_REASON}")
@pytest.mark.slow
def test_glm_coefficients_match_r(tmp_path):
    """Run the R comparison script and assert coefficients match within 1e-4.

    This test executes ``benchmarks/compare_with_r.py`` which:
    1. Runs ``run_r_checks.R`` to fit GLMs in R on synthetic data
    2. Fits the same data using Aurora-GLM
    3. Compares coefficients, deviance, AIC, fitted values

    Tolerances match those defined in the comparison script:
    - coef-tol: 1e-4
    - deviance-tol: 0.02
    - fitted-tol: 2e-4
    - aic-tol: 500 (kept loose for backward compat; after Phase 1.1 fix the
      Aurora AIC should match R much more closely for non-Gaussian families)

    Failure modes:
    - Non-zero exit from subprocess -> R or Python error in comparison
    - 'success: False' in JSON -> coefficient/deviance/fitted mismatch
    """
    output_json = tmp_path / "comparison_results.json"

    cmd = [
        "python",
        str(COMPARE_SCRIPT),
        "--output",
        str(output_json),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=300,
    )

    if result.returncode != 0:
        pytest.fail(
            f"R comparison script failed (exit {result.returncode}).\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    if not output_json.is_file():
        pytest.fail(
            f"Comparison script did not produce {output_json}.\n"
            f"stdout:\n{result.stdout}"
        )

    data = json.loads(output_json.read_text())
    failed = [r for r in data.get("results", []) if not r.get("success", False)]
    if failed:
        summary = "\n".join(
            f"  - {r.get('family')}/{r.get('link')} rep={r.get('replicate')}: "
            f"coef_max_diff={r.get('coef_max_abs_diff')}, "
            f"deviance_diff={r.get('deviance_abs_diff')}, "
            f"fitted_diff={r.get('mean_fitted_abs_diff')}"
            for r in failed[:5]
        )
        pytest.fail(f"{len(failed)} R comparison(s) failed:\n{summary}")

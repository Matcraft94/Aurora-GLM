# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Sphinx configuration for Aurora-GLM documentation."""

import os
import sys

# Add project root to path so autodoc can find aurora
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "Aurora-GLM"
copyright = "2025, Lucy Eduardo Arias"
author = "Lucy Eduardo Arias"

# Read version from package
__version__ = "0.7.0"
try:
    from aurora import __version__ as _v

    __version__ = _v
except ImportError:
    pass

version = __version__
release = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_title = "Aurora-GLM"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#3f51b5",
        "color-brand-content": "#303f9f",
        "color-brand-secondary": "#5c6bc0",
    },
    "dark_css_variables": {
        "color-brand-primary": "#7986cb",
        "color-brand-content": "#9fa8da",
        "color-brand-secondary": "#5c6bc0",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autoclass_content = "class"
autodoc_preserve_defaults = True

# -- Options for autosummary -------------------------------------------------

autosummary_generate = True
autosummary_imported_members = True

# -- Options for Napoleon (NumPy docstrings) ---------------------------------

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_type_aliases = {
    "ArrayLike": "array_like",
    "ndarray": "numpy.ndarray",
}

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "statsmodels": ("https://www.statsmodels.org/stable", None),
}

# -- Mock imports (optional backends that may not be installed) ---------------

autodoc_mock_imports = ["torch", "jax", "jax.numpy", "numpyro", "pymc", "arviz"]

# -- Options for copybutton --------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"

# -- Options for linkcheck ---------------------------------------------------

linkcheck_ignore = [
    r"https://github\.com/Matcraft94/Aurora-GLM/issues",
]
linkcheck_workers = 5

# -- Source suffix ------------------------------------------------------------

source_suffix = ".rst"
master_doc = "index"

# -- LaTeX -------------------------------------------------------------------

latex_elements = {
    "preamble": r"""
\usepackage{amsmath}
\usepackage{amssymb}
""",
}

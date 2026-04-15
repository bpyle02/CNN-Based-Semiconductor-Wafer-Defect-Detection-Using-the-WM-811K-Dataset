"""Sphinx configuration for the Wafer Defect Detection project."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------
_CONF_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CONF_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

# -- Project information -----------------------------------------------------
project = "Wafer Defect Detection"
author = "AI 570 Team 4"
copyright = "2026, AI 570 Team 4"

# Pull version from src/__init__.py::__version__ if exposed
def _discover_version() -> str:
    fallback = "0.2.0"
    init_path = _REPO_ROOT / "src" / "__init__.py"
    if not init_path.is_file():
        return fallback
    try:
        for line in init_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("__version__"):
                # __version__ = "X.Y.Z"
                _, _, rhs = line.partition("=")
                return rhs.strip().strip("\"'") or fallback
    except OSError:
        return fallback
    return fallback


version = _discover_version()
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "autoapi.extension",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
master_doc = "index"

# -- MyST configuration ------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]
myst_heading_anchors = 3

# -- AutoAPI configuration ---------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../../src"]
autoapi_root = "api"
autoapi_keep_files = False
# Let sphinx-autoapi generate api/index.rst and inject itself into the
# master toctree. This avoids us hand-rolling a shim that autoapi would
# wipe on rebuild (autoapi owns its root directory).
autoapi_add_toctree_entry = True
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
# Do NOT include imported-members: it spawns duplicate object descriptions
# (e.g. src.training.TrainConfig vs src.training.config.TrainConfig) that
# trigger warnings and break `-W` builds in CI.
autoapi_python_class_content = "both"

# -- Napoleon configuration --------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# -- autodoc -----------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "matplotlib",
    "sklearn",
    "scipy",
    "cv2",
    "PIL",
    "tqdm",
    "yaml",
    "seaborn",
    "albumentations",
    "timm",
    "tensorboard",
    "wandb",
    "mlflow",
    "flask",
    "fastapi",
    "uvicorn",
    "grpc",
    "grpc_tools",
]

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"{project} v{version}"
html_show_sourcelink = True

# Suppress common noisy warnings that are informational / out of our control
# (they originate from upstream docstrings rather than Sphinx configuration).
suppress_warnings = [
    "autoapi.python_import_resolution",
    "myst.header",
    "misc.highlighting_failure",
    "ref.python",
    "toc.not_readable",
]

# Default code-block highlight language — prevents Pygments from trying to
# lex shell-style snippets (`!nvidia-smi`, `!pip install ...`) as Python.
highlight_language = "text"

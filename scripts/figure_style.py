"""Shared Matplotlib styling for the analysis figures.

Registers the TeX Gyre Heros font (the sans-serif companion of the A&A text
font) and sets consistent, publication-quality defaults: the manuscript font,
slightly larger tick numbers, and a little padding between the tick numbers and
the axes box. Call :func:`apply` once, immediately after importing Matplotlib
and selecting the backend, before any figure is created.
"""

from __future__ import annotations

import glob
import os

import matplotlib as mpl
from matplotlib import font_manager

_FONT_NAME = "TeX Gyre Heros"

# Directories searched for the TeX Gyre Heros OpenType files. The environment
# variable wins; the TeX Live / texmf locations are the common system fallback,
# so the font is found without any per-machine configuration.
_FONT_DIRS = [
    os.environ.get("GALAXYDISKSIZE_FONT_DIR", ""),
    "/usr/share/texmf/fonts/opentype/public/tex-gyre",
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre",
    "/usr/local/texlive/texmf-dist/fonts/opentype/public/tex-gyre",
]


def _register_font() -> bool:
    """Register the non-condensed TeX Gyre Heros faces; return ``True`` if found."""
    registered = False
    for directory in _FONT_DIRS:
        if not directory or not os.path.isdir(directory):
            continue
        for path in glob.glob(os.path.join(directory, "texgyreheros-*.otf")):
            try:
                font_manager.fontManager.addfont(path)
                registered = True
            except Exception:  # noqa: BLE001 - a missing/odd font file must not break plotting
                pass
    return registered


def apply() -> None:
    """Apply the shared figure style: font, larger tick numbers, tick padding."""
    _register_font()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [_FONT_NAME, "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": _FONT_NAME,
            "mathtext.it": f"{_FONT_NAME}:italic",
            "mathtext.bf": f"{_FONT_NAME}:bold",
            # Larger tick numbers and a little breathing room from the axes box.
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
            "axes.labelsize": 16,
            "legend.fontsize": 12,
        }
    )

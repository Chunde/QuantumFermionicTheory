"""Notebook initialization if mmf_setup.nbinit() fails.

Include something like the following in your first cell:

import mmf_setup;mmf_setup.nbinit()
%pylab inline --no-import-all
from nbimports import *                # Conveniences like clear_output

Note: you must start your Jupyter notebook in this directory for this
to work.

"""

__all__ = [
    "reload",
    "display",
    "clear_output",
    "NoInterrupt",
    "mmf_setup",
    "imcontourf",
]

try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3

import os.path
import sys
import mmf_setup

try:
    mmf_setup.nbinit()
    sys.path.insert(os.path.join(mmf_setup.HGROOT, "mmf-hfb"))
except:
    mmf_setup.nbinit()
    DIR = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(DIR, ".."))

from IPython.display import display, clear_output

from mmfutils.contexts import NoInterrupt
from mmfutils.plot import imcontourf

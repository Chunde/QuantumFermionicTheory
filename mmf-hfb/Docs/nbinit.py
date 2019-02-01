"""Notebook initialization if mmf_setup.nbinit() fails.

Include the following in your first cell:

    from nbinit import *

Note: you must start your Jupyter notebook in this directory for this
to work.

"""
import os.path
import sys
import mmf_setup
try:
    mmf_setup.nbinit()
    sys.path.insert(os.path.join(mmf_setup.HGROOT, 'mmf-hfb'))
except:
    mmf_setup.nbinit(hgroot=False)
    DIR = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(DIR, '..'))

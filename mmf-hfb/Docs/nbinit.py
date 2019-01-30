"""Notebook initialization if mmf_setup.nbinit() fails.

Include the following in your first cell:

    import mmf_setup
    try: mmf_setup.nbinit()
    except: import nbinit

Note: you must start your Jupyter notebook in this directory for this
to work.

"""
import mmf_setup

try:
    mmf_setup.nbinit()
except:
    import sys
    sys.path.insert(0, '../')
    mmf_setup.nbinit(hgroot=False)



# Environment path configuration for testing
import sys
import os
import inspect
from os.path import join
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir+"/../mmf_hfb")


"""Hartree-Fock-Bogoliubov (HFB) code for cold atoms and neutron matter.

This code includes some density functionals for studying the unitary
Fermi gas (UFG) including:

* BdG: Standard BdG equations (Eagles-Leggett model)
* SLDA: Symmetric spin-balanced UFG - superfluid local density approximation.
* ASLDA: Asymetric SLDA (spin unbalanced).

See:
https://bitbucket.org/Chunde/quantum-fermion-theories
"""

# This setup.py file is modified from the description below: see their
# version for documentation links and descriptions
#
# https://github.com/pypa/sampleproject/

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
from io import open

NAME = "mmf-hfb"
URL = 'https://bitbucket.org/Chunde/quantum-fermion-theories'

install_requires = [
    'numpy',
    'scipy>=0.17.1', 
    'matplotlib>=1.5',
    'uncertainties', 
    'sympy',
    'ad',
    'mmf_setup>=0.1.11', 
    'mmfutils>=0.4.8',
    'zope.interface>=3.8.0',
    'numba',
]

test_requires = [
    'flake8',
    'pytest-cov',
    'pytest-runner',
]  

mypackage_root_dir = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(mypackage_root_dir, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

# Read version number from VERSION file
# https://packaging.python.org/guides/single-sourcing-package-version/
with open(path.join(mypackage_root_dir, 'VERSION')) as version_file:
    VERSION = version_file.read().strip()
    
setup(
    name=NAME,
    version=VERSION,
    description='HFB code for the unitary Fermi gas',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url=URL,
    author='The Forbes Group',
    
    # This should be a valid email address corresponding to the author listed
    # above.
    author_email='michael.forbes+python@gmail.com',

    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    # Note that this is a string of words separated by whitespace, not a list.
    keywords='HFB, MMF, physics',
    packages=find_packages(exclude=['_Maple', '_trash', 'Docs', 'tests']),
    install_requires=install_requires,
    extras_require={
        'test': test_requires,
    },
    
    # If there are data files included in your packages that need to be
    # installed, specify them here.
    #
    # If using Python 2.6 or earlier, then these have to be included in
    # MANIFEST.in as well.
    package_data={
        # 'sample': ['package_data.dat'],
    },
    
    project_urls={  # Optional
        'Bug Reports': '/'.join([URL, 'issues']),
        'Source': URL,
    },
)


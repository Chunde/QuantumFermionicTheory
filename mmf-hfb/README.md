# README

This project can be found at https://cocalc.com/projects/89f60b3e-35bd-473e-b0c4-b00d176b9ff8/files/repositories/mmf-hfb/

# Structure

`_ext`: External resources not managed under version control.  For
   example, if we depend on the sources of another project, we will
   clone it here and create appropriate symlinks to the project.  (Tag
   the revision numbers in the `.mrfreeze` file or elsewhere as
   appropriate.
`docs`: Documentation.
`mmf_hfb`: Python source code.
`tests`: Testing.

# Code Structure

`mmf_hfb/interfaces.py`: <incomplete> Defines the public interfaces
  for the code.  Users should make sure that they only use the methods
  defined here.  Developers should make sure they implement all of the
  interface components.

`mmf_hfb/homogeneous.py`: <incomplete> Simplified code for solving the
  functionals for homogeneous matter.  This code performs integrals
  and does not use a basis.  Useful for high-precision testing and for
  getting initial state.
  
`mmf_hfb/hfb.py`: <incomplete> Full HFB code.  This code combines
  functionals like the BdG, SLDA, and ASLDA with bases like DVR and
  Periodic to solve physical problems.
  
Code to be removed:

`mmf_hfb.homogeneous_aslda.py`
`mmf_hfb.homogeneous_mmf.py`






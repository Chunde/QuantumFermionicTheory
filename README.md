# README

This Repo can be find at [https://bitbucket.org/Chunde/quantum-fermion-theories/src/default/](https://github.com/Chunde/QuantumFermionicTheory)

# Structure

`_ext`: External resources not managed under version control.  For
   example, if we depend on the sources of another project, we will
   clone it here and create appropriate symlinks to the project.  (Tag
   the revision numbers in the `.mrfreeze` file or elsewhere as
   appropriate.)
`mmf-hfb`: Python package with full source code for the project.

# Developer Guidelines

* Documentation files should always start with Capital letters in
  CamelCase (sometimes called CapWords) so that the jupytext files are
  not confused with python code.
* Python code should always be lower-case with underscores to
  differentiate it from jupytext documentation.
  
  
* Please follow all guidelines posted here:

  * [Coding Standards](http://labs.wsu.edu/forbes/public/student_resources/prerequisites/#Coding)

  For example:
  
  * Please make sure your code passes Flake8 tests.  You can check by
    running something like:
  
    ```bash
    flake8 mmf_hfb/integrate.py
    ```

    If you need to ignore a check, please explicitly add this to
    `setup.cfg` with an explanation.

   * Provide tests for your code.  Aim to get as close to 100% code
     coverage as possible.

  * Please commit your code with messages that conform to the
    instructions posted on the following page:
  

### What is this repository for? ###

I probably will put other things such as some testing code, some write-up for the future thesis here.


BCS
Homogeneous
DVR
ASLDA

FFLO
Quantum Friction


### What is this repository for? ###

* Quick summary
* Version
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact

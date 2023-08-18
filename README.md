# CPSRx4_pytorch

The software allows to test CPSRx4 from Python.


## Pre-requisites:

1. Python
	-> version 3.6 or above
	-> http://www.python.org

2. SWIG
	-> version 1.3 or above
	-> http://www.swig.org

3. Numpy
	-> module must be available in your PYTHONPATH environment variable in order to be found by 
	-> version 1.0 or above
	-> http://numpy.scipy.org/

4. CurveLab
	-> version 2.0.2 or above
	-> http://www.curvelet.org

***

## Installation:

1. Clone this repsitory.

2. Set these required environment variables:

	- FDCT: folder where your CurveLab installation is
	- FFTW: folder where your fftw installation is

3. In the PyCurvelab folder, run the following command:

	- python setup.py build install
	- the package will be installed as pyct module

4. In python, simply "import pyct" and you're off

5. To see how to use, type "help(pyct.fdct2)" or "help(pyct.fdct3)"

***

`setup.py uses python's distutils, which offers many options for a customized installation.
run "python setup.py install --help" for more information`

***

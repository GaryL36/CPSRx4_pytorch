# CPSRx4_pytorch

The software allows to test CPSRx4 from Python.


## Pre-requisites:

1. Python
	-> version 3.6 or above
	-> http://www.python.org

2. Pytorch
	-> version 1.10.0 or above
	-> https://pytorch.org/

3. SWIG
	-> version 1.3 or above
	-> http://www.swig.org

4. Numpy
	-> module must be available in your PYTHONPATH environment variable in order to be found by 
	-> version 1.0 or above
	-> http://numpy.scipy.org/

5. CurveLab
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

## Tips and Tricks for Dependencies

### FFTW

For FFTW 2.1.5, you must compile with position-independent code support. Do that with

```bash
./configure --with-pic --prefix=/home/user/opt/fftw-2.1.5 --with-gcc=/usr/bin/gcc
```

The `--prefix` and `--with-gcc` are optional and determine where it will install FFTW and where to find the GCC compiler, respectively. We recommend using the same compile for FFTW, CurveLab and `curvelops`.

### CurveLab

In the file `makefile.opt` set `FFTW_DIR`, `CC` and `CXX` variables as required in the instructions. To keep things consistent, set `FFTW_DIR=/home/user/opt/fftw-2.1.5` (or whatever directory was used in the `--prefix` option). For the others, use the same compiler which was used to compile FFTW.

### curvelops

The `FFTW` variable is the same as `FFTW_DIR` as provided in the CurveLab installation. The `FDCT` variable points to the root of the CurveLab installation. It will be something like `/path/to/CurveLab-2.1.3` for the latest version.

## Disclaimer

This package contains no CurveLab code apart from function calls. It is provided to simplify the use of CurveLab in a Python environment. Please ensure you own a CurveLab license as per required by the authors. See the [CurveLab website](http://curvelet.org/software.html) for more information. All CurveLab rights are reserved to Emmanuel Candes, Laurent Demanet, David Donoho and Lexing Ying.


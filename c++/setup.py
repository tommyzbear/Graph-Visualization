#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# sparse layout extension module
_sparse_layout = Extension(
    name="_sparse_layout",
    sources=["SparseLayout_wrap.cxx", "SparseLayout.cpp"],
    libraries=['tbb'],
    # extra_compile_args=["-O3"],
    include_dirs=[numpy_include],
)

# sparse layout setup
setup(name="Sparse Layout",
      description="Input takes in a random 2D arrays and Sparse matrix contain entries of edges, then return a graph layout in 2D space",
      author="Heran Zhang",
      version="1.0",
      ext_modules=[_sparse_layout]
      )

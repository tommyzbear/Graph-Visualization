# Sparse Layout
`sparse_layout` is a C++ extension package for Python which is a drawing tool using SGD drawing algorithm incorporating with Sparse Stress Model.

# Build
## Requirement
`sparse_layout` require SWIG interface, Python3 and GCC 7.3 onwards.

## Compiling
First we need to use the SWIG interface to generate the wrapper for our C++ implementations using the below command
`swig -c++ -python SparseLayout.i`
Then we will execute the `setup.py` to build our Python package incoporating with GCC compiler.
`python3 setup.py build`
After the code being compiled successfully, a `build` folder will be generated and move the package file under the `lib*` subdirectory to your base directory where your Python script locates.

## Import
`import sparse_layout as cpp`

## Input format
The input for the graph drawing methods are
* `X` random 2D numpy array
* `I, J` list of vertices where an edge is connected between `i, j` 
* `V` weight of the edge (for weighted graph only)
* `sampling_scheme` a string providing the sampling scheme you want to use for Sparse Stress Layout, currently there are a few options available `"random", "mis", "max_min_sp", "max_min_random_sp", "max_min_euc"`
* `k` an integer justifies the number of pivots
* `sgd_iter` an integer justifies the number of SGD iterations
* `eps` a float justifies the epsilon value for SGD

## List of functions
`
cpp.sparse_layout_naive_unweighted(X, I, J, sampling_scheme, k, sgd_iter, eps)
cpp.sparse_layout_MSSP_unweighted(X, I, J, sampling_scheme, k, sgd_iter, eps)
cpp.layout_unweighted(X, I, J, sgd_iter, eps)
cpp.layout_weighted(X, I, J, V, sgd_iter, eps)
cpp.sparse_layout_naive_weighted(X, I, J, V, sampling_scheme, k, sgd_iter, eps)
cpp.sparse_layout_MSSP_weighted(X, I, J, V, sampling_scheme, k, sgd_iter, eps)
`

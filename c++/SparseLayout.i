%module sparse_layout
%{
    #define SWIG_FILE_WITH_INIT
    extern void layout_unweighted(int n, double *X, int m, int *I, int *J, int t_max, double eps);
    extern void layout_weighted(int n, double *X, int m, int *I, int *J, double *V, int t_max, double eps);
    extern void sparse_layout_naive_unweighted(int n, double *X, int m, int *I, int *J, char *sampling_scheme, int k, int t_max, double eps);
    extern void sparse_layout_MSSP_unweightd(int n, double *X, int m, int *I, int *J, char *sampling_scheme, int k, int t_max, double eps);
    extern void sparse_layout_naive_weighted(int n, double *X, int m, int *I, int *J, double *V, char *sampling_scheme, int k, int t_max, double eps);
    extern void sparse_layout_MSSP_weightd(int n, double *X, int m, int *I, int *J, double *V, char *sampling_scheme, int k, int t_max, double eps);
    extern void stress_unweighted(int n, double *X, int m, int *I, int *J);
    extern void stress_weighted(int n, double *X, int m, int *I, int *J, double *V);
%}

%include "numpy.i"
%init %{
    import_array();
%}

//vertex positions, where the positions are changed after SGD
%apply (double *INPLACE_ARRAY2, int DIM1, int DIM2){(double *X, int n, int kd)}

//edge indices, fixed value
%apply (int *IN_ARRAY1, int DIM1){(int *I, int len_I), (int *J, int len_J)}

//edge weights, fixed value
%apply (double *IN_ARRAY1, int DIM1){(double *V, int len_V)}

//direct MDS
%apply (double *IN_ARRAY1, int DIM1){(double *d, int len_d), (double *w, int len_w), (double *eta, int len_eta)}

// string handling
%apply (char *STRING, int LENGTH) { (char *data, int size) };

extern void layout_unweighted(int n, double *X, int m, int *I, int *J, int t_max, double eps);
extern void layout_weighted(int n, double *X, int m, int *I, int *J, double *V, int t_max, double eps);
extern void sparse_layout_naive_unweighted(int n, double *X, int m, int *I, int *J, char *sampling_scheme, int k, int t_max, double eps);
extern void sparse_layout_MSSP_unweightd(int n, double *X, int m, int *I, int *J, char *sampling_scheme, int k, int t_max, double eps);
extern void sparse_layout_naive_weighted(int n, double *X, int m, int *I, int *J, double *V, char *sampling_scheme, int k, int t_max, double eps);
extern void sparse_layout_MSSP_weightd(int n, double *X, int m, int *I, int *J, double *V, char *sampling_scheme, int k, int t_max, double eps);
extern void stress_unweighted(int n, double *X, int m, int *I, int *J);
extern void stress_weighted(int n, double *X, int m, int *I, int *J, double *V);

%rename (layout_unweighted) np_layout_unweighted;
%exception np_layout_unweighted{
    $action
    if(PyErr_Occurred()) SWIG_fail;
}
%rename (layout_weighted) np_layout_weighted;
%exception np_layout_weighted{
    $action
    if(PyErr_Occurred()) SWIG_fail;
}
%rename (sparse_layout_naive_unweighted) np_sparse_layout_naive_unweighted;
%exception np_sparse_layout_naive_unweighted{
    $action
    if(PyErr_Occurred()) SWIG_fail;
}
%rename (sparse_layout_MSSP_unweightd) np_sparse_layout_MSSP_unweighted;
%exception np_sparse_layout_MSSP_unweighted{
    $action
    if(PyErr_Occurred()) SWIG_fail;
}
%rename (sparse_layout_naive_weighted) np_sparse_layout_naive_weighted;
%exception np_sparse_layout_naive_weighted{
    $action
    if(PyErr_Occurred()) SWIG_fail;
}
%rename (sparse_layout_MSSP_weightd) np_sparse_layout_MSSP_weighted;
%exception np_sparse_layout_MSSP_weighted{
    $action
    if(PyErr_Occurred()) SWIG_fail;
}
%rename (stress_unweighted) np_stress_unweighted;
%exception np_stress_unweighted{
    $action
    if(PyErr_Occurred()) SWIG_fail;
}
%rename (stress_weighted) np_stress_weighted;
%exception np_stress_weighted{
    $action
    if(PyErr_Occurred()) SWIG_fail;
}

%inline %{
    void dimension_check(int kd){
        if (kd != 2){
            PyErr_Format(PyExc_ValueError, "only 2D positions are currently supported");
            return;
        }
    }
    void unweighted_edge_check(int len_I, int len_J){
        if (len_I != len_J){
            PyErr_Format(PyExc_ValueError, "arrays of indices do not have same length");
        }
    }
    void weighted_edge_check(int len_I, int len_J, int len_V) {
        if (len_I != len_J || len_J != len_V) {
            PyErr_Format(PyExc_ValueError, "arrays of indices do not have same length");
            return;
        }
    }
    void pivot_check(int k, int n){
        if (k > n){
            PyErr_Format(PyExc_ValueError, "number of pivots cannot exceed number of vertices");
        }
    }
    void np_layout_unweighted(double *X, int n, int kd, int *I, int len_I, int *J, int len_J, int t_max, double eps){
        dimension_check(kd);
        unweighted_edge_check(len_I, len_J);
        layout_unweighted(n, X, len_I, I, J, t_max, eps);
    }
    void np_layout_weighted(double *X, int n, int kd, int *I, int len_I, int *J, int len_J, double *V, int len_V, int t_max, double eps){
        dimension_check(kd);
        weighted_edge_check(len_I, len_J, len_V);
        layout_weighted(n, X, len_I, I, J, V, t_max, eps);
    }
    void np_sparse_layout_naive_unweighted(double *X, int n, int kd, int *I, int len_I, int *J, int len_J, char *data, int size, int k, int t_max, double eps){
        dimension_check(kd);
        unweighted_edge_check(len_I, len_J);
        pivot_check(k, n);
        sparse_layout_naive_unweighted(n, X, len_I, I, J, data, k, t_max, eps);
    }
    void np_sparse_layout_MSSP_unweighted(double *X, int n, int kd, int *I, int len_I, int *J, int len_J, char *data, int size, int k, int t_max, double eps){
        dimension_check(kd);
        unweighted_edge_check(len_I, len_J);
        pivot_check(k, n);
        sparse_layout_MSSP_unweightd(n, X, len_I, I, J, data, k, t_max, eps);
    }
    void np_sparse_layout_naive_weighted(double *X, int n, int kd, int *I, int len_I, int *J, int len_J, double *V, int len_V, char *data, int size, int k, int t_max, double eps){
        dimension_check(kd);
        weighted_edge_check(len_I, len_J, len_V);
        pivot_check(k, n);
        sparse_layout_naive_weighted(n, X, len_I, I, J, V, data, k, t_max, eps);
    }
    void np_sparse_layout_MSSP_weighted(double *X, int n, int kd, int *I, int len_I, int *J, int len_J, double *V, int len_V, char *data, int size, int k, int t_max, double eps){
        dimension_check(kd);
        weighted_edge_check(len_I, len_J, len_V);
        pivot_check(k, n);
        sparse_layout_MSSP_weightd(n, X, len_I, I, J, V, data, k, t_max, eps);
    }
    void np_stress_unweighted(double *X, int n, int kd, int *I, int len_I, int *J, int len_J){
        stress_unweighted(n, X, len_I, I, J);
    }
    void np_stress_weighted(double *X, int n, int kd, int *I, int len_I, int *J, int len_J, double *V, int len_V){
        stress_weighted(n, X, len_I, I, J, V);
    }
%}
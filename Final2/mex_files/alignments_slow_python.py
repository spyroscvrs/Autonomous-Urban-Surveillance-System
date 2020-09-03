import sys, platform
import ctypes, ctypes.util
from ctypes import POINTER, c_double, c_int,byref
import numpy as np
# mylib_path = ctypes.util.find_library("./mylib.so")

# if not mylib_path:
#     print("Unable to find the specified library.")
#     sys.exit()

try:
    mylib = ctypes.CDLL("./mex_files/alignments_slow.so")
except OSError:
    
    print("Unable to load the system C library", OSError.strerror)
    sys.exit()

#free_mem = mylib.free_mem
#free_mem.argtypes = [POINTER(c_double)]
#free_mem.restype = None

alignments_slow_fync = mylib.mexFunction_alignment_slow
alignments_slow_fync.argtypes= [POINTER(c_double), c_int, c_int,POINTER(POINTER(c_double)),POINTER(c_int),POINTER(c_int)]

def use_alignments_slow(input_vect, x_col, x_row):
    input_d = np.array(input_vect, float)
    inp_d_c = input_d.ctypes.data_as(POINTER(c_double))
    x_in = c_int(x_col)
    n_in = c_int(x_row)
    
    out_d_c = POINTER(c_double)()
    x_out = c_int()
    n_out = c_int()
    alignments_slow_fync(inp_d_c, x_in, n_in, byref(out_d_c), byref(x_out), byref(n_out))
    return [np.round(out_d_c[i],100) for i in range(x_out.value * n_out.value)], n_out.value
#    free_mem(inp_d_c)
    
    
#    void  mexFunction(double * input_points, int X_in, int N_in, double ** output_points, int *X_out, int *N_out)
#    return [np.round(result_ctype[i],100) for i in range(6 * x_row)]
#a = use_alignments_slow(list(np.array(np.random.randint(0,512, size=(1000)), float)),2, 500)
#print(np.shape(res))
#print(res)
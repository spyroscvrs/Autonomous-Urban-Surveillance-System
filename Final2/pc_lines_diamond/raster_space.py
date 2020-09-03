import sys, platform
import ctypes, ctypes.util
from ctypes import POINTER, c_double, c_int,byref, c_float, c_int8,CDLL
import numpy as np
from ctypes.util import find_library
# mylib_path = ctypes.util.find_library("./mylib.so")

# if not mylib_path:
#     print("Unable to find the specified library.")
#     sys.exit()
libc = CDLL(find_library("c"))
try:
    print('importing mx_raster_space')
    mylib = ctypes.CDLL("./pc_lines_diamond/lib/mx_raster_space.so")
except OSError:
    
    print("Unable to load the system C library", OSError.strerror)
    sys.exit()

#free_mem = mylib.free_mem
#free_mem.argtypes = [POINTER(c_double)]
#free_mem.restype = None

mexFunction_fync = mylib.mexFunction
#                                      [LinesData,    spaceSIze  ,numLines, pSace_out]           
mexFunction_fync.argtypes= [POINTER(c_float), POINTER(c_int), c_int, POINTER(POINTER(c_int))]
#void mexFunction(float * LinesData, int * SpaceSize, int numLines, int ** pSpace_out)

free_int_fync = mylib.free_int_array
free_int_fync.argtypes = [POINTER(POINTER(c_int))]

#mexFunction1 = mylib.mexFunction1
def use_raster_space(linesData, space_size, numlines):
    linesData_d = np.array(linesData, np.float32)
    linesData_c = linesData_d.ctypes.data_as(POINTER(c_float))
    
    space_size_d = ctypes.c_int * len(space_size)
    space_size_c = space_size_d(*space_size)
    
    num_lines_c = c_int(numlines)
    
    out_d_c = POINTER(c_int)()
    
#    mexFunction1()
    mexFunction_fync(linesData_c, space_size_c, num_lines_c, byref(out_d_c))
    result = [out_d_c[i] for i in range(space_size[0] * space_size[1])]
    libc.free(out_d_c)
    
#    free_int_fync(byref(out_d_c));
    return result
#

def getdata():
    ac = np.random.randint(0, 10, [100, 2])
    b = np.ones(100)
    lines = np.c_[ ac[:, 0],b, ac[:,1], b] # last one is for w
    return lines.T, len(lines)

if __name__ =="__main__":
    lines, nlines = getdata()
#    print(lines[:, :10])
    res = use_raster_space(lines.T.ravel(), [321, 321], nlines)
#    print(res[:10])
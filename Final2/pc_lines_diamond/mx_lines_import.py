# -*- coding: utf-8 -*-

import sys, platform
import ctypes, ctypes.util
from ctypes import POINTER, c_double, c_int,byref, c_float, c_int8
import numpy as np
import cv2

# mylib_path = ctypes.util.find_library("./mylib.so")

# if not mylib_path:
#     print("Unable to find the specified library.")
#     sys.exit()

try:
    mylib = ctypes.CDLL("./pc_lines_diamond/lib/mx_lines.so")
except OSError:
    
    print("Unable to load the system C library", OSError.strerror)

#free_mem = mylib.free_mem
#free_mem.argtypes = [POINTER(c_double)]
#free_mem.restype = None

mexFunction_fync = mylib.mexFunction
mexFunction_fync.argtypes= [POINTER(c_int), c_int, c_int, POINTER(c_int), c_int, POINTER(POINTER(c_float)),  POINTER(c_int)]
#void mexFunction(int * data, int width, int height, int * rads, int rads_size, float * mx_lines_data)

#mexFunction1 = mylib.mexFunction1
def use_mx_lines(imageData, width,height, rads,rads_size):
    imageData_d = ctypes.c_int * len(imageData)
    imageData_c = imageData_d(*imageData)
    
    width_c = c_int(width)
    height_c = c_int(height)
    
    print(rads)
    rads_d = ctypes.c_int *len(rads)
    rads_c = rads_d(*rads)
    
    rads_size_c = c_int(rads_size)
    
    out_d_c = POINTER(c_float)()
    lines_num_out = c_int()
    
#    mexFunction1()
    mexFunction_fync(imageData_c, width_c, height_c, rads_c, rads_size_c, byref(out_d_c), byref(lines_num_out))
    return [out_d_c[i] for i in range(lines_num_out.value * 4)], lines_num_out.value


def getdata():
    img = cv2.imread('/media/ixtiyor/New Volume/datasets/bdd/bdd100k_images/bdd100k/images/10k/test/af0a7e94-89b00000.jpg', 0)
    edges = cv2.Canny(img,100,200)
    return edges

if __name__ =="__main__":
    edges = getdata()
    height, width = np.shape(edges)
    edges = np.int32(edges/255)
    patches = np.array(np.arange(6, 26, 4), np.int32)
    res,len_lines = use_mx_lines(edges.ravel(),width, height, patches, len(patches))
    print(res)
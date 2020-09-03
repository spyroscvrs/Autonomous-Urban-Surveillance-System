import cv2
import os
import numpy as np
from pc_lines_diamond.mx_lines import  fit_ellipse, gety,get_coeffs
from pc_lines_diamond.ransac.ransac import run_ransac, estimate, is_inlier
import math
import time
#def length(v):
#  return math.sqrt(dotproduct(v, v))
def auto_canny(image, sigma=0.1):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged
def get_orientation(mag, ori,thresh=1.0, num_bins=8):
    h, w = np.shape(mag)
    orientation_map = np.zeros((*np.shape(mag), num_bins))
    for i in range(h):
        for j in range(w):
            mag_pixel = mag[i,j]
            if(mag_pixel > thresh):
                 oriPixel = ori[i,j]
                 bins = np.arange(0,361,360/(num_bins))
                 for ib in range(len(bins)-1):
                     if(ib ==1 or len(bins)-1 - 1):continue
#                     print('checking from ', bins[ib] , ' to ', bins[ib+1] )
                     if(oriPixel >= bins[ib] and oriPixel < bins[ib+1]):
                         
                         orientation_map[i,j,ib] = mag_pixel
                         break
#            print(i, j)
    return orientation_map
def get_orientation_matrix_way(mag, ori,thresh=1.0, num_bins=8):
    h, w = np.shape(mag)
    orientation_map = np.zeros((*np.shape(mag), num_bins))
    bins = np.arange(0,361,360/(num_bins))
    for ib in range(len(bins)-1):
        rws,cols = np.where((ori>bins[ib]) & (ori<bins[ib+1]))
        orientation_map[rws,cols, ib] = mag[rws,cols]
    return orientation_map
def background_test(B, H, t1=11065969, t2=12065969):
    diff = abs(H - B)
#    img = np.zeros((*np.shape(H), 3))
#    min_val = np.min(H)
#    print(np.max(diff), np.min(diff))
    rws_t2,cols_t2,bins_t2 = np.where(diff < t2)
    rws_t1,cols_t1,bins_t1 = np.where(H<t1)
#    rws_t3,cols_t3,bins_t3 = np.where()
    H[rws_t1, cols_t1, bins_t1] = 0
#    H[rws_t3, cols_t3, bins_t3] = 0
    H[rws_t2, cols_t2, bins_t2] = 0
#    imgp[rws_t2, cols_t2] = []
    return H

if(__name__ == '__main__'):
    try:
        video_src = "test_simple_3.mp4"
    except:
        video_src = 0
        
    cam = cv2.VideoCapture(video_src)
    _ret, frame = cam.read()
    width = frame.shape[1]
    height = frame.shape[0]
    #i = 0
    B = []
    alpha = 0.99
    while True:
        _ret, frame = cam.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        start = time.time()
        frame_gray = auto_canny(frame_gray, 0.1)
        sobelx = cv2.Sobel(frame_gray,cv2.CV_64F,1,0,ksize=11)
        sobely = cv2.Sobel(frame_gray,cv2.CV_64F,0,1,ksize=11)
        
        magnitude = cv2.magnitude(sobelx, sobely) # computes sqrt(xi^2 + yi^2)
        phase = cv2.phase(sobelx,sobely,angleInDegrees=True) # computes angel between x and y
        
        H = get_orientation_matrix_way(magnitude, phase,1.0,8)
        
        if(len(B) == 0):
            B = H
        else:
            B = alpha * B + (1-alpha) * H
        
        H = background_test(B, H)
        H = np.sum(H,2)
        H = (H/np.max(H) * 255).astype('uint8')
        #vp_1 = [width/2, 0]
        print('time spent', time.time() - start)
        cv2.imshow('edges',H)
        
        ch = cv2.waitKey(1)
        if ch == 27:
            cv2.destroyAllWindows()        
            break
    

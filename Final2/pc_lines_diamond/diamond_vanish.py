# -*- coding: utf-8 -*-
import functools
import numpy as np
from pc_lines_diamond.raster_space import use_raster_space
from pc_lines_diamond.mx_lines_import import use_mx_lines
import math
import cv2
# there is a problem with sign function in python so this is a workaround
# https://stackoverflow.com/questions/1986152/why-doesnt-python-have-a-sign-function
sign = functools.partial(math.copysign, 1)
def auto_canny(image, sigma=0.43):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged
def diamond_vanish(image,  Normalization, SpaceSize, VanishNumber,oldspace = []):
    """
        Detection of vanishing point using diamond space
        input:
            lines           float array;
                            shape [a,b,c,w] x n, where ax + by + c = 0
                            if we use np.polyfit, then b is always 1, we cannot go back
            normalization   float number;
                            normlalization of the image (1 means normalization from -1 to 1)
            SpaceSize       int number;
                            resulution of the accumulation space (final space has dims SpaceSize x SpaceSize)
        output:
            Results structure with fields:
            Results.Space         accumulated diamond space (further is used for orthogonalization)
            Results.PC_VanP       positions of the maxima in R.Space
            Results.PC_VanP_Norm  normalized position of the maxima (R.Space bounded from -1 to 1)
            Results.CC_VanP       position of the vanishing point in the input image coordinates
    """
#    VanishNumber = 3
#    Normalization = 0.4
#    SpaceSize = 321
#    image = np.zeros((400, 400,3), np.uint8);
#    for i in range(400):
#        if(i>250):
#            image[i  , i,:] = [0,0,255]
#            image[i  , 400-i-1,:] = [0,0,255]
    
    result = {}#{'Space':[], 'PC_VanP':[],'PC_VanP_Norm':[],'CC_VanP':[]}
    result['PC_VanP'] = np.zeros([VanishNumber, 2])
    result["PC_VanP_Norm"] = np.zeros([VanishNumber, 2])
    result["CC_VanP"] = np.zeros([1, 2])
    patches = np.array(np.arange(6, 26, 4), np.int32)
#    if(len(lines) == 0):
    edges = auto_canny(np.array(image,np.uint8), 0.1)#cv2.Canny(np.array(image,np.uint8))
#    edges = cv2.Canny(np.array(image,np.uint8), 0, 255, 3, L2gradient=False)
    
    edges = np.int32(edges/255)
#    
#    edges = np.uint8(edge)
#    image = np.uint8(img*255)
    edgest = np.pad(edges, (patches[-1],patches[-1]), pad_with, padder=0)
    res,len_lines = use_mx_lines(edgest.ravel(),edgest.shape[0],edgest.shape[1], patches, len(patches))
#        if(len_lines <= 0):
#            return result,[], edges
    lines  = np.array_split(res, len_lines, axis=0)
    lines = np.array(lines)
#    if(len(oldlines)> 0):
#        lines = np.r_[lines,oldlines]
    
    # i do not understand how this may help ???
    lines[:,2] =(lines[:,2])* Normalization
    returning_lines = lines
    
    subpixelradius = 2
    threshold = 0.05
    
    
#    v = 0
    for v in range(VanishNumber):
        print(np.shape(lines))
        space = use_raster_space(lines.ravel(), [SpaceSize,SpaceSize],len(lines))
        
        space = np.reshape(space , (SpaceSize, SpaceSize)).T
#        if(not len(oldspace) == 0):
#            space = space + oldspace
    #        print(np.shape(find_maximum(space, subpixelradius)))
        if(v==0):
            result["Space"] = space 
        result['PC_VanP'][v,:] = find_maximum(space, subpixelradius)
        
#        result["Space"] = space
        result["PC_VanP_Norm"][v,:] =  normalize_PC_points(result['PC_VanP'][v,:], SpaceSize)
        
#         get lines close to VP
#         we are giving lines as n x 3 
#        print(np.shape(lines))
        distance = point_to_lines_dist(result["PC_VanP_Norm"][v,:], lines[:,0:3]) # shu method da xato bor 
#         remove lines
        args= np.where(distance < threshold)[0]
#        print('before lines shape = ', np.shape(lines), np.shape(args))
         
        lines = np.delete(lines, args, 0)
#        print('after lines shape = ', np.shape(lines), np.shape(args))
#        
        
    result["CC_VanP"] = PC_point_to_CC(Normalization, result["PC_VanP_Norm"], image.shape[:-1]) 
#    resvps = np.int32(abs(result["CC_VanP"]))
#        print("detected vp =====", resvps)
#    try:
#    for i in range(len(resvps[resvps[:,0] >0])):
#        cv2.circle(image, (resvps[i][0], resvps[i][1]), 5, [0,0,255], -1)
##    except:
##        pass
#    cv2.imshow("edges", np.float32(edges * 255))
#    cv2.imshow('grey', image)
#    ch = cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    print(result["CC_VanP"])
    return result, returning_lines, edges
def diamond_vanish_with_lines(lines,width, height,  Normalization, SpaceSize, VanishNumber,oldspace = []):
    """
        Detection of vanishing point using diamond space
        input:
            lines           float array;
                            shape [a,b,c,w] x n, where ax + by + c = 0
                            if we use np.polyfit, then b is always 1, we cannot go back
            normalization   float number;
                            normlalization of the image (1 means normalization from -1 to 1)
            SpaceSize       int number;
                            resulution of the accumulation space (final space has dims SpaceSize x SpaceSize)
        output:
            Results structure with fields:
            Results.Space         accumulated diamond space (further is used for orthogonalization)
            Results.PC_VanP       positions of the maxima in R.Space
            Results.PC_VanP_Norm  normalized position of the maxima (R.Space bounded from -1 to 1)
            Results.CC_VanP       position of the vanishing point in the input image coordinates
    """
#    VanishNumber = 3
#    Normalization = 0.4
#    SpaceSize = 521
#    width = 400
#    height = 400
#    image = np.zeros((400, 400,3), np.uint8);
#    for i in range(400):
#        if(i<159):
#            image[i  , i,:] = [255,255,255]
#            image[i  , 400-i-1,:] = [255,255,255]
    result = {}#{'Space':[], 'PC_VanP':[],'PC_VanP_Norm':[],'CC_VanP':[]}
    result['PC_VanP'] = np.zeros([VanishNumber, 2])
    result["PC_VanP_Norm"] = np.zeros([VanishNumber, 2])
    result["CC_VanP"] = np.zeros([1, 2])
#    lines[:, :2] = np.round(lines[:, :2],4)
#    print(type(lines))
    lines[:,2] =lines[:,2]* Normalization
#    lines = np.round(lines, 4)
    subpixelradius = 8
#    threshold = 0.05
#    v = 0
#    for v in range(VanishNumber):
#        print(np.shape(lines))
    space = use_raster_space(lines.ravel(), [SpaceSize,SpaceSize],len(lines))
    
    space = np.reshape(space , (SpaceSize, SpaceSize)).T
    if(len(space[0])>0 and len(oldspace) != 0):
            space = space + oldspace
#        print(np.shape(space))
#    if(v==0):
    result["Space"] = space 
    result['PC_VanP'][0,:] = find_maximum(space, subpixelradius)
    
#        result["Space"] = space
    result["PC_VanP_Norm"][0,:] =  normalize_PC_points(result['PC_VanP'][0,:], SpaceSize)
    
#         get lines close to VP
#         we are giving lines as n x 3 
#        print(np.shape(lines))
#    distance = point_to_lines_dist(result["PC_VanP_Norm"][0,:], lines[:,0:3]) # shu method da xato bor 
#         remove lines
#    args= np.where(distance < threshold)[0]
#        print('before lines shape = ', np.shape(lines), np.shape(args))
     
#    lines = np.delete(lines, args, 0)
#        print('after lines shape = ', np.shape(lines), np.shape(args))
#        
        
    result["CC_VanP"] = PC_point_to_CC(Normalization, result["PC_VanP_Norm"], [height,width ]) 
    
#
#    cv2.imshow("img", np.uint8(space)*255);cv2.waitKey(0);
#    cv2.destroyAllWindows()
#    resvps = np.int32(abs(result["CC_VanP"]))-1
#        print("detected vp =====", resvps)
#    try:
#    for i in range(len(resvps)):
#        cv2.circle(image, (resvps[i][0], resvps[i][1]), 5, [255,255,255], -1)
#    except:
#        pass
#    cv2.imshow("edges", np.float32(edges * 255))
#    cv2.imshow('grey', image)
#    ch = cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    print(result["CC_VanP"])
    return result

def PC_point_to_CC(normalization, vanishpc, imgshape):
    u = vanishpc[:, 0]
    v = vanishpc[:, 1]
    
#    print(u, v)
   
#    print(np.sign(v) * v + np.sign(u) * u - 1)
    normvanishcc = np.c_[v, np.sign(v) * v + np.sign(u) * u - 1, u]
    
    reg = np.where(abs(normvanishcc[:,2]) > 0.005)[0]
    noreg = np.where(abs(normvanishcc[:,2]) <= 0.005)[0]
#    print(np.shape(abs(normvanishcc[:,2]) ),reg, noreg)
#    if(len(reg) > 0):
    normvanishcc[reg, :] = np.apply_along_axis(np.divide, 0, normvanishcc[reg,:], normvanishcc[reg,2])
    if(len(noreg)> 0):
        normvanishcc[noreg, :] = normr(normvanishcc[noreg,:])
    
    vanishcc =normvanishcc
    if(len(noreg)> 0):
        vanishcc[noreg, 2] = 0
    
    w = imgshape[1]
    h = imgshape[0]
    m = max(w, h)
    vanishcc[reg,0] = (vanishcc[reg,0]/normalization * (m-1) + w + 1) / 2
    vanishcc[reg,1] = (vanishcc[reg,1]/normalization * (m-1) + h + 1) / 2
    return vanishcc
    
def normr(data):
    """
    normalize data by rows
    data is n x m shape
    n is the number of rows
    """
    newdata = [ row/np.sqrt(np.sum(np.square(row))) for row in data]
    return newdata

def point_to_lines_dist(Point, Lines):
    """
    lines are n x 3 format
    """
    
#    print(Point)
#    print(np.shape(Lines))
    
    x = Point[0]
    y = Point[1]
    
    T = np.array([
         [0,-1,1],
         [1,-1,0],
         [0,-1,0],
         [0,-1,1],
         [1,1,0 ],
         [0,-1,0],
         [0,1,1 ],
         [1,-1,0],
         [0,-1,0],
         [0,1,1 ],
         [1,1,0],
         [0,-1,0]
        ])
    L = np.dot(Lines, T.T)
    p = np.zeros([len(Lines), 4])
    p[:, 0] = np.dot(L[:, 0:3], np.array([x, y, 1]))/np.sqrt(np.sum(np.square(L[:,0:2]),1))
    p[:, 1] = np.dot(L[:, 3:6], np.array([x, y, 1]))/np.sqrt(np.sum(np.square(L[:,3:5]),1))
    p[:, 2] = np.dot(L[:, 6:9], np.array([x, y, 1]))/np.sqrt(np.sum(np.square(L[:,6:8]),1))
    p[:, 3] = np.dot(L[:, 9:12], np.array([x, y, 1]))/np.sqrt(np.sum(np.square(L[:,9:11]),1))
    
    D = np.min(abs(p),1)
    return D
    
def normalize_PC_points(VanP, spacesize):
    VanP = np.array(VanP) + 1
    normvp = (2*VanP - (spacesize+1))/(spacesize-1)
    return normvp

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value        

def find_maximum(space, R):
    r,c = np.where(space==np.max(space) )
#    print(r, c)
    S = np.pad(space, (R,R), pad_with, padder=0)
    r = r
    c = c
    
    O = S[r[0]:r[0] + R*2+1, c[0]:c[0]+R*2+1]
    
    
    mc, mr = np.meshgrid(np.arange(-R,R+1),np.arange(-R,R+1))
    
    SR = O * mr
    SC = O * mc
    
    C = c[0] + np.sum(SC[:])/np.sum(O[:])
    R = r[0] + np.sum(SR[:])/np.sum(O[:])

#    print(np.shape(C), np.shape(R))
#    print(C)

    VanPC = [C,R]
    return VanPC;
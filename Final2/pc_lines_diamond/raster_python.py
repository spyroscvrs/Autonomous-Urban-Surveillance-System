# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import cv2
from pc_lines_diamond.raster_space import use_raster_space
def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l

def convert_line_to_pc_line_endpoint(line, space_c):
    """
    line is 4 value array where 4 values are:
        a, b ,c w values
        from a x + b y + c = 0
        w is 1
        space_c is the middle point of the diamond space in our case 
        if the diamond space size is w then it is (w+1)/2, w is always odd number
    """
    a,b,c,w = line
    alpha = np.sign(a * b)
    beta = np.sign(b * c)
    thigma = np.sign(a * c)
    ax = alpha * a / (c + thigma * a)
    bx  = -alpha * c / (c + thigma * a)
    
    # here we are adding 1  to make it more than 1 and multiplying with 
    # space_c  to scale to diamond space size
    
    line_endp_ss_pp = [[int((ax+1) * space_c), int((bx+1) * space_c)]]
    line_endp_st_pp = [[int((b/ (c + beta * b) + 1) * space_c), int(space_c)]]
    line_endp_ts_pp = [[int(space_c), int((b/ (a + alpha * b) +1) *space_c)]]
    line_endp_tt_pp = [[int((-ax + 1) * space_c), int((-bx + 1) * space_c)]]
    
    res = np.r_[line_endp_ss_pp,line_endp_st_pp,line_endp_ts_pp,line_endp_tt_pp]
    return res

def rasterize_lines(lines, endpoints, space_c):
    """
    lines are n x 4 format where 4 values are [a,b,c,d]
    from ax + by + c =0, w is 1 which is used to convert to 
    homogeneous coordinate
    """
    print(lines)
    diamond_space = np.zeros((space_size,space_size))
    for i in range(len(lines)):
        print(i)
        weight = lines[i][3]
        for j in np.arange(0,3,1):
            end = endpoints[i]       
#            print('end = ', end)
            if(abs(end[j+1][1]-end[j][1]) > abs(end[j+1][0]-end[j][0])):
                diamond_space = lineV(end[j], end[j+1], diamond_space, weight)
            else:
                diamond_space = lineH(end[j], end[j+1], diamond_space, weight)
    
    diamond_space[end[3][1], end[3][0]] = diamond_space[end[3][1], end[3][0]] + 1
    
    return diamond_space

def lineH(endpoint0, endpoint1, space, weight):
    # change accumulator by longer axis
    slope = (endpoint1[1] - endpoint0[1]) / (endpoint1[0] - endpoint0[0])
    
    y_start = endpoint0[1] + 0.5
    y_iter = y_start
    step = -1
    if(endpoint0[0] < endpoint1[0]):
        step = 1
    slope = slope * step
#    c=1
    print('line H is started')
    for x in np.arange(endpoint0[0],endpoint1[0],step):
#         print(y_iter, int(x))
         space[int(y_iter), x]  = space[int(y_iter), x] + weight
         y_iter = y_start +  slope
#         c = c+ 1
    return space
    
def lineV(endpoint0, endpoint1, space, weight):
    slope = (endpoint1[0] - endpoint0[0]) / (endpoint1[1] - endpoint0[1])
    
    x_start = endpoint0[0] + 0.5
    x_iter = x_start
    
    step = -1
    if(endpoint0[1] < endpoint1[1]):
        step = 1
    slope = slope * step
#    print('line V is started')
    for y in np.arange(endpoint0[1], endpoint1[1], step):
#        print(y, int(x_iter), endpoint0[1], endpoint1[1])
        space[y, int(x_iter)] = space[y, int(x_iter)] + weight
        x_iter = x_iter + slope
    return space
def PC_point_to_CC(normalization, vanishpc, imgshape):
    u = vanishpc[:, 0]
    v = vanishpc[:, 1]
    
#    print(u, v)
   
#    print(np.sign(v) * v + np.sign(u) * u - 1)
    normvanishcc = np.c_[v, np.sign(v) * v + np.sign(u) * u - 1, u]
#    print(normvanishcc)
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
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value     
def find_maximum(space, R):
    r,c = np.where(np.max(space) == space)
#    print(r, c)
    S = np.pad(space, (R,R), pad_with, padder=0)
    
    
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

def normr(data):
    """
    normalize data by rows
    data is n x m shape
    n is the number of rows
    """
    newdata = [ row/np.sqrt(np.sum(np.square(row))) for row in data]
    return newdata

#def fitt_ellipse(points):
#    good_ellipse = 0
#    int

def get_lines():
    return np.array([[0.1, 0.2,0.03, 1],
                     [0.1, 0.2,0.33, 1],
                      [0.1, 0.2,0.43, 1]])

if __name__ == "__main__":
#    test line is x + 2 where full written form is
    fig, axs = plt.subplots(1, figsize=(10,10))
    lines  = get_lines()
    for i in range(len(lines)):
        a, b,c,_ = lines[i]
        b = -b
#        c = c #/ (512)
        pxs = [0, 511]
        pys = [pxs[0]*a + c, pxs[1] * a + c]
        axs.plot(pxs, pys,'k-')
    plt.show()
#    
    fig, axs = plt.subplots(1, figsize=(10,10))
    space_size = 321
#    
    resall = []
#    space = np.zeros((space_size, space_size,3), np.uint8)
    for i in range(len(lines)):
        res = convert_line_to_pc_line_endpoint(lines[i], (space_size-1.0)/2)
        axs.plot(res[0:2,0], res[0:2,1],'k-')
        axs.plot(res[1:3,0], res[1:3,1],'k-')
        axs.plot(res[2:4,0], res[2:4,1],'k-')
#        axs.plot([res[0,0], res[-1,0]],[res[0,1], res[-1,1]] ,'k-')
        resall.append(res)
        color = (255, 0, 0)
#        cv2.polylines(space, np.uint8(res.reshape((-1, 1, 2))), False, color, -1) 
#        space = space+ cv2.polylines(np.zeros((space_size, space_size,3), np.uint8), np.int32([res]), False, (1,0,0))
    space = rasterize_lines(np.array(lines), np.int32(resall), space_size)
    
    space1 = use_raster_space(lines.ravel(), [space_size,space_size],len(lines))
    space1 = np.reshape(space1 , (space_size, space_size)).T
#    diamond_space[np.int32(res[:,0]), np.int32(res[:,1])] = diamond_space[np.int32(res[:,0]), np.int32(res[:,1])] + 1
#    diamond_space[np.int32(res1[:,0]), np.int32(res1[:,1])] = diamond_space[np.int32(res1[:,0]), np.int32(res1[:,1])] + 1
#    plt.show()
#    space = cv2.rectangle(space, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), color, linewidth)
#    find_maximum(img,2)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(space1, interpolation='nearest')
    plt.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(space, interpolation='nearest')
    plt.tight_layout()
    plt.show()


#import cv2
##space[space>0] = 1
#img = space[:, :, 0]
#img[img>0] = 255
#img = cv2.resize(img, (512,512))
#cv2.imshow("img", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#   
#points = np.array([( 0 , 3),
#    ( 1 , 2),
#    ( 1 , 7),
#    ( 2 , 2),
#    ( 2 , 4),
#    ( 2 , 5),
#    ( 2 , 6),
#    ( 2 ,14),
#    ( 3 , 4),
#    ( 4 , 4),
#    ( 5 , 5),
#    ( 5 ,14),
#    ( 6 , 4),
#    ( 7 , 3),
#    ( 7 , 7),
#    ( 8 ,10),
#    ( 9 , 1),
#    ( 9 , 8),
#    ( 9 , 9),
#    (10,  1),
#    (10,  2),
#    (10 ,12),
#    (11 , 0),
#    (11 , 7),
#    (12 , 7),
#    (12 ,11),
#    (12 ,12),
#    (13 , 6),
#    (13 , 8),
#    (13 ,12),
#    (14 , 4),
#    (14 , 5),
#    (14 ,10),
#    (14 ,13)])
#    
#    
#    
#    
#plt.plot(points[:,0], points[:,1], '.')
#plt.show()
#
#x = points[:,0][:,np.newaxis]
#y = points[:,1][:,np.newaxis]
#D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
#S = np.dot(D.T,D)
#C = np.zeros([6,6])
#C[0,2] = C[2,0] = 2; C[1,1] = -1
#E, V =  np.linalg.eig(np.dot(np.linalg.inv(S), C))
#n = np.argmax(np.abs(E))
#a = V[:,n]

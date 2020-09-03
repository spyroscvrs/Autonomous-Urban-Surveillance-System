# -*- coding: utf-8 -*-
import numpy as np
import math
import cv2

def get_diamond_c_from_original_coords(x,y, a,b, width, height,padding =22, radius=22):
    """
    formula is 
    ax + by + c = 0
    x = (xorig + padding - wc) / norm
    y = (yorig + padding - hc) / norm
    where x, y is in diamond space
    """
    wc = (width + padding * 2 - 1) / 2
    hc = (height + padding * 2 - 1) / 2
    norm = max(wc, hc) - padding
    
    c = -(a * (x+padding -wc)) / norm - (b * (y +padding - hc)) / norm
    return c

def get_original_c_from_original_points(x,y, a,b):
#    print('inputted points are ', np.shape(o))
#    xt = x - np.average(x)
#    yt = y - np.average(y)
#    print(xt)
#    D =  np.c_[xt, yt]
    c = -b * y - a *x
    return c

def gety(x,a,b,c):
    """
    ax  + by + c = 0
    """
    y = (-a*x - c) / b
    return y

def get_coeffs(points):
    goal_inliers = len(points)
    max_iterations = 3
    m, b,new_points  = run_ransac(points, estimate, lambda x, y: is_inlier(x, y, 0.1), goal_inliers, max_iterations, 20)
    a,b,c = m
    c = -b * new_points[0][1] - a * new_points[0][0]
    return a,b,c
    
def get_focal_using_vps(vp1, vp2, width, height):
    # image size is considered always not odd
    U = np.array(vp1)[:2]
    V = np.array(vp2)[:2]
    #print("U,V is:")
    #print(U)
    #print(V)
    P = np.array([width/2-1, height/2-1])
    f = np.sqrt(np.dot(-(U-P), (V-P)))

    #f=math.sqrt(-np.dot(np.subtract(u,c),np.subtract(v,c))) #focal length

    return f

def get_third_VP(vp1, vp2, f, width, height):
    Ut = np.array([vp1[0],vp1[1], f])
    Vt = np.array([vp2[0],vp2[1], f])
    Pt = np.array([width/2-1, height/2-1, 0])
    W = np.cross((Ut-Pt), (Vt-Pt))
    return W

def get_rotation_matrix(vp1, vp2, vp3, f,width,height):
    """
    vp1 is the virst vanishing point directed Z axis
    vp2 is the second vanishing point directed X axis
    vp3 is the third vanishing point directed Y axis
#    Rotation matrix should be Z - Y - X (yaw, pitch, roll)
    
    z * vz = K*[r1 r2 r3 | t] zinf
    zinf = [0 0 1 0]'
    z*vz = K * r3
    r3 =  Kinv * vz/ (||Kinv * vz||)
    """
    K= np.array([
            [f, 1, width/2-1],
            [0, f, height/2-1],
            [0, 0, 1]
            ])
    Kinv = np.linalg.inv(K)
    r3 = np.dot(Kinv , vp1)/ np.sqrt(np.sum(np.square( np.dot(Kinv , vp1))))
    r1 =  np.dot(Kinv , vp2)/ np.sqrt(np.sum(np.square( np.dot(Kinv , vp2))))
    r2 = np.cross(r3, r1)
    R = []
    R.append(r1)
    R.append(r2)
    R.append(r3)
    R = np.array(R).T
    return R

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R, degree=True) :
    alpha = math.atan2(R[0,2], R[2,2]) # pan
    beta = math.asin(R[1,2]) # tilt angle
    if(degree):
        return np.array([alpha * 180/np.pi, beta* 180/np.pi])
    else:
        return np.array([alpha, beta])

def calculate_distance(vp1, h_bottom, f, h_cam, tilt_angle = 0):
    y = vp1[1]
    res = f * h_cam / ((h_bottom - y)* math.cos(tilt_angle))
    return res

def take_lane_towards_horizon(point1, point2, length=30):
    try:
        xdiff_1 = (point1[0] - point2[0])
        ydiff_1 = (point1[1] - point2[1])
        len_1 = math.sqrt(xdiff_1*xdiff_1 + ydiff_1*ydiff_1)
        line1 = [point1, (int(point1[0] + length* xdiff_1/len_1),int(point1[1] + length* ydiff_1/len_1))]
        return line1
    except:
        return []
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))
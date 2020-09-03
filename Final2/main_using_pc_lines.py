# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Wed Feb 19 08:51:41 2020
@author: ixtiyor
"""
from random import randrange
import numpy as np
import cv2
from pylsd import lsd
import os
import math
#from common import anorm2, draw_str
from pclines_point_alignment import params#, detect_vps_given_lines
# parameters to change
from pc_lines_diamond.diamond_vanish import  diamond_vanish_with_lines
from pc_lines_diamond.ransac.ransac import run_ransac, estimate, is_inlier
from pc_lines_diamond.utils import get_diamond_c_from_original_coords, gety,get_focal_using_vps,get_third_VP
from pc_lines_diamond.utils import get_rotation_matrix,rotationMatrixToEulerAngles,take_lane_towards_horizon
#from moving_edge_main import background_test, get_orientation_matrix_way
import random
import darknet_video
import pickle


class App:
    def __init__(self, video_src):
        # parameters to change
        self.track_len = 6
        self.alpha = 0.95 # for moving edge detection
        self.B = [] # for moving edge detection, edge bins
        self.detect_interval = 3
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.track_circle_color = (0, 255, 0)
        self.track_line_color = (0, 0, 255)
        self.lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.3))
        self.track_lines = []
        self.track_lines_coeffs = []
        self.moving_line_coeffs = []
        self.feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.5,
                       minDistance = 7,
                       blockSize = 7 )
#        self.track_len_threshold = self.track_len * 0.3
        self.space_old = []
        self.space_old_moving_edges = [] # for second vanishing point
        self.vp_1 = [] # first vanishing pooint
        self.vp_2 = [] # second vanishing pooint
        self.vp_3 = [] # third vanishing point
        
        self.vp_1_unchanged_life_count  = 0
        self.vp_2_started = False
        self.f = 0 # focal length
        self.R = None # Rotation matrix
        self.calibrate = True
        self.init()
    def init(self):
        if(not self.calibrate):
            with open('calibration.pickle','rb') as f:
                loaded_obj = pickle.load(f)
            self.vp_1 = loaded_obj['vp1']
            self.vp_2 = loaded_obj['vp2']
            self.vp_3 = loaded_obj['vp3']
            self.f = loaded_obj['f']
            self.R = loaded_obj['R']
    def save_calibration(self):
        calib_to_save = {"vp1":self.vp_1, "vp2":self.vp_2,"vp3":self.vp_3, "f":self.f, "R":self.R}
        with open('calibration.pickle','wb') as f:
            pickle.dump(calib_to_save, f)

    
    def get_len_tracks(self, i):
        x = self.tracks[i][0][0]
        y = self.tracks[i][0][1]
        len_= 0
        diff_xs = 0
        diff_ys = 0
        for j in np.arange(1, len(self.tracks[i]),1):
            xn = self.tracks[i][j][0]
            yn = self.tracks[i][j][1]
            diff_x = (xn - x)
            diff_y = (yn - y)
            len_ += math.sqrt(diff_x * diff_x + diff_y * diff_y)
            diff_xs += diff_x
            diff_ys += diff_y
        return len_, (diff_xs, diff_ys)
    def get_len_track(self, track):
        x = track[0][0]
        y = track[0][1]
        len_= 0
        diff_xs = 0
        diff_ys = 0
        for j in np.arange(1, len(track),1):
            xn = track[j][0]
            yn = track[j][1]
            diff_x = (xn-x)
            diff_y = (yn-y)
            len_ += math.sqrt(diff_x * diff_x + diff_y * diff_y)
            diff_xs += diff_x
            diff_ys += diff_y
        return len_

    def run(self):
        _ret, frame = self.cam.read()
#        frame = cv2.resize(frame, (512,512 ))
#        frame = rotate_bound(frame, 270)
        width = frame.shape[1]
        height = frame.shape[0]
        self.prms = params(width, height)
        vis = frame.copy()
        while True:
            _ret, frame = self.cam.read()
            print(_ret)
            if(_ret):
                frame = cv2.resize(frame, (width,height ))
                
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = darknet_video.detect_from_image(frame)
                detections_new, output_formatted = darknet_video.resize_detections(detections, height, width)
                if(self.calibrate):
    #                edges = np.zeros_like(frame)
                    moving_lines = []
                    for detection in detections_new:
                        x, y, w, h = detection[2][0],\
                                detection[2][1],\
                                detection[2][2],\
                                detection[2][3]
                        xmin, ymin, xmax, ymax = darknet_video.convertBack(
                            float(x), float(y), float(w), float(h))
                        padding = 1    
                        frame_crop = frame_gray[ymin-padding:ymax+padding, xmin-padding:xmax+padding]
                        if(frame_crop.shape[0] * frame_crop.shape[1] > 0):
                            moving_lines_t = lsd.lsd(np.array(frame_crop, np.float32))
                            moving_lines_t = moving_lines_t[:,0:4] 
                            for j in range(moving_lines_t.shape[0]):
                                if(int(moving_lines_t[j, 0] + xmin) < height and int(moving_lines_t[j, 1] + ymin) < width):
                                    pt1 = (int(moving_lines_t[j, 0] + xmin - padding), int(moving_lines_t[j, 1] + ymin - padding))
                                    pt2 = (int(moving_lines_t[j, 2] + xmin - padding), int(moving_lines_t[j, 3] + ymin - padding))
                                    length = np.sqrt(np.sum(np.square(np.array(pt1) - np.array(pt2))))
                                    if(length > 10 and length < 200):
                                        moving_lines.append([pt1[0],pt1[1],pt2[0],pt2[1]])
        #                                cv2.line(vis, pt1,pt2, (0, 255, 255), 1)
        #                    cv2.imshow('frame_crop', frame_crop)
                           
                    #vis = np.zeros(frame.shape, np.uint8)                                                                        
                    if len(self.tracks) > 0:
                        img0, img1 = self.prev_gray, frame_gray
                        p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
                        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
                        d = abs(p0-p0r).reshape(-1, 2).max(-1)
                        good = d < 1 # parameter to change
                        new_tracks = []  
                        new_lines = []
                        new_line_coeffs = []
                        for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                            if not good_flag:
                                continue
                            if self.frame_idx % self.detect_interval == 0:
                                vis = frame.copy()
    #                            cv2.circle(vis, (x, y), 1, self.track_circle_color, -1)
        #                    print("(x, y)", (x, y))
                            tr.append((x, y))
                            len_tr = self.get_len_track(tr)
                            if len(tr) > self.track_len:
                                del tr[0]
                            if(len_tr > 25):
                                m, b,new_points  = run_ransac(np.array(tr), estimate, lambda x, y: is_inlier(x, y, 0.1), 2, 3, len(tr))
                                if(m is None):
                                    continue
                                a,b,c = m
                                corig = -a * new_points[0][0] - b * new_points[0][1]
                                c = get_diamond_c_from_original_coords(new_points[0][0],new_points[0][1],a,b,self.prms.w,self.prms.h)
                                coeffs = [a,b,c,1]#[a/b,b/b,c/b,1/b]#[a,b,c,1]#[a/b,b/b,c/b,1/b]
                                new_line_coeffs.append(coeffs)
                                
                                xs = [-width, width]
                                ys= [gety(xs[0], a,b,corig),gety(xs[1], a,b,corig)]
        #                        print(ys)
        #                        try:
        #                            cv2.line( vis, (xs[0], int(ys[0])), (xs[1], int(ys[1])), (0,255,255), 2, 8 );        
        #                        except:pass
                            new_tracks.append(tr)
                                
        
                        self.tracks = new_tracks
                        self.track_lines = new_lines
                        self.track_lines_coeffs = new_line_coeffs
        #            if(self.vp_1_unchanged_life_count > 65):
        ##                frame_gray_2  = np.pad(frame_gray, (22,22), pad_with, padder=0)
        #                sobelx = cv2.Sobel(frame_gray,cv2.CV_64F,1,0,ksize=7)
        #                sobely = cv2.Sobel(frame_gray,cv2.CV_64F,0,1,ksize=7)
        #                magnitude = cv2.magnitude(sobelx, sobely) # computes sqrt(xi^2 + yi^2)
        #                phase = cv2.phase(sobelx,sobely,angleInDegrees=True) # computes angel between x and y
        #                
        #                H = get_orientation_matrix_way(magnitude, phase,0.3,8)
        #                
        #                if(len(self.B) == 0):
        #                    self.B = H
        #                else:
        #                    self.B = self.alpha * self.B + (1-self.alpha) * H
        #                
        #                H = background_test(self.B, H, t2 = 40000,t1=30000)
        #                H = np.sum(H,2)
        #                H = (H/np.max(H) * 255).astype('uint8')
                    if(self.vp_1_unchanged_life_count > 50 or self.vp_2_started):
        #                moving_lines= cv2.HoughLinesP(H,1,np.pi/180, 1, minLineLength=8,maxLineGap=5) 
                        self.vp_2_started = True
                        if(moving_lines is not None):
                            for line in moving_lines:
                                x1,y1,x2,y2 = np.int32(line)#[0]
    #                            cv2.line(edges,(x1,y1),(x2,y2),(0,255,255),2)
        #                    
                            new_moving_line_coeffs = []
                            for i in range(len(moving_lines)):
                                x1,y1,x2,y2 = moving_lines[i]#[0]
                                if((y1+y2)/2 <= 0.80 * self.prms.h):
                                    degree = 90
                                    if(len(self.vp_1) > 0):
            #                            degree = math.atan2(self.vp_1[1] - (y1+y2)/2,self.vp_1[0] - (x1+x2)/2)
            #                            degree = math.degrees(degree)
                                        lt_vp1 = [self.vp_1[0] - (x1+x2)/2, self.vp_1[1] - (y1+y2)/2]
                                        lt_cur_line = [x2-x1,y2-y1]
                                        rad = np.arccos(np.dot(lt_cur_line, lt_vp1)/ ((np.sqrt(np.dot(lt_cur_line,lt_cur_line))) * np.sqrt(np.dot(lt_vp1, lt_vp1))))
                                        degree = rad * 180 / np.pi
            #                        print('between angle is', degree)
                                    degree_threshold = 45
                                    if(degree >= degree_threshold and degree <= 180-degree_threshold):
                                        m, b,new_points  = run_ransac(np.array([[x1,y1],[x2,y2]]), estimate, lambda x, y: is_inlier(x, y, 0.1), 2, 3, 2)
                                    
                                        if(m is None):
                                            continue
                                        a,b,c = m
                                        corig = -a * new_points[0][0] - b * new_points[0][1]
                                        c = get_diamond_c_from_original_coords(new_points[0][0],new_points[0][1],a,b,self.prms.w,self.prms.h)
                                        coeffs = [a,b,c,1]#[a/b,b/b,c/b,1/b]#[a,b,c,1]#[a/b,b/b,c/b,1/b]
                                        new_moving_line_coeffs.append(coeffs)
                                        
                                        xs = [-width, width]
                                        ys= [gety(xs[0], a,b,corig),gety(xs[1], a,b,corig)]
                    #                        print(ys)
                                        try:
                                            cv2.line( vis, (xs[0], int(ys[0])), (xs[1], int(ys[1])), (0,255,255), 2, 8 );        
                                        except:pass
                        self.moving_line_coeffs = new_moving_line_coeffs
                        
                        if(len(self.moving_line_coeffs) > 0):
                            result_moving_edges = diamond_vanish_with_lines(np.array(self.moving_line_coeffs), self.prms.w,self.prms.h,0.4, 321,1,self.space_old_moving_edges)
                            self.space_old_moving_edges = result_moving_edges["Space"]
                            resvps_moving_edges = np.int32(result_moving_edges["CC_VanP"])
                            self.vp_2  = resvps_moving_edges[0]-1
                            for i in range(len(resvps_moving_edges)):
                                if(resvps_moving_edges[i][0] > 0 and resvps_moving_edges[i][1] > 0):
                                    cv2.circle(vis, (resvps_moving_edges[i][0]-1, resvps_moving_edges[i][1]-1), 5, self.track_circle_color, -1)
                        
                            
                            print("detected vp =====", self.vp_2)
                    if self.frame_idx % self.detect_interval == 0:
                        if(len(self.track_lines_coeffs) > 0):
                            result = diamond_vanish_with_lines(np.array(self.track_lines_coeffs), self.prms.w,self.prms.h,0.4, 321,1,self.space_old)
                            self.space_old = result["Space"]
                            resvps = np.int32(result["CC_VanP"])
                            if(len(self.vp_1) > 0):
                                
                                if(abs(np.sum((resvps[0] - 1 - self.vp_1))) <= 6): # if the point changes more than 6 pixel then, start again
                                    self.vp_1_unchanged_life_count  = self.vp_1_unchanged_life_count + 1
                                else:
                                    self.vp_1_unchanged_life_count  =0 #self.vp_1_unchanged_life_count - 1
                                    
                            self.vp_1  = resvps[0] - 1
                            print("detected vp =====", resvps, self.vp_1_unchanged_life_count)
                            for i in range(len(resvps)):
                                if(resvps[i][0] > 0 and resvps[i][1] > 0):
                                    cv2.circle(vis, (resvps[i][0]-1, resvps[i][1]-1), 5, self.track_circle_color, -1)
        #                self.vp_1_unchanged_life_count  = self.vp_1_unchanged_life_count + 1     
                        mask = np.zeros_like(frame_gray)
                        mask[:] = 255
                        for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                            cv2.circle(mask, (x, y), 2, 0, -1)
                    
                        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **self.feature_params)
                        if p is not None:
                            for x, y in np.float32(p).reshape(-1, 2):
                                self.tracks.append([(x, y)])
        
                        for i in range(len(self.tracks)):
                            cv2.polylines(vis, np.int32([self.tracks[i]]), False, self.track_line_color)
                    
                    if(len(self.vp_1) > 0 and len(self.vp_2) > 0):
                        self.f = get_focal_using_vps(self.vp_1, self.vp_2, self.prms.w, self.prms.h)
                        print('focal length = =  ', self.f)
                        self.vp_3 = get_third_VP(self.vp_1, self.vp_2, self.f, self.prms.w, self.prms.h)
                        if(len(self.vp_3) > 0):
                            # now using those vanishing points we can calculate Rotation matrix which is the normal vectors of the vanishing points(multiplied by K if exist)
                            self.R = get_rotation_matrix(self.vp_1, self.vp_2, self.vp_3,self.f, self.prms.w, self.prms.h)
                            # details of R is follows
                            # first row is for z axis, second for y , third is for x axis
                            print('euler angles are : ', rotationMatrixToEulerAngles(self.R))
                    for i in range(10):
                        random.seed(i)
                        line1 = []
                        line2 = []
                        line3 = []
                        point_r = (randrange(width), randrange(height))
                        if(len(self.vp_1) > 0):
                            line1 = take_lane_towards_horizon(point_r,self.vp_1, length=30)
                        if(len(self.vp_2) > 0):
                            line2 = take_lane_towards_horizon(point_r,self.vp_2, length=30)   
                        if(len(self.vp_3) > 0):
                            line3 = take_lane_towards_horizon(point_r,self.vp_3, length=30)   
                            
                        if(len(line1) > 0):
                            cv2.arrowedLine(vis, line1[1], line1[0]  , (0,0,255), 2, 8)
                        if(len(line2) > 0):
                            cv2.arrowedLine(vis, line2[1],  line2[0], (255,0,0), 2, 8)
                        if(len(line3) > 0):
                            cv2.arrowedLine(vis, line3[1],  line3[0], (0,255,0), 2, 8)                  
                    self.prev_gray = frame_gray
                    cv2.imshow('img', vis)
                else:
                    cv2.imshow('img', frame)
            self.frame_idx += 1
            ch = cv2.waitKey(1)
            if ch == 27 or not _ret:
                if(self.calibrate):
                    self.save_calibration()
                break
def main():
    videos_root = '/media/ixtiyor/New Volume/datasets/auto_callibration/videos/g1'
    videos = os.listdir(videos_root)
    try:
        #video_src="test_simple_1.mp4"
        video_src =  videos_root + "/" + videos[1]  # "GOPR2036_half.mp4" #
    except: 
        video_src = 0

    #print(video_src)
    App(video_src).run()
    cv2.destroyAllWindows()
    print('Done')
    
if __name__ == "__main__":
    main()

#    import gc
##    cv2.destroyAllWindows()
##    img = cv2.imread("/media/ixtiyor/New Volume/datasets/bdd/bdd100k_images/bdd100k/images/10k/test/af962335-00000000.jpg",-1)
#    img = np.zeros((400, 400,3), np.uint8);
#    for i in range(400):
#        if(i<159):
##            img[i  , i,:] = [255,255,255]
#            img[i  , 400-i-1,:] = [255,255,255]
#            img[400-i-1  ,i ,:] = [255,255,255]
##    img = cv2.resize(img, (512,512 ))
#    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    
#    space_size=  1024
##    old_space = []
#    w = img.shape[1]
#    h = img.shape[0]
#    normalization = 0.4
#    m_norm = max([w, h])
#    imgdraw = img.copy()
#    lines = lsd.lsd(np.array(frame_gray, np.float32))
#    lines = lines[:,0:4]
##        lines = np.array([[0,0, 50,50], [399,0,50,50]])
#    line_coeffs = []
#    max_iterations = 2
#    for j in range(len(lines)):
#        goal_inliers = 2
#        input_points = []
#        input_points.append([lines[j][0],lines[j][1]])
#        input_points.append([lines[j][2],lines[j][3]])
#        input_points = np.array(input_points)
#        m, b,new_points  = run_ransac(input_points, estimate, lambda x, y: is_inlier(x, y, 0.1), goal_inliers, max_iterations, 20)
#        
#        if(m is None):
#            continue
#        a,b,c = m
##            b, a,_,_ = fit_ellipse(input_points)
##            new_points = input_points
##            c = -a * new_points[0][0]  - b*new_points[0][1]
#        
#        if(len(np.shape(new_points))>1):
#            c = get_diamond_c_from_original_coords(new_points[0][0],new_points[0][1],a,b,w,h)
##                c = -b * new_points[0][1] - a * new_points[0][0]
#        else:
#            c = get_diamond_c_from_original_coords(new_points[0],new_points[1],a,b,w,h)
##                c = -b * new_points[1] - a * new_points[0]
#        coeffs = [a,b,c,1]#[a/b,b/b,c/b,1/b]#[a,b,c,1]#[a/b,b/b,c/b,1/b]
#        line_coeffs.append(coeffs)
#    line_coeffs = np.array(line_coeffs)
#    result = diamond_vanish_with_lines(line_coeffs,w,h,normalization, space_size,1,[])
##        result, line_coeffs, _ = diamond_vanish(img, normalization, space_size, 3, [])
#    for j in range(lines.shape[0]):
#        pt1 = (int(lines[j, 0]), int(lines[j, 1]))
#        pt2 = (int(lines[j, 2]), int(lines[j, 3]))
#        cv2.line(imgdraw, pt1,pt2, (0, 255, 255), 1)
#    
##    for j in range(len(line_coeffs)):
##        a,b,c,w = line_coeffs[j]
##        xs = [-1,1]
##        ys= [gety(xs[0], a,b,c),gety(xs[1], a,b,c)]
##            if(not (np.any(np.isnan(xs)) or np.any(np.isnan(ys)))):
##                x1 = int((xs[0]/normalization * (m_norm-1) + w + 1) / 2)
##                y1 = int((ys[0]/normalization * (m_norm-1) + h + 1) / 2)
##                x2 = int((xs[1]/normalization * (m_norm-1) + w + 1) / 2)
##                y2 = int((ys[1]/normalization * (m_norm-1) + h + 1) / 2)
##                cv2.line( imgdraw, (x1,y1), (x2, y2), (255,255,0), 2, 8 );
##            ys= [gety(xs[0], a,b,c),gety(xs[1], a,b,c)]
##            cv2.line( imgdraw, (xs[0], int(ys[0])-1), (xs[1], int(ys[1])-1), (0,255,255), 2, 8 );        
#    resvps = np.int32(abs(result["CC_VanP"]))
#    print("detected vp =====", resvps)
##    resvps = resvps[resvps[:,0] >0]
#    for j in range(len(resvps)):
#        cv2.circle(imgdraw, (resvps[j][0], resvps[j][1]), 5, [255,0,255], -1)
#    #    except:
#    ##        print('error')
#    ##    dimimg = result['Space'] / np.max(result['Space']) * 255
##        old_space = result['Space']
##        del result
#    gc.collect()
#    cv2.imshow("img", imgdraw)
#    ch = cv2.waitKey(0)
##    if(ch == 27): break
#    cv2.destroyAllWindows()
#    
#        
#    # save line_coeffs to mat file
#    url ='/home/ixtiyor/Downloads/2013-BMVC-Dubska-source (2)/'
#    import numpy, scipy.io
#    scipy.io.savemat(url+'line_cooeffs.mat', mdict={'arr': line_coeffs})

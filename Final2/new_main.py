from __future__ import print_function
from random import randrange
import numpy as np
import cv2
from pylsd import lsd
from skimage import io, feature, color, transform
import os
import math
from pclines_point_alignment import params#, detect_vps_given_lines
# parameters to change
from pc_lines_diamond.diamond_vanish import  diamond_vanish_with_lines
from pc_lines_diamond.ransac.ransac import run_ransac, estimate, is_inlier
from pc_lines_diamond.utils import get_diamond_c_from_original_coords, gety,get_focal_using_vps,get_third_VP
from pc_lines_diamond.utils import get_rotation_matrix,rotationMatrixToEulerAngles,take_lane_towards_horizon
from moving_edge_main import background_test, get_orientation_matrix_way
import random
import pickle
import backupVP
import imutils


class App:
    def __init__(self, video_src):
        # parameters to change
        self.track_len = 50
        self.alpha = 1 # for moving edge detection
        self.B = [] # for moving edge detection, edge bins
        self.detect_interval = 1
        self.tracks = []
        #self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.track_circle_color = (0, 255, 0)
        self.track_line_color = (0, 0, 255)
        self.lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.3))
        self.track_lines = []
        self.track_lines_coeffs = []
        self.moving_line_coeffs = []
        self.feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 7 )
#        self.track_len_threshold = self.track_len * 0.3
        self.space_old = []
        self.space_old_moving_edges = [] # for second vanishing point
        self.vp_1 = [] # first vanishing pooint
        self.vp_2 = [] # second vanishing pooint
        self.vp_3 = [] # third vanishing point

        self.checked=False #check if 1st vanishing point is estimated correctly

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

    
    def rotate_image(self,image, angle):
      image_center = tuple(np.array(image.shape[1::-1]) / 2)
      rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
      result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
      return result

    
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

    def run(self,cap):
        _ret, frame = cap.read()        

        frame = imutils.resize(frame, width=500)
        width = frame.shape[1]
        height = frame.shape[0]
        self.prms = params(width, height)

        vis = frame.copy()
        while True:
            _ret,frame=cap.read()

            #frame=self.rotate_image(frame,(90+180))
            #frame = imutils.resize(frame, height=600)
            #width = frame.shape[1]
            #height = frame.shape[0]
            #self.prms = params(width, height)

            print(_ret)
            if(_ret):
                frame = cv2.resize(frame,(width,height))
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if(self.calibrate):  
                    #print(self.tracks)                                               
                    if len(self.tracks) > 0:
                        img0, img1 = self.prev_gray, frame_gray
                        p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
                        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
                        d = abs(p0-p0r).reshape(-1, 2).max(-1)

                        good = d < 0.2 # parameter to change
                        
                        new_tracks = []  
                        new_lines = []
                        new_line_coeffs = []
                        for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                            if not good_flag:
                                continue
                            if self.frame_idx % self.detect_interval == 0:
                                vis = frame.copy()

                            tr.append((x, y))
                            #print(tr)
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

                    vp1_lifecount=19 #20 consecutive votes -> stable
                    new_moving_line_coeffs = []
                    if(self.vp_1_unchanged_life_count > vp1_lifecount or self.vp_2_started):

                        if self.checked==False:
                            print("Check")
                            #print()
                            vp1_b=backupVP.main(frame)

                            unit_vector_1 = vp1_b / np.linalg.norm(vp1_b)
                            unit_vector_2 = self.vp_1 / np.linalg.norm(self.vp_1)
                            dot_product = np.dot(unit_vector_1, unit_vector_2)
                            angle = np.arccos(dot_product)*180/np.pi #in degrees


                            #dist=np.linalg.norm(self.vp_1-vp1_b)
                            print(angle)
                            #print(vp1_b)
                            if angle>40:
                                print("Test not passed")
                                self.vp_1=vp1_b
                            self.checked=True


                        self.vp_2_started = True
                        sobelx = cv2.Sobel(frame_gray,cv2.CV_64F,1,0,ksize=7)
                        sobely = cv2.Sobel(frame_gray,cv2.CV_64F,0,1,ksize=7)
                        magnitude = cv2.magnitude(sobelx, sobely) # computes sqrt(xi^2 + yi^2)
                        phase = cv2.phase(sobelx,sobely,angleInDegrees=True) # computes angel between x and y
                        H = get_orientation_matrix_way(magnitude, phase,0.3,8)
                        self.B = H               
                        #H = background_test(self.B, H, t2 = 40000,t1=30000)
                        H = np.sum(H,2)
                        H = (H/np.max(H) * 255).astype('uint8')
                        #cv2.imshow('edges',H)


                    
                    #AUTH H IF TREXEI OTAN EXEI OLOKLHRWTHEI TO VP_1 estimation
                    #if(self.vp_1_unchanged_life_count > vp1_lifecount or self.vp_2_started):   

                        moving_lines= cv2.HoughLinesP(H,1,np.pi/180, 1, minLineLength=8,maxLineGap=5)
                        if(moving_lines is not None):
                            for line in moving_lines:
                                x1,y1,x2,y2 = np.int32(line)[0]
                                
                            for i in range(len(moving_lines)):
                                x1,y1,x2,y2 = moving_lines[i][0]
                                if((y1+y2)/2<=0.80*self.prms.h):
                                    #degree=90
                                    if(len(self.vp_1) > 0):
                                        lt_vp1 = [self.vp_1[0] - (x1+x2)/2, self.vp_1[1] - (y1+y2)/2]
                                        lt_cur_line = [x2-x1,y2-y1]
                                        rad = np.arccos(np.dot(lt_cur_line, lt_vp1)/ ((np.sqrt(np.dot(lt_cur_line,lt_cur_line))) * np.sqrt(np.dot(lt_vp1, lt_vp1))))
                                        degree = rad * 180 / np.pi
                                    degree_threshold = 45
                                    if(degree >= degree_threshold and degree <= 180-degree_threshold):
                                        m, b,new_points =run_ransac(np.array([[x1,y1],[x2,y2]]),estimate, lambda x, y: is_inlier(x, y, 0.1), 2, 3, 2)
                                        if(m is None):
                                            continue
                                        a,b,c = m
                                        corig = -a * new_points[0][0] - b * new_points[0][1]
                                        c = get_diamond_c_from_original_coords(new_points[0][0],new_points[0][1],a,b,self.prms.w,self.prms.h)
                                        coeffs = [a,b,c,1] #[a/b,b/b,c/b,1/b]#[a,b,c,1]#[a/b,b/b,c/b,1/b]
                                        new_moving_line_coeffs.append(coeffs)
                                        
                                        xs = [-width, width]
                                        ys= [gety(xs[0], a,b,corig),gety(xs[1], a,b,corig)]

                        self.moving_line_coeffs = new_moving_line_coeffs
                        if(len(self.moving_line_coeffs) > 0):
                            result_moving_edges = diamond_vanish_with_lines(np.array(self.moving_line_coeffs), self.prms.w,self.prms.h,0.4, 321,1,self.space_old_moving_edges)
                            self.space_old_moving_edges = result_moving_edges["Space"]
                            resvps_moving_edges = np.int32(result_moving_edges["CC_VanP"])
                            self.vp_2  = resvps_moving_edges[0]-1
                            for i in range(len(resvps_moving_edges)):
                                if(resvps_moving_edges[i][0] > 0 and resvps_moving_edges[i][1] > 0):
                                    cv2.circle(vis, (resvps_moving_edges[i][0]-1, resvps_moving_edges[i][1]-1), 5, self.track_circle_color, -1)      
                            print("detected vp2 =====", self.vp_2)


                    #AUTH H IF TREXEI GIA NA VRETHEI TO 1o Vanishing point
                    if self.frame_idx%self.detect_interval==0 and self.vp_2_started==False:
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
                            print("detected vp1 =====", resvps, self.vp_1_unchanged_life_count)
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

                    #AUTH H IF TREXEI AFOU EXOUN VRETHEI 1o kai 2o VP                    
                    if(len(self.vp_1) > 0 and len(self.vp_2) > 0):
                        self.f = get_focal_using_vps(self.vp_1, self.vp_2, self.prms.w, self.prms.h)
                        print('focal length =  ', self.f)
                        self.vp_3 = get_third_VP(self.vp_1, self.vp_2, self.f, self.prms.w, self.prms.h)
                        if(len(self.vp_3) > 0):
                            # now using those vanishing points we can calculate Rotation matrix which is the normal vectors of the vanishing points(multiplied by K if exist)
                            self.R = get_rotation_matrix(self.vp_1, self.vp_2, self.vp_3,self.f, self.prms.w, self.prms.h)

                    #GIA VISUALISATION OF VANISHING POINTS DIRECTION
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
                    print(self.f)
                    self.save_calibration()
                break
        return self.f,self.vp_1,self.vp_2

    
def main(video_src):
    cap=cv2.VideoCapture(video_src)
    focal,v1,v2=App(video_src).run(cap)
    cv2.destroyAllWindows()
    print('Done')
    return focal,v1,v2

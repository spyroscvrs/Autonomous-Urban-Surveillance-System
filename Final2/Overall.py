import argparse
import os
import time
from distutils.util import strtobool
from skimage import feature, color, transform, io
import numpy as np
import logging
import math
import cv2
from deep_sort import DeepSort
from detectron2_detection import Detectron2
from util import draw_bboxes2,draw_bboxes3
import new_main as ac
import matplotlib.pyplot as plt #ploting
import torch
from draw_mask import cropper
import statistics 
import imutils
    

class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))
        #use_cuda=False
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)
        self.vdo = cv2.VideoCapture()
        self.detectron2 = Detectron2()
        self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)

    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*"MPEG")
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self,f,u,v):
        #file=open("coords.txt", "w+") #open file
        file=open(self.args.coordfile, "w+") #open file
        fps = self.vdo.get(cv2.CAP_PROP_FPS) 
        scales=np.array([])
        while self.vdo.grab():
            #start = time.time()
            _,im = self.vdo.retrieve()
            #im=cv2.resize(im,(500,500))

            #img = cv2.imread(file , 0)
            im = imutils.resize(im, width=500)
            #cv2.imshow('image' , im)

            bbox_xcycwh, cls_conf, cls_ids, cls_masks = self.detectron2.detect(im)
            c=[int(im.shape[1]/2),int(im.shape[0]/2)]
            delta=np.array([1])
            #----------------------------FIND SCALE------------------------------- 
            if self.vdo.get(cv2.CAP_PROP_POS_FRAMES)<10:
                print("Scale Check")
                real_pixel_widths,car_or_not=cropper(im,cls_masks,cls_ids,cls_conf,bbox_xcycwh)
                #print(len(real_pixel_widths))
                for i in range(len(real_pixel_widths)): # exoume len(), zevgaria 
                    U=[u[0],u[1],f]
                    V=[v[0],v[1],f]
                    C=[c[0],c[1],f]
                    W=np.cross(np.subtract(U,C),np.subtract(V,C))
                    n=W/np.linalg.norm(W)
                    po=np.append(n,delta)
                    in_coord1=real_pixel_widths[i][0]
                    in_coord2=real_pixel_widths[i][1]
                    zeo=np.array([0]) 
                    p1=[in_coord1[0]-c[0],in_coord1[1]-c[1],f]
                    coord1=-(delta/np.dot(np.append(p1,zeo),po))*p1
                    p2=[in_coord2[0]-c[0],in_coord2[1]-c[1],f]
                    coord2=-(delta/np.dot(np.append(p2,zeo),po))*p2
                    distance=np.linalg.norm(np.subtract(coord1,coord2))
                    if car_or_not is True:
                        scales = np.append(scales,1.8/distance) #1.8 meters average width of car
                    else:
                        scales=np.append(scales,0.385/distance)  #38.5 centimeters meters average width of shoulders
                if len(scales)!=0:
                    scale=statistics.mean(scales) 
    #            print(scale)
            if len(scales)==0: #shmainei oti de mporeses na pareis oute ena scale
                print("Speed Estimation Not Available")
                scale=1
            #----------------------------END FIND SCALE-------------------------------


            if bbox_xcycwh is not None:
                bbox_xcycwh[:, 3:] *= 1
                outputs = self.deepsort.update(bbox_xcycwh, cls_conf, cls_ids, im)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                   # print(bbox_xyxy)    
                    #xy->panw,aristera kai katw,deksia
                    identities = outputs[:, -2]
                    classname = outputs[:, -1]
                    data_x=[]
                    data_y=[]
                    ids=[]
                    #print(classname)
                    #im = draw_bboxes2(im, bbox_xyxy, identities,classname)
                    for i in range(len(identities)):
                        if (classname[i]>=0) and (classname[i]<=9): 
                            yin=bbox_xyxy[i,3]
                            xin=(bbox_xyxy[i,2]-bbox_xyxy[i,0])+bbox_xyxy[i,0]
                            #pixel to road plane coordinates
                            delta=np.array([1]) 
                            U=[u[0],u[1],f]
                            V=[v[0],v[1],f]
                            C=[c[0],c[1],f]
                            W=np.cross(np.subtract(U,C),np.subtract(V,C))
                            n=W/np.linalg.norm(W)
                            po=np.append(n,delta)
                            in_coord1=[xin,yin]
                            p=[in_coord1[0]-c[0],in_coord1[1]-c[1],f]
                            zeo=np.array([0]) 
                            coord1=-(delta/np.dot(np.append(p,zeo),po))*p
                            print()
                            if u[0]>c[0]:
                                coord=[-scale*coord1[0],scale*coord1[1]]
                            else:
                                coord=[scale*coord1[0],-scale*coord1[1]]

                            data_x=np.append(data_x,np.array(coord[0]))
                            data_y=np.append(data_y,np.array(coord[1]))
                            ids=np.append(ids,identities[i])

                            file.write("%f,%f,%d,%d\n" % (coord[0],coord[1],identities[i],classname[i]))

                    im = draw_bboxes3(im, bbox_xyxy, identities,classname)
#                    plt.show()
            
            file.write("NewFrame\n")
            #print(frame_current)
            #end = time.time()
            #print("time: {}s, fps: {}".format(end - start, 1 / (end - start)))

            if self.args.display:
                cv2.imshow("test", im)
                cv2.waitKey(1)
            if self.args.save_path:
                self.output.write(im)
            # exit(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--coordfile", type=str, default="coords.txt")
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.3)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo.mp4")
    parser.add_argument("--use_cuda", type=str, default="False")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    f,u,v=ac.main(args.VIDEO_PATH)
    with Detector(args) as det:
        det.detect(f,u,v)


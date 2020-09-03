import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from skimage import io
import statistics 


def corners(np_array):
    ind = np.argwhere(np_array)
    res = []
    for f1, f2 in product([min,max], repeat=2):
        res.append(f1(ind[ind[:, 0] == f2(ind[:, 0])], key=lambda x:x[1]))
    return res

def cropper(img, mask_array,classes,scores,bbox_xcycwh):
    num_instances = mask_array.shape[0]
    mask_array = np.moveaxis(mask_array, 0, -1)
    mask_array_instance = []
    #img = imread(str(org_image_path))
    #print(bbox_xcycwh)
    car=True

    allcorners=[]


    output = np.zeros_like(img)
    score_car=[]
    for k in range(len(scores)):
        if classes[k]==2:
        	score_car.append(scores[k])

    if len(score_car)==0: #no cars detected
        car=False
        #then we check for pedestrians
        for k in range(len(scores)):
            if classes[k]==0:
                allcorners.append([bbox_xcycwh[k][0],bbox_xcycwh[k][0]+bbox_xcycwh[k][2]])
    else:
    	#print(score_car)
        mean_score=statistics.mean(score_car) 
        j=0
        for i in range(num_instances):
            if classes[i]!=2 or scores[i]<=mean_score:
                continue
            cornerss=corners(mask_array[:, :, i:(i+1)])
            a=[cornerss[0][1],cornerss[0][0]]
            b=[cornerss[2][1],cornerss[2][0]]
            c=[cornerss[1][1],cornerss[1][0]]
            d=[cornerss[3][1],cornerss[3][0]]
            mazi=[[a,b],[c,d]]
            norm1=np.linalg.norm(np.subtract(a,b))
            norm2=np.linalg.norm(np.subtract(c,d))

            if norm1>=bbox_xcycwh[i][2]/3:
                allcorners.append([a,b])
            if norm2>=bbox_xcycwh[i][2]/3:
                allcorners.append([c,d])

            mask_array_instance.append(mask_array[:, :, i:(i+1)])
            output = np.where(mask_array_instance[j] == True, 255, output)    
            j=j+1
        #io.imshow(output)
        #plt.show()		
    return allcorners,car

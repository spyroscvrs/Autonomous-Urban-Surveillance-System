import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]
cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_coords", type=str,default="coords.txt")
    parser.add_argument("--output_name", type=str)
    return parser.parse_args()


args=parse_args()
f=open(args.input_coords, "r")

coordinates=[]

while True: 
    line=f.readline() 
    currentline = line.split(",")
    if line=="NewFrame\n":
        continue
    if not line: 
        break
    coordinates.append([float(currentline[0]),float(currentline[1]),int(currentline[2])])
    
  
f.close()
ids=np.array(coordinates)[:,2]
uniqueid=np.unique(ids)

for i in range(len(uniqueid)):
    index=np.where(ids==uniqueid[i])
    index=index[0]
    coords=np.array(coordinates)[index.astype(int),0:2]
   # coords=np.array(coords)[:,0:2]
    #color=colors[int(uniqueid[i])%len(colors)]
    #plt.scatter(coords[:,0],coords[:,1],color)
    plt.scatter(coords[:,0],coords[:,1])


plt.xlabel("x")
plt.ylabel("y")
plt.title("Vehicle/Pedestrian Trajectories")
plt.savefig("Results/"+args.output_name+".png")



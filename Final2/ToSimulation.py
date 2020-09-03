import os
import math
import argparse
from lxml import etree
from lxml.builder import E
import numpy as np
import cv2


def create_etree():
    entities = etree.Element("Entities")
    story = etree.Element("Story", name="")
    for i in range(len(unique_differentIDs)):
        if Simornot[i]==1:
            if i==ind:
                vechname="Ego"
            else:
                vechname="vehicle"+str(i)
            entity = E.Object(
                E.CatalogReference(
                    catalogName="VehicleCatalog", entryName="Default car"
                ),
                name=vechname
            )
            entities.append(entity)

    for j in range(len(unique_differentIDs)):
        if Simornot[j]==1:
            if j==ind:
                vechname="Ego"
            else:
                vechname="vehicle"+str(j)
            traject = etree.Element("Trajectory",name=vechname,closed="false",domain="time")

            index=np.where(differentIDs==unique_differentIDs[j])
            instances=np.array(cf)[index]
            prevangle=0
            for k in range(len(instances)):
                if k!=len(instances)-1:
                    instance1=instances[k,1:3]
                    instance2=instances[k+1,1:3]
                    vector=[(instance2[0]-instance1[0])*10,(instance2[1]-instance1[1])*10]
                    vector2=[0,100]
                    unit_vector_1 = vector/np.linalg.norm(vector)
                    unit_vector_2 = vector2/np.linalg.norm(vector2)
                    dot_product = np.dot(unit_vector_1, unit_vector_2)
                    angle = np.arccos(dot_product)*180/np.pi
                    if np.isnan(angle): #means that object is not moving
                        angle=prevangle
                if k==len(instances)-1:
                    angle=prevangle
                prevangle=angle

                track=E.Vertex(
                    E.Position(E.World(x=str(instances[k,1]),y=str(instances[k,2]),z="0",h=str(angle),p="0",r="0")),
                    E.Shape(E.Polyline()),
                    reference=str((instances[k,0])/fps+3))
                traject.append(track)


        maneuver=E.Maneuver(
            E.Event(
                E.Action(
                    E.Private(
                        E.Routing(
                            E.FollowTrajectory(
                                traject,
                                E.Longitudinal(E.Timing(domain="absolute",scale="1.0",offset="0.0")),
                                E.Lateral(purpose="position")
                            ),
                        ),
                    ),name=vechname
                ),
                E.StartConditions(
                    E.ConditionGroup(
                        E.Condition(E.ByValue(E.SimulationTime(value="0.12",rule="greater_than")),name="SimulationStarted",delay="0",edge="rising")
                    )
                ),name=vechname,priority="overwrite"
            ),name=vechname
        )


        acts = E.Act(
            E.Sequence(
                E.Actors(
                    E.Entity(name=vechname)
                ),
                maneuver,
                name=vechname, numberOfExecutions="1"
            ),
            E.Conditions(
                E.Start(
                    E.ConditionGroup(
                        E.Condition(
                            E.ByValue(
                                E.SimulationTime(value="0", rule="greater_than"),
                            ),
                            name="SimulationStarted", delay="0", edge="rising"
                        ),
                    ),
                ))
            ,name=vechname
        )
        story.append(acts)

        #piase to kathe id kai tha perneis, thesh 0 -> frame number, kai to x,y


    openScenario = E.OpenSCENARIO(
        E.FileHeader(
            revMajor="0",
            revMinor="1",
            date="2020-07-31T12:36:42",
            description="",
            author="Imperial College"
        ),
        E.ParameterDeclaration(),
        E.Catalogs(
            E.VehicleCatalog(
                E.Directory(path="osc_catalog_vehicles.xosc"),
            ),
            E.DriverCatalog(
                E.Directory(path="UNDEFINED"),
            ),
            E.PedestrianCatalog(
                E.Directory(path="UNDEFINED"),
            ),
            E.PedestrianControllerCatalog(
                E.Directory(path="UNDEFINED"),
            ),
            E.MiscObjectCatalog(
                E.Directory(path="UNDEFINED"),
            ),
            E.EnvironmentCatalog(
                E.Directory(path="UNDEFINED"),
            ),
            E.ManeuverCatalog(
                E.Directory(path="UNDEFINED"),
            ),
            E.TrajectoryCatalog(
                E.Directory(path="UNDEFINED"),
            ),
            E.RouteCatalog(
                E.Directory(path="UNDEFINED"),
            ),
        ),
        E.RoadNetwork(
            E.Logics(filepath="UNDEFINED"),
            E.SceneGraph(filepath="UNDEFINED"),
        ),
        entities,
        E.Storyboard(
            E.Init(
                E.Actions(
                )
            ),
            story
        )
    )
    return openScenario



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str)
    parser.add_argument("--input_coords", type=str)
    parser.add_argument("--output_name", type=str)
    return parser.parse_args()



args=parse_args()


cap = cv2.VideoCapture(args.input_video);
fps=cap.get(cv2.CAP_PROP_FPS) #get fps of video, we need that for the Vertex
print(fps)
f=open(args.input_coords, "r")
count=1
cf=[]
while True: 
    line=f.readline() 
    currentline = line.split(",")
    if not line: 
        break
    if line=="NewFrame\n":   
        count=count+1 #auksanoume to count
    #if line!="NewFrame\n" and currentline[3]!=0:
    if line!="NewFrame\n":
        cf.append([count,float(currentline[0]),float(currentline[1]),int(currentline[2])])  
f.close() 

differentIDs=np.array(cf)[:,3]
unique_differentIDs=np.unique(differentIDs)


maximum=0
ind=0

Simornot=[0]*len(unique_differentIDs)

for i in range(len(unique_differentIDs)):
    index=np.where(differentIDs==unique_differentIDs[i])
    instances=np.array(cf)[index]
    if len(instances)>maximum:
        maximum=len(instances)
        ind=i
    if len(instances)>=int(maximum/5):
        Simornot[i]=1

openScenario = create_etree()
str = etree.tostring(openScenario, pretty_print=True, encoding='utf-8', xml_declaration=True)
newfile = "Results/"+args.output_name+".xosc"

with open(newfile, 'wb') as file:
    file.write(str)
file.close()
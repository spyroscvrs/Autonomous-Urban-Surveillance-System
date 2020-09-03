import numpy as np
import cv2


def suppression(out_pred,out_det,max_bbox_overlap):
    if len(out_pred)>0 and len(out_det)>0:
        together=np.concatenate((out_pred, out_det))
    if len(out_pred)>0 and len(out_det)==0:
        together=out_pred
    if len(out_pred)==0 and len(out_det)>0:
        together=out_det
    #out_pred.extend(out_det)

    #MEXRI EDW PREPEI NA EXEIS ENA BOXES POU NA EXEIS KRATHSEI TA DET KATA PRIORITY KAI META TA PRED
    
    #boxes = boxes.astype(np.float)
    pick = []
    x1 = together[:, 0]
    y1 = together[:, 1]
    x2 = together[:, 2]
    y2 = together[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))

    #otan teleiwnei to while exeis ta indices ta opoia mporeis na xrhsimopoihseis
    #final=[]
    final=together[pick]
    #for v in pick:
    #    final=np.append(final,together[v],axis=1)
    #final = [together[v] for v in pick]
    #print(final)
    return final

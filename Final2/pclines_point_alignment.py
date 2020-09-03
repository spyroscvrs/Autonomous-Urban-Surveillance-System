import math 
import functools
import numpy as np
from pylsd import lsd
import cv2 as cv
from mex_files.alignments_slow_python import use_alignments_slow
from mex_files.alignments_fast_python import use_alignments_fast
from gmm_mml import GmmMml
import matplotlib.pyplot as plt
import matplotlib
import scipy
import matplotlib.lines as mlines
from sklearn.cluster import AgglomerativeClustering

# there is a problem with sign function in python so this is a workaround
# https://stackoverflow.com/questions/1986152/why-doesnt-python-have-a-sign-function
#sign = functools.partial(math.copysign, 1)

class params:
    def __init__(self, w, h, focal_ratio=1.05455933619402):
        self.w = w
        self.h = h
        self.LENGTH_THRESHOLD = 30.71 # param to change
#        self.LENGTH_THRESHOLD = math.sqrt(self.w + self.h)/self.LENGTH_THRESHOLD 
        self.GMM_KS = [5,5,5] 
        self.REFINE_THRESHOLD = 0.375 # theta
        self.VARIATION_THRESHOLD = 0.15 # (\zeta)
        self.DUPLICATES_THRESHOLD = 0.1# (\delta)
        self.MAX_POINTS_ACCELERATION = 100 # use acceleration if number of points is larger than this
        self.MANHATTAN = True
        self.ppd = [2, 2] # principal point = [W,H]/prms.ppd ; values taken from YUD
        self.FOCAL_RATIO = focal_ratio # default for YUD
        
def detect_vps_given_lines(frame_gray, prms,lines, frame_to_draw=None, old_straight=[], old_twisted=[]):
    """
    given lines with shape [n x 4] computes vps
    """
    denoised_lanes = denoise_lanes(lines, prms)
    if(len(denoised_lanes) > 0):
        points_staright, points_twisted = convert_to_PClines(denoised_lanes, prms)
    else:
        points_staright , points_twisted = [], [], []
#        
#    if(len(old_straight) > 0 and len(old_twisted) >0):
#        print('before appending ', np.shape(points_staright), np.shape(points_twisted), np.shape(old_straight), np.shape(old_twisted))
#        if(len(points_staright)>0):
#            points_staright = np.r_[points_staright, old_straight]
#        else:
#            points_staright = old_straight
#        if(len(points_twisted) > 0):
#            points_twisted = np.r_[points_twisted, points_twisted]
#        print('after appending ', np.shape(points_staright), np.shape(points_twisted), np.shape(old_straight), np.shape(old_twisted))

#    print(np.shape(points_staright),np.shape(points_twisted))
    if(len(points_staright) == 0 or len(points_twisted) == 0): return []
    
    detections_straight, m1, b1 =  find_detections(points_staright, prms)
    detections_twisted, m2, b2 =  find_detections(points_twisted, prms)
    
    print('detections', np.shape(detections_straight),np.shape(detections_twisted))
    
    if(len(detections_straight) == 0 and len(detections_twisted) == 0):
        return [], [], []
        
    # gather initial vanishing point detections
    mvp_all, NFAs = read_detections_as_vps(detections_straight, m1, b1, detections_twisted, m2 ,b2, prms)
    
    print('\n\n\nbefore appending ', np.shape(points_staright), np.shape(points_twisted), np.shape(mvp_all))
    # refine detections, this returns 2 x ? array ? is the vps left after refining
    mvp_all = refine_detections(mvp_all, lines, prms)
    
    print('after appending ', np.shape(points_staright), np.shape(points_twisted), np.shape(mvp_all))
    
    mvp_all,NFAs = remove_dublicates(mvp_all, NFAs, prms)
    for i in range(len(mvp_all[0])):
        p1 = np.int32(mvp_all[:, i])
        cv.circle(frame_to_draw,tuple(p1),5,(0,255,0),3)
    
    return mvp_all.T, points_staright, points_twisted  # return as n x2 shape where n is the number of vps

def detect_vps(frame_gray, prms, frame_to_draw=None, points_staright_old=[], points_twisted_old=[]):
    lines = lsd.lsd(np.array(frame_gray, np.float32))
    lines = lines[:,0:4]
    for i in range(lines.shape[0]):
        pt1 = (int(lines[i, 0]), int(lines[i, 1]))
        pt2 = (int(lines[i, 2]), int(lines[i, 3]))
        if(not frame_to_draw is None):
            cv.line(frame_to_draw, pt1,pt2, (0, 0, 255), 1)
    denoised_lanes = denoise_lanes(lines, prms)
    
    if(len(denoised_lanes) > 0):
        points_staright, points_twisted = convert_to_PClines(denoised_lanes, prms)
    else:
        points_staright , points_twisted = [], []
     
    
    detections_straight, m1, b1 =  find_detections(points_staright, prms)
    detections_twisted, m2, b2 =  find_detections(points_twisted, prms)
    
    
    if(len(detections_straight) == 0 and len(detections_twisted) == 0):
        return []
        
    # gather initial vanishing point detections
    mvp_all, NFAs = read_detections_as_vps(detections_straight, m1, b1, detections_twisted, m2 ,b2, prms)
    
    # refine detections, this returns 2 x ? array ? is the vps left after refining
    mvp_all = refine_detections(mvp_all, lines, prms)
    mvp_all,NFAs = remove_dublicates(mvp_all, NFAs, prms)
    for i in range(len(mvp_all[0])):
        p1 = np.int32(mvp_all[:, i])
        cv.circle(frame_to_draw,tuple(p1),5,(0,255,0),3)
    
    return mvp_all.T# return as n x2 shape where n is the number of vps

print("please finish manhattan world")
# TO DO the manhattan world
#def compute_horizon_line_manhattan(mvp_all, NFAs, lines_lsd, prms):
#    # computes horizontal line from vps using the NFA values to apply
#    # orthogonality constraintes
#    # this is conversion of matlab code written by Jose Lezama
#    # Converting author : Majidov  Ikhtiyor
#    H = prms.w
#    W = prms.h
#    
#    # york urban parameters (given)
#    # focal = 6.05317058975369
#    # pixelSize = 0.00896875
#    # pp = [307.551305282635, 251.454244960136]
#    pp = np.array([W, H])/prms.ppd
#    FOCAL_RATIO = prms.FOCAL_RATIO # 
#    my_vps = image_to_gaussian_sphere(mvp_all, W, H, FOCAL_RATIO, pp)
#    
def remove_dublicates(vps, NFAs, prms):
    # vps is 2 x n array n is the number of points
    THRESHOLD = prms.DUPLICATES_THRESHOLD
    #agglomerative clustering using single link
    if(len(vps[0]) == 1):
        return vps, NFAs
    clus,n_clus = aggclus(vps.T, THRESHOLD)
    
    final_vps = []
    final_NFAs = []
    for i in range(n_clus):
        args = np.where(clus == i)[0]
        if(len(args) == 1):
            args = args[0]
            final_vps.append(vps[:, args])
            final_NFAs.append(NFAs[args])
        else:
            ind = np.argmax(NFAs[args])
            final_vps.append(vps[:, ind])
            final_NFAs.append(NFAs[ind])  
    return np.array(final_vps).T, np.array(final_NFAs)

def aggclus(X, THRESHOLD):
    """
    agglomerative clustering using single link
    X is the n x m vector where n is the number of samples
    uses euclidian distance
    """
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=THRESHOLD).fit(X)
    return clustering.labels_,clustering.n_clusters_

def refine_detections(mvp_all, lines_lsd, prms):
    """
    refines detections using lines from LSD
    """
    def refine_vp_iteration(lines, vp, THRESHOLD, H , W):
        """
            finds intersection of each line in cluster  with vanishing point segments
        """

        # lines must have shape L x 4 [[x1,y1,x2,y2] ... ]
        # after that mp should become L x 2 [[xm1,ym1] ... ]
        mp = np.c_[lines[:,0] + lines[:,2], lines[:,1] + lines[:, 3]]/2
        L = len(lines) # this probably the number of lines
        O = np.ones(L) # this give one dimensional array
        Z = np.zeros(L)
        vpmat = my_repmat2(vp, [L, 1]) # this should result L x 2 matrix
        VP = my_cross(np.c_[mp, O], np.c_[vpmat, O])
        
        # this should return L x 3 matrix, full of ks (from i,j,k)  
        VP3 = my_repmat(VP[:, 2], [1, 3])
        VP  = VP/VP3 # we are making third axis equal to one
        # VP shape must be N x 3
        # mp_vp will become like:
        # [[x1 x2]
        #  [y1 y2]]
        
        a = VP[:,0]
        b = VP[:,1]
        
        angle = abs(np.arctan(-a/b)-np.arctan( (lines[:,3]-lines[:,1])/(lines[:,2]-lines[:,1]) ))
        
        angle = np.array([min(k, np.pi - k) for k in angle])
        z2  = np.where(angle<2 *np.pi /180)[0]
        # obtain a refined VP estimte from sub cluster z2
        lengths = np.sum(np.square(np.c_[lines[:,0],lines[:,1]] - np.c_[lines[:,2], lines[:,3]]), -1)
        
        weights = lengths/ np.max(lengths)
        lis = line_to_homogeneous(lines)
    
        Q = np.zeros([3,3])
        Is = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
                ])
        l2 = lis[z2]
        w2 = weights[z2]
        w2 = my_repmat(w2, [1, 3]).T
        
        # matlab dot function equals to np.diagonal(np.dot(a.T, a))
        # matlab * equals to np.dot(a.T, a.T) (please check this one)
        bt = np.dot(Is,l2.T) # in matlab it was bt = (Is'*l2)
        b = np.diagonal(np.dot(bt.T, l2.T)) # 
        
        b = my_repmat(b, [1, 3]).T
        w2t = w2 * (l2.T/ b)
        Q = np.dot(w2t, l2)
        p = np.array([0, 0, 1])
        A = np.c_[2* Q, -p]
        vp = null_mine(A)
        vp = vp[0:2,0]/vp[2,0]
        return vp
    
    def refine_vp(lines, vp, prms):
        """
        given a cluster of line segments aand two segments indicated by p1 and 
        p2 obtain the main vanishing point determined by the segments
        """
        THRESHOLD = prms.REFINE_THRESHOLD
        H = prms.h
        W = prms.w
        vp_orig = vp
        vp = refine_vp_iteration(lines, vp, THRESHOLD, H,W)
        
        variation = norm(vp -vp_orig)/norm(vp_orig)
        if(variation > prms.VARIATION_THRESHOLD or np.any(np.isnan(vp) | np.isinf(vp))):
            vp = vp_orig
        return vp
    D = 0
    if(len(mvp_all) > 0):
        D = len(mvp_all[0])
    mvp_refined = np.zeros((D, 2))
    for i in range(D):
        vp = mvp_all[:, i].T
        vp = refine_vp(lines_lsd, vp, prms)
        mvp_refined[i, :] = vp
    mvp_all = mvp_refined.T
    return mvp_all
def norm(vect1):
    """
    input is one dimensional array, or vector
    applies p norm to vector:
    https://www.mathworks.com/matlabcentral/answers/117178-what-does-the-function-norm-do#answer_125313
    """
    return np.sqrt(np.sum(np.square(vect1)))
    
def null_mine(a):
    return scipy.linalg.null_space(a)

def line_to_homogeneous(l):
    """
    converts line in [x1,y1,x2,y2] format to homogeneous coordinates
    """
    x1 = l[:, 0]
    y1 = l[:, 1]
    x2 = l[:, 2]
    y2 = l[:, 3]
    
    dx = x2- x1
    dy = y2 - y1
    a = -dy
    b = dx
    c = -a * x1 - b * y1
    L = np.c_[a, b, c]
    return L

def my_repmat2(A, siz):
    """
    a = array([[1, 2, 3, 4],
       [5, 6, 7, 8]])    
    my_repmat2(a[:,0],[3,1])
    array([[1, 5],
           [1, 5],
           [1, 5]])
    """
    return np.tile(A, [siz[0],1])

def my_repmat(A, siz):
    """
    a = array([[1, 2, 3, 4],
       [5, 6, 7, 8]])
    my_repmat(a[:,0],[1,3])
    array([[1, 1, 1],
           [5, 5, 5]])
    """
    return np.tile(A, [siz[1],1]).T

def my_cross(a, b):
    return np.cross(a,b)
def read_detections_as_vps(detections_straight, m1, b1, detections_twisted, m2, b2, prms):
    # converts alignment detections to vanishing points in the image 
    # returns mvp_all, a list of vanishing points and NFAs, their corresponding
    # -log10(true NFA)
    
    # this is a conversion of matlab code written by Jose Lezama <jlezama@gmail.com>
    # converting author : Majidov Ikhtiyor
    
    if(len(detections_straight)==0 and len(detections_twisted) == 0):
        return [], []
    H = prms.h
    W = prms.w
    D1 = len(detections_straight)
    D2 = len(detections_twisted)
    if(D1 > 0):
        NFAs1 = detections_straight[:, 5]
    else:
        NFAs1 = []
    
    if(D2 > 0):
        NFAs2 = detections_twisted[:, 5]
    else:
        NFAs2 = []
    
    # get vps in image coordinates (do PClines inverse)
    d = 1
    x1 = b1
    y1 = d * m1 + b1
    
    x2 = b2
    y2 = -1* (-d * m2 + b2)
    
    x1 = x1 * W
    y1 = y1 * H
    
    x2 = x2 * W
    y2 = y2 * H
    
    vps1 = np.c_[x1, y1] # [[x11, y11], [x12, y12] ...]
    vps2 = np.c_[x2, y2]
#    here,after below code our matrix become like 
#    [
#    [x11 x12 .. x21 x22 ...]
#    [y11 y12 .. y21 y22 ...]
#    ]
    mvp_all = np.c_[vps1.T, vps2.T] 
#    here  our matrix will become like
#    [nfas11 nfas12 .... nfas21 nfas22 ... ]
    NFAs = np.r_[NFAs1, NFAs2]
    
    # remove nan (infinity vp)
    # we will take second axis because we want to delete  by column
#    print('before deleting nans', np.shape(NFAs), np.shape(mvp_all))
    args = np.where(np.isnan(mvp_all) | np.isinf(mvp_all))[1]
    mvp_all = np.delete(mvp_all,args, 1)
    NFAs = np.delete(NFAs, args, 0)
#    print('before deleting nans', np.shape(NFAs), np.shape(mvp_all))
    return mvp_all, NFAs

def get_ellipse_endpoints(mu, cov, level=2, draw=False):
        uu, ei, vv = np.linalg.svd(cov)
        a = np.sqrt(ei[0] * level * level)
        b = np.sqrt(ei[1] * level * level)
        theta = np.array([0, np.pi])
        xx = np.dot(a , np.cos(theta))
        yy = np.dot(b , np.sin(theta))
        cord = np.c_[xx.T, yy.T].T
        cord = np.dot(uu , cord)
        x0 = cord[0][0]  + mu[0]
        x1 = cord[0][1] + mu[0]
        y0 = cord[1][0] + mu[1]
        y1 = cord[1][1]  + mu[1]
        
        thetas = np.arange(0, 2*np.pi, 0.01)
        xxs = a * np.cos(thetas)
        yys = b * np.sin(thetas)
        
        cords = np.c_[xxs.T, yys.T].T
        cords = np.dot(uu , cords)
        if(draw):
            plt.plot(cords[0]+ mu[0] , cords[1] +  mu[1])
        return np.array([x0, y0, x1, y1])
        
def run_mixtures(points, Ks=[20, 40, 60], filename="candidate_pairs.txt", draw=False):
    # Runs Figueiredo et al. GMM algorithm with different parameters (number of
    # Gaussians). The endpoints of the ellipses found are saved as candidates
    # for the alignment detector.
    # Parameters:
    # - points: list of 2D points
    # - Ks: number of Gaussians to try (example: [20 40 60])
    # - file_path: path where to save a text file with the obtained pairs of
    # points
    points = np.round(points) 
    points = np.vstack({tuple(row) for row in points}) # only getting unique rows
    colors = ['r']
    all_bestpairs = []
    threshold = 1e-4
    k = 0
    
    while(k < len(Ks)):
        K = Ks[k]
        unsupervised=GmmMml(max(2,K-7),K,0,threshold, 0,  max_iters=2)        
        new_labels = None
        try:
            if(draw):
                new_labels = unsupervised.fit_transform(points)
                new_labels = np.argmax(new_labels,-1)
            else:
                unsupervised.fit(points)
            k = k +1
        except:
            threshold*=10
            continue
        if(draw):
#            fig, ax = plt.subplots()
            plt.scatter(points[:,0],points[:,1], alpha=0.3,s=10, color = colors*len(points), marker=',')
        
        best_pairs = np.zeros((unsupervised.bestk, 4))
        for comp in range(unsupervised.bestk):
            best_pair = get_ellipse_endpoints(unsupervised.bestmu[comp],unsupervised.bestcov[:,:, comp], 2, draw=draw)
            best_pairs[comp, :] = best_pair
        if(len(all_bestpairs)==0):
            all_bestpairs = best_pairs
        else:
            all_bestpairs = np.r_[best_pairs]
#        if(draw):
#            plt.show()
    return np.array(all_bestpairs)

def find_detections(points, prms, draw=False):
    # now skip to slow version
    M = np.max(points)
    m = np.min(points)
    
    
    points = (points - m)/(M - m) * 512 # this version of the alignment detector expects a 512 x 512 domain
#    points = np.int32(points)
    points = np.unique(np.round(points,2), axis=0)
    
    print(points)
    N = len(points)
    
    print(len(points))
    
    if(N >= prms.MAX_POINTS_ACCELERATION):
        print('accelerated detection started')
        candidates = run_mixtures(points, prms.GMM_KS,'',draw=draw)
        n_candidates = len(candidates) * 2
        # make cadidates arrey ready for usage in c
        # each point should locate each other like x0,x1 ... , y0, y1 ....
        candidates = candidates.ravel()
        candidates = np.r_[np.take(candidates, np.arange(0,len(candidates),2)),np.take(candidates, np.arange(1,len(candidates),2))]
        
        # make points ready for usage
        # coversion from [[x,y],[x1,y1]] to [x,x1 .. y,y1]
        points = list(points.transpose().ravel())
        detections,n_out = use_alignments_fast(points,2,N, candidates, n_candidates)
    else:
        print('slow detection started')
        points = list(points.transpose().ravel())
        detections,n_out = use_alignments_slow(points, 2, N)
        
    print('detection finished')    
    if(len(detections) > 0):
        detections = np.array(np.array_split(detections, n_out))
    if(not len(detections) == 0):
        dets = detections[:, 0:4]
        dets = dets/512 * (M -m)+ m
        detections[:, 0:4] = dets
        detections[:, 4] = detections[:, 4] / 512 * (M - m)
        x1= dets[:,0]
        y1= dets[:,1]
        x2= dets[:,2]
        y2= dets[:,3]
        
        dy = y2 - y1
        dx = x2 - x1
        m_out = dy/dx
        b_out = (y1 * x2 - y2 * x1)/dx
        if(draw):
            for i in range(len(m_out)):
                x = np.arange(-1, 1, 0.1)
                y= m_out[i] * x + b_out[i] 
                x = (x - m)/(M - m) * 512
                y = (y - m)/(M - m) * 512
                plt.plot(x, y)
            plt.show()
        return detections, m_out, b_out 
    else:
        return [], [], []
    
#    return detections
def denoise_lanes(lines, prms):
    """
    lanes: array with shape n x 4
    [x1, y1 , x2, y2]
    """
    new_lines = np.array(lines)
    
    if(len(new_lines) > 0):
        lengths =np.sum(np.sqrt([np.power(new_lines[:, 2] - new_lines[:,0], 2),np.power(new_lines[:,3] - new_lines[:,1], 2)]), 0)
    else:
        return []

    # now denoise according to length_threshold 
    lengths  = np.ravel(lengths)
    matched_args = np.where(lengths > prms.LENGTH_THRESHOLD)
    # un_matched_args = np.where(lengths <= prms.LENGTH_THRESHOLD)
    lines_large = new_lines[matched_args]
    # lines_short = new_lines[un_matched_args]
    return lines_large

def convert_to_PClines(lines, prms):
    """
        lines in the shape of n x 4 or n x 2
        where 
        4 values indicates:
        x, y, x1, y1 : defining coordinates of the line
    """
    H = prms.h 
    W = prms.w 
    L = len(lines)
    points_straight = PCLines_straight_all(lines/ np.tile([W, H, W, H],[L, 1]))
    points_twisted = PCLines_twisted_all(lines/ np.tile([W, H, W, H],[L, 1]))
    args_4_del_strait = np.where((points_straight[:,0]>2) | (points_straight[:,1]>2)\
                               | (points_straight[:,0]<-1) | (points_straight[:,1]<-1) \
                               | (np.isnan(points_straight[:,0])) | (np.isnan(points_straight[:,1]) ))[0]

    args_4_del_twisted = np.where((points_twisted[:,0]>1) | (points_twisted[:,1]>1)| \
                                   (points_twisted[:,0]<-2) | (points_twisted[:,1]<-2) | \
                                   (np.isnan(points_twisted[:,0])) | (np.isnan(points_twisted[:,1])))[0]
    
    if(len(args_4_del_strait) > 0):
        points_straight = np.delete(points_straight, args_4_del_strait, axis=0) 
    if(len(args_4_del_twisted) > 0):
        points_twisted = np.delete(points_twisted, args_4_del_twisted, axis=0)         
    return [points_straight, points_twisted]

def PCLines_straight_all(l):
    """
        transforms line as [x1,y1, x2, y2] or a point as [x,y] with PCLines straight
        transform coordinates should be normalized
    """ 
    d = 1.0 # arbitrary distance between vertical axes x and y
    L = len(l[0])
    if(L == 4):
        x1 = l[:, 0]
        y1 = l[:, 1]
        x2 = l[:, 2]
        y2 = l[:, 3]
        dy = y2 - y1
        dx = x2 - x1
        m = dy / dx
        b = (y1 * x2 - y2 * x1)/ dx 
        PCline = np.tile(d, [len(b),1])
        PCline = np.append(PCline, np.reshape(b, (len(b), 1)), 1)
        PCline = np.append(PCline, np.reshape(1-m, [len(m),1]), 1) # homogeneous coordinates
        
        u = PCline[:, 0] / PCline[:,2]
        v = PCline[:, 1] / PCline[:,2]
        res =np.append(np.reshape(u, (len(u), 1)), np.reshape(v, (len(v), 1)), 1)
        return res
    elif(L == 2):
        """it is a point"""
        x = l[:, 0]
        y = l[:, 1]
        b = x 
        m = (y - x) /d
        u = m 
        v = b 
        res =np.append(np.reshape(u, (len(u), 1)), np.reshape(v, (len(v), 1)), 1)
        return res

def PCLines_twisted_all(l):
    """
        transforms line as [x1,y1, x2, y2] or a point as [x,y] with PCLines twisted
        transform coordinates should be normalized
    """ 

    d = 1 # arbitrary distance between vertical axes x and y
    L = len(l[0])
    if(L == 4):
        x1 = l[:, 0]
        y1 = l[:, 1]
        x2 = l[:, 2]
        y2 = l[:, 3]
        dy = y2 - y1
        dx = x2 - x1
        m = dy / dx
        b = (y1 * x2 - y2 * x1)/ dx 
        PCline = np.tile(-d, [len(b),1])
        PCline = np.append(PCline, -1 *np.reshape(b, (len(b), 1)), 1)
        PCline = np.append(PCline, np.reshape(1+m, [len(m),1]), 1) # homogeneous coordinates
        u = PCline[:, 0] / PCline[:,2]
        v = PCline[:, 1] / PCline[:,2]
        res =np.append(np.reshape(u, (len(u), 1)), np.reshape(v, (len(v), 1)), 1)
        return res
    elif(L == 2):
        """it is a point"""
        x = l[:, 0]
        y = l[:, 1]
        b = x 
        m = (y + x) /d
        u = m 
        v = b 
        res =np.append(np.reshape(u, (len(u), 1)), np.reshape(v, (len(v), 1)), 1)
        return res

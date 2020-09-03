import sys, platform
import ctypes, ctypes.util
from ctypes import POINTER, c_double, c_int,byref
import numpy as np
#import sys
#sys.path.append("..")
#from pclines import run_mixtures 
try:
    mylib = ctypes.CDLL("./mex_files/alignments_fast.so")
except OSError:
    
    print("Unable to load the system C library", OSError.strerror)
    sys.exit()

#free_mem = mylib.free_mem
#free_mem.argtypes = [POINTER(c_double)]
#free_mem.restype = None

alignments_fast_fync = mylib.mexFunction_alignment_fast
# NCP_in number of candidate points
#------------------------------[input_points,       Xin,   N_in, input_candidate_points, NCP_in ,   output_points            , x_out,         , n_out]
alignments_fast_fync.argtypes= [POINTER(c_double), c_int, c_int, POINTER(c_double),      c_int,    POINTER(POINTER(c_double)),POINTER(c_int),POINTER(c_int)]
# x in is the number of features for each points(columns) , x_row is number of points
def use_alignments_fast(input_points, x_in, n_in, input_candidate_points, ncp_in):
    input_points = np.array(input_points, float)
    input_points_c = input_points.ctypes.data_as(POINTER(c_double))
    x_in_c = c_int(x_in)
    n_in_c = c_int(n_in)
    input_candidate_points_c = input_candidate_points.ctypes.data_as(POINTER(c_double))
    ncp_in_c = c_int(ncp_in)
    out_d_c = POINTER(c_double)()
    x_out = c_int()
    n_out = c_int()
    alignments_fast_fync(input_points_c, x_in_c, n_in_c,input_candidate_points_c,ncp_in_c, byref(out_d_c), byref(x_out), byref(n_out))
    return [np.round(out_d_c[i],100) for i in range(x_out.value * n_out.value)], n_out.value

def get_a():
    a = np.array([[-2.17075071e-02,  5.53605403e-01],
       [ 5.86835162e-01,  9.54891190e-01],
       [-3.69022878e-02,  5.58551667e-01],
       [-1.00750462e-02,  5.56522451e-01],
       [-3.23013830e-02,  5.53011886e-01],
       [ 9.49686331e-02,  5.12309514e-01],
       [ 7.48423455e-02,  5.21522400e-01],
       [ 6.71935559e-02,  5.25229549e-01],
       [-2.38076599e-02,  5.61383663e-01],
       [-3.07084685e-01,  6.11733765e-01],
       [ 3.03332490e-01,  7.06071451e-02],
       [ 7.03232414e-02,  5.21429689e-01],
       [ 1.74443239e+00,  7.69091860e-01],
       [ 1.49830349e-01,  4.87209022e-01],
       [ 8.12060775e-02,  5.15678543e-01],
       [ 1.46926387e-01,  4.94615323e-01],
       [ 6.40841924e-02,  5.24741796e-01],
       [ 6.23022925e-02,  5.25247463e-01],
       [-4.69722944e-02,  5.54918675e-01],
       [ 9.59344486e-02,  5.12552741e-01],
       [ 8.39044666e-02,  5.16302488e-01],
       [ 8.25591394e-02,  5.27938026e-01],
       [ 1.47523689e-01,  4.94405883e-01],
       [ 7.21774273e-02,  5.12219086e-01],
       [ 1.69887914e+00,  7.41472590e-01],
       [-6.14005672e-01,  8.45011895e-01],
       [ 1.47034847e-01,  4.93846753e-01],
       [-4.13807349e-01,  7.53615232e-01],
       [ 1.53396974e-01,  4.88717983e-01],
       [ 1.54585999e-01,  4.89377301e-01],
       [-5.29965378e-01,  7.51749351e-01],
       [-4.19715190e-01,  7.61789517e-01],
       [ 8.64427556e-02,  5.14868006e-01],
       [-4.98134767e-01,  8.46914242e-01],
       [ 1.51788117e+00,  1.16065934e-02],
       [-5.52080357e-01,  7.60825509e-01],
       [ 7.09558288e-02,  5.21554622e-01],
       [ 4.15666697e-01,  3.50340211e-01],
       [ 1.25271549e-01,  5.31461197e-01],
       [-6.12954952e-01,  7.60951462e-01],
       [-6.41967013e-01,  8.51900808e-01],
       [ 9.48512105e-02,  5.12354992e-01],
       [-3.00160991e-01,  6.49329163e-01],
       [ 1.10353448e+00,  1.64677456e-01],
       [ 9.26141202e-02,  5.11243507e-01],
       [ 9.35704732e-02,  5.12229618e-01],
       [-6.81012759e-01,  8.55046893e-01],
       [-8.21592262e-01,  8.57716836e-01],
       [ 3.72833431e-01,  3.18701895e-01],
       [ 4.33654591e-01,  3.55812257e-01],
       [ 1.98754719e+00,  9.04854659e-01],
       [-6.50534138e-01,  8.48510870e-01],
       [ 3.51608651e-01,  3.82229045e-01],
       [ 4.15430586e-01,  3.46852738e-01],
       [ 1.47147864e-01,  4.90059602e-01],
       [-6.42831925e-01,  7.74206707e-01],
       [ 1.13195497e-01,  8.74852800e-01],
       [ 4.45728439e-01,  3.60165431e-01],
       [ 4.74435544e-01,  3.80870132e-01],
       [ 2.83193881e-01,  4.37281902e-01],
       [-9.11020993e-01,  8.75739848e-01],
       [ 5.77138682e-02,  5.26205419e-01],
       [ 3.55783883e-01,  3.79599221e-01],
       [-7.48822753e-01,  7.91270072e-01],
       [ 3.81243489e-01,  3.49868114e-01],
       [-5.19547647e-01,  8.33387636e-01],
       [ 4.41973414e-01,  3.77934652e-01],
       [ 2.84375665e-01,  4.34148475e-01],
       [ 3.82953176e-01,  3.96981648e-01],
       [-7.15477413e-01,  8.20765320e-01],
       [ 2.89787318e-02,  5.29816351e-01],
       [-8.17003975e-01,  8.35277150e-01],
       [ 4.12919953e-01,  4.01556187e-01],
       [-5.70810990e-01,  8.38755032e-01],
       [-1.66860503e-01, -4.61630819e-02],
       [ 4.94102745e-01,  3.73072459e-01],
       [ 3.88503148e-01,  3.94508144e-01],
       [-7.60100845e-01,  8.21933099e-01],
       [ 4.17305335e-01,  3.97805749e-01],
       [ 1.62747417e-04,  5.45127208e-01],
       [-5.25205383e-01,  9.45790513e-01],
       [ 1.28475638e-01,  4.96974967e-01]])
    
    M = np.max(a)
    m = np.min(a)
    a = (a - m)/(M - m) * 512
    return a,M, m

#candidates = run_mixtures(a, [10])
if __name__ == '__main__':
    import pclines
    import matplotlib.pyplot as plt
    import matplotlib
    a,M,m = get_a()
    candidates = pclines.run_mixtures(a,[30,30,30], draw=True)
    n_candidates = len(candidates) * 2
    print(candidates)
    candidates = candidates.ravel()
    candidates = np.r_[np.take(candidates, np.arange(0,len(candidates),2)),np.take(candidates, np.arange(1,len(candidates),2))]
    print(candidates)
    points = list(a.transpose().ravel())
    detections, nout = use_alignments_fast(points, 2, len(a), candidates,n_candidates )
    
    if(len(detections) > 0):
        detections = np.array(np.array_split(detections, nout))
    if(not len(detections) == 0):
        dets = detections[:, 0:4]
        dets = dets/512 * (M -m)+ m
        detections[:, 0:4] = dets
        x1= dets[:,0]
        y1= dets[:,1]
        x2= dets[:,2]
        y2= dets[:,3]
        
        dy = y2 - y1
        dx = x2 - x1
        m = dy/dx
        b = (y1 * x2 - y2 * x1)/dx
        for i in range(len(m)):
            x = np.arange(0, 512, 1)
            y= m[i] * x + b[i] 
            plt.plot(x, y)
        plt.show()
#        x = np.arange(0, 512, 1)
#        y= m * x + b 
#        plt.plot(x, y,'go--', linewidth=2, markersize=5)
#        print(dets, m, b)
    plt.show()
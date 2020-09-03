import random
import numpy as np
def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=30):
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    data = list(data)
    for i in range(max_iterations):        
#        print('len of data is ', len(data))
#        if(len(data) == 2):
#            s = data
#        else:
        s = random.sample(data, int(sample_size))
        m = estimate(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                ic += 1

#        print(s)
#        print('estimate:', m,)
#        print('# inliers:', ic)

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
#        print('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
#    print(best_model, best_ic)
    return best_model, best_ic, s
def augment(xys):
    axy = np.ones((len(xys), 3))
    axy[:, :2] = xys
    return axy

def estimate(xys):
    axy = augment(xys[:2])
    return np.linalg.svd(axy)[-1][-1, :]

def is_inlier(coeffs, xy, threshold):
    return np.abs(coeffs.dot(augment([xy]).T)) < threshold
#def get_lines_coeffs(lines_xy_coordinates, radius = 22, img_shape=
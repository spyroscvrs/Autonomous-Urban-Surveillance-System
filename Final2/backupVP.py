import os
import sys
import argparse
import ipdb
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage import io, feature, color, transform
import cv2

st = ipdb.set_trace

def read_image(path):
    image = io.imread(path)
    return image

def get_canny_edges(image, sigma):
    edges = feature.canny(color.rgb2gray(image), sigma=sigma)
    return edges

def get_hough_lines(edges, line_length, line_gap):
    lines = transform.probabilistic_hough_line(edges, line_length=line_length, line_gap=line_gap)
    return np.asarray(lines)

def visualize_inliers(image, edges, lines, inlier_lines_list, colors, fig_name='detected_lines.png'):
    subplot_count = len(inlier_lines_list) + 3

    fig, axes = plt.subplots(3, subplot_count-3, figsize=(15, 15), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')

    ax[1].imshow(edges, cmap=cm.gray)
    ax[1].set_title('Canny edges')

    ax[2].imshow(edges * 0)
    for line in lines:
        p0, p1 = line
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim((0, image.shape[1]))
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_title('Probabilistic Hough')
    for i in range(len(inlier_lines_list)):
        ax[i+3].imshow(edges * 0)
        for line in lines[inlier_lines_list[i]]:
            p0, p1 = line
            ax[i+3].plot((p0[0], p1[0]), (p0[1], p1[1]), colors[i])
        ax[i+3].set_xlim((0, image.shape[1]))
        ax[i+3].set_ylim((image.shape[0], 0))
        ax[i+3].set_title('RANSAC {} Inliers'.format(str(i)))

    for a in ax:
        a.set_axis_off()

    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()

def visualize_vanishing_points(vp1, vp2, vp3, image, lines, edges, inlier_lines_list, colors, fig_name):
    vps = [vp1, vp2, vp3]
    for i in range(len(inlier_lines_list)):
        plt.imshow(image)
        for line in lines[inlier_lines_list[i]]:
            p0, p1 = line
            plt.plot((p0[0], p1[0]), (p0[1], p1[1]), colors[i])

        plt.plot([vps[i][0]], [vps[i][1]], colors[i]+'X', markersize=5)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(fig_name.split('.')[0] + str(i) + '.png') 
        plt.close()

    plt.imshow(image)
    for i in range(len(inlier_lines_list)):
        for line in lines[inlier_lines_list[i]]:
            p0, p1 = line
            plt.plot((p0[0], p1[0]), (p0[1], p1[1]), colors[i])

    plt.plot([vps[0][0]], [vps[0][1]], colors[0]+'X', markersize=5)
    plt.plot([vps[1][0]], [vps[1][1]], colors[1]+'X', markersize=5)
    plt.plot([vps[2][0]], [vps[2][1]], colors[2]+'X', markersize=5)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(fig_name) 
    plt.close() 

def calculate_metric_angle(current_hypothesis, lines, ignore_pts, ransac_angle_thresh):
    current_hypothesis = current_hypothesis / current_hypothesis[-1]
    hypothesis_vp_direction = current_hypothesis[:2] - lines[:,0]
    lines_vp_direction = lines[:,1] - lines[:,0]
    magnitude = np.linalg.norm(hypothesis_vp_direction, axis=1) * np.linalg.norm(lines_vp_direction, axis=1)
    magnitude[magnitude == 0] = 1e-5
    cos_theta = (hypothesis_vp_direction*lines_vp_direction).sum(axis=-1) / magnitude
    theta = np.arccos(np.abs(cos_theta))
    inliers = (theta < ransac_angle_thresh * np.pi / 180)
    inliers[ignore_pts] = False
    return inliers, inliers.sum()

def run_line_ransac(lines, ransac_iter, ransac_angle_thresh, ignore_pts=None):
    best_vote_count = 0
    best_inliers = None
    best_hypothesis = None
    if ignore_pts is None:
        ignore_pts = np.zeros((lines.shape[0])).astype('bool')
        lines_to_chose = np.arange(lines.shape[0])
    else:
        lines_to_chose = np.where(ignore_pts==0)[0]
    for iter_count in range(ransac_iter):
        idx1, idx2 = np.random.choice(lines_to_chose, 2, replace=False)
        l1 = np.cross(np.append(lines[idx1][1], 1), np.append(lines[idx1][0], 1))
        l2 = np.cross(np.append(lines[idx2][1], 1), np.append(lines[idx2][0], 1))

        current_hypothesis = np.cross(l1, l2)
        if current_hypothesis[-1] == 0:
            continue
        inliers, vote_count = calculate_metric_angle(current_hypothesis, lines, ignore_pts, ransac_angle_thresh)
        if vote_count > best_vote_count:
            best_vote_count = vote_count
            best_hypothesis = current_hypothesis
            best_inliers = inliers
    return best_hypothesis, best_inliers

def get_vp_inliers(image_path, sigma, iterations, line_len, line_gap, threshold):
    image = image_path
    edges = get_canny_edges(image, sigma=sigma)
    lines = get_hough_lines(edges, line_length=line_len, line_gap=line_gap)

    best_hypothesis_1, best_inliers_1 = run_line_ransac(lines, iterations, threshold)
    ignore_pts = best_inliers_1
    best_hypothesis_2, best_inliers_2 = run_line_ransac(lines, iterations, threshold, ignore_pts=ignore_pts)
    ignore_pts = np.logical_or(best_inliers_1, best_inliers_2)
    best_hypothesis_3, best_inliers_3 = run_line_ransac(lines, iterations, threshold, ignore_pts=ignore_pts)
    inlier_lines_list = [best_inliers_1, best_inliers_2, best_inliers_3]
    best_hypothesis_1 = best_hypothesis_1 / best_hypothesis_1[-1]
    best_hypothesis_2 = best_hypothesis_2 / best_hypothesis_2[-1]
    best_hypothesis_3 = best_hypothesis_3 / best_hypothesis_3[-1]
    hypothesis_list = [best_hypothesis_1, best_hypothesis_2, best_hypothesis_3]
    viz_stuff = [image, edges, lines]
    return inlier_lines_list, hypothesis_list, viz_stuff

def main(frame):
    #cap=cv2.VideoCapture(video_src)
    #ret,frame = cap.read()
    height,width = frame.shape[:2]
    c=[int(height/2),int(width/2)]
    #while(ret):
    sigma=2
    iterations=3000
    line_len=15
    line_gap=1
    threshold=3
    inlier_lines_list, hypothesis_list, viz_stuff = get_vp_inliers(frame, sigma, iterations, line_len, line_gap, threshold)
    image, edges, lines = viz_stuff
    best_hypothesis_1, best_hypothesis_2, best_hypothesis_3 = hypothesis_list
    arrayv=[[int(best_hypothesis_1[0]),int(best_hypothesis_1[1])],[int(best_hypothesis_2[0]),int(best_hypothesis_2[1])],[int(best_hypothesis_3[0]),int(best_hypothesis_3[1])]]
    d=10000000000
    u=[0,0]
    for i in range(3):
        newv=np.subtract(arrayv[i],c)
        distance=np.linalg.norm(newv)
        if  distance<d:
            d=distance
            u=arrayv[i]
            k=i
    del arrayv[k]

    u=[u[0],u[1],0]
    return u
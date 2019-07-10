#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:13:20 2019

@author: mgontar
"""
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)

    return (int(min(x_coordinates)), int(min(y_coordinates)), int(max(x_coordinates)-min(x_coordinates)), int(max(y_coordinates)-min(y_coordinates)))

def get_iou(bb1, bb2):
    x11 = bb1[0]
    x12 = bb1[0]+bb1[2]
    y11 = bb1[1]
    y12 = bb1[1]+bb1[3]
    x21 = bb2[0]
    x22 = bb2[0]+bb2[2]
    y21 = bb2[1]
    y22 = bb2[1]+bb2[3]
    # determine the coordinates of the intersection rectangle
    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (x12 - x11) * (y12 - y11)
    bb2_area = (x22 - x21) * (y22 - y21)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


datasets = ['DragonBaby', 'Surfer', 'Ironman']
initial_rects = [(160,83,56,65), (275,137,23,26), (206,85,49,57)]
dataset_sizes = [113, 376, 166]
dataset_gt_sep = [',','\t',',']

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

for dsi in range(len(datasets)):

    path = './datasets/'+datasets[dsi]+'/img/'
    file_path = path+'0001.jpg'
    
    # take first frame of the video
    frame = cv2.imread(file_path, cv2.IMREAD_COLOR)
    hsv_old = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width, channels = frame.shape
    # setup initial location of window
    c,r,w,h = initial_rects[dsi]  # simply hardcoded the values
    
    # set up the ROI for tracking
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
    video = cv2.VideoWriter(datasets[dsi]+'.avi', 0, 1, (width,height))
    
    gt_rects = np.genfromtxt('./datasets/'+datasets[dsi]+'/groundtruth_rect.txt',delimiter=dataset_gt_sep[dsi],dtype=int)
    
    iou_lk = []
    
    # points to track
    p0 = []
    for x in range(w):
        for y in range(h):
           p0.append([c+x, r+y]) 
           
    p0 = np.array(p0, dtype=np.float32)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame)
    
    for i in range(1, dataset_sizes[dsi]):
        frame = cv2.imread(path+'{:04d}.jpg'.format(i), cv2.IMREAD_COLOR)
        hsv_new = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        co,ro,wo,ho = gt_rects[i]
        cv2.rectangle(frame, (co,ro), (co+wo,ro+ho), (255, 0, 0),2)
        
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(hsv_old, hsv_new, p0, None, **lk_params)

        c,r,w,h = bounding_box(p1)
        
        
        track_window_lk = (c,r,w,h)
        
        # Draw it on image
        cv2.rectangle(frame, (c,r), (c+w,r+h), (0, 255, 0),2)
        
        iou = get_iou(gt_rects[i], track_window_lk)
        iou_lk.append(iou)
        
        cv2.putText(frame, 'IoU LK:{:03.2f}'.format(iou), (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

    
        video.write(frame)
        
        hsv_old = hsv_new
    

    fig = plt.figure()
    ax = plt.axes()
    plt.title('LK performance on '+datasets[dsi]+' dataset')
    plt.xlabel('Frame #')
    plt.ylabel('Intersection over Union (IoU)');
    frames = np.linspace(1, dataset_sizes[dsi]-1, num = dataset_sizes[dsi]-1)
    
    ax.plot(frames, iou_lk, color='#00FF00', label="LK")

    plt.legend()
    plt.savefig(datasets[dsi]+'.png')
    cv2.destroyAllWindows()
    video.release()
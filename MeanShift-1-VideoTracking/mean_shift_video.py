#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:13:20 2019

@author: mgontar
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

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



datasets = ['DragonBaby', 'Surfer', 'Ironman', 'Deer', 'Girl']
initial_rects = [(160,83,56,65), (275,137,23,26), (206,85,49,57), (306,5,95,65), (57,21,31,45)]
dataset_sizes = [113, 376, 166, 71, 500]
dataset_gt_sep = [',','\t',',',',','\t']

for dsi in range(len(datasets)):

    path = './datasets/'+datasets[dsi]+'/img/'
    file_path = path+'0001.jpg'
    
    # take first frame of the video
    frame = cv2.imread(file_path, cv2.IMREAD_COLOR)
    height, width, channels = frame.shape
    # setup initial location of window
    c,r,w,h = initial_rects[dsi]  # simply hardcoded the values
    track_window_ms = (c,r,w,h)
    track_window_cs = (c,r,w,h)
    
    # set up the ROI for tracking
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    
    video = cv2.VideoWriter(datasets[dsi]+'.avi', 0, 1, (width,height))
    
    gt_rects = np.genfromtxt('./datasets/'+datasets[dsi]+'/groundtruth_rect.txt',delimiter=dataset_gt_sep[dsi],dtype=int)
    
    iou_ms = []
    iou_cs = []    
    
    for i in range(1, dataset_sizes[dsi]):
        frame = cv2.imread(path+'{:04d}.jpg'.format(i), cv2.IMREAD_COLOR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    
        x,y,w,h = gt_rects[i]
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0),2)
        # apply meanshift to get the new location
        ret, track_window_ms = cv2.meanShift(dst, track_window_ms, term_crit)
    
        # Draw it on image
        x,y,w,h = track_window_ms
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0),2)
        
        iou = get_iou(gt_rects[i], track_window_ms)
        iou_ms.append(iou)
        
        cv2.putText(frame, 'IoU MS:{:03.2f}'.format(iou), (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
        
        # apply meanshift to get the new location
        ret, track_window_cs = cv2.CamShift(dst, track_window_cs, term_crit)
        
        # Draw it on image
        x,y,w,h = track_window_cs
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255),2)

        iou = get_iou(gt_rects[i], track_window_cs)
        iou_cs.append(iou)
        cv2.putText(frame, 'IoU CS:{:03.2f}'.format(iou), (2,20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
    
        video.write(frame)
    

    fig = plt.figure()
    ax = plt.axes()
    plt.title('MeanShift and CamShift performance on '+datasets[dsi]+' dataset')
    plt.xlabel('Frame #')
    plt.ylabel('Intersection over Union (IoU)');
    frames = np.linspace(1, dataset_sizes[dsi]-1, num = dataset_sizes[dsi]-1)
    
    ax.plot(frames, iou_ms, color='#00FF00', label="MeanShift")
    ax.plot(frames, iou_cs, color='#FF0000', label="CamShift")
    plt.legend()
    plt.savefig(datasets[dsi]+'.png')
    cv2.destroyAllWindows()
    video.release()
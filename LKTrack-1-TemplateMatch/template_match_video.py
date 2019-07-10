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

for dsi in range(len(datasets)):

    path = './datasets/'+datasets[dsi]+'/img/'
    file_path = path+'0001.jpg'
    
    # take first frame of the video
    frame = cv2.imread(file_path, cv2.IMREAD_COLOR)
    height, width, channels = frame.shape
    # setup initial location of window
    c,r,w,h = initial_rects[dsi]  # simply hardcoded the values
    
    # set up the ROI for tracking
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
    video = cv2.VideoWriter(datasets[dsi]+'.avi', 0, 1, (width,height))
    
    gt_rects = np.genfromtxt('./datasets/'+datasets[dsi]+'/groundtruth_rect.txt',delimiter=dataset_gt_sep[dsi],dtype=int)
    
    iou_ssd = []
    iou_ncc = [] 
    iou_sad = [] 
    
    for i in range(1, dataset_sizes[dsi]):
        frame = cv2.imread(path+'{:04d}.jpg'.format(i), cv2.IMREAD_COLOR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        co,ro,wo,ho = gt_rects[i]
        cv2.rectangle(frame, (co,ro), (co+wo,ro+ho), (255, 0, 0),2)
        
        # apply SSD to get the new location
        res = cv2.matchTemplate(hsv,hsv_roi,cv2.TM_SQDIFF)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        c,r = min_loc
        
        track_window_ssd = (c,r,w,h)
        
        # Draw it on image
        cv2.rectangle(frame, (c,r), (c+w,r+h), (0, 255, 0),2)
        
        iou = get_iou(gt_rects[i], track_window_ssd)
        iou_ssd.append(iou)
        
        cv2.putText(frame, 'IoU SSD:{:03.2f}'.format(iou), (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

        
        # apply NCC to get the new location
        res = cv2.matchTemplate(hsv,hsv_roi,cv2.TM_CCORR_NORMED)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        c,r = max_loc
        
        track_window_ncc = (c,r,w,h)
        
        # Draw it on image
        cv2.rectangle(frame, (c,r), (c+w,r+h), (0, 0, 255),2)
        
        iou = get_iou(gt_rects[i], track_window_ncc)
        iou_ncc.append(iou)

        cv2.putText(frame, 'IoU NCC:{:03.2f}'.format(iou), (2,20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
        
        # apply SAD to get the new location
        minSAD = sys.maxsize

        # loop through the search image
        for sad_y in range(height-h + 1):
            for sad_x in range(width-w + 1):
                SAD = np.sum(np.abs(hsv[sad_y:sad_y+h, sad_x:sad_x+w] - hsv_roi))
                if (minSAD > SAD):
                    minSAD = SAD
                    c = sad_y
                    r = sad_x
                
        track_window_sad = (c,r,w,h)
        
        # Draw it on image
        cv2.rectangle(frame, (c,r), (c+w,r+h), (255, 0, 255),2)
        
        iou = get_iou(gt_rects[i], track_window_sad)
        iou_sad.append(iou)

        cv2.putText(frame, 'IoU SAD:{:03.2f}'.format(iou), (2,30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)
    
        video.write(frame)
    

    fig = plt.figure()
    ax = plt.axes()
    plt.title('SSD, NCC and SAD performance on '+datasets[dsi]+' dataset')
    plt.xlabel('Frame #')
    plt.ylabel('Intersection over Union (IoU)');
    frames = np.linspace(1, dataset_sizes[dsi]-1, num = dataset_sizes[dsi]-1)
    
    ax.plot(frames, iou_ssd, color='#00FF00', label="SSD")
    ax.plot(frames, iou_ncc, color='#FF0000', label="NCC")
    ax.plot(frames, iou_sad, color='#FF00FF', label="SAD")
    plt.legend()
    plt.savefig(datasets[dsi]+'.png')
    cv2.destroyAllWindows()
    video.release()
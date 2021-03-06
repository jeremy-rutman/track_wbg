__author__ = 'jeremy'

import numpy as np
import cv2
#from ml_utils import imutils
import os
import copy

class bg_sub():
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorKNN()
        self.visual_output=False
        self.smallest_box_fraction=0.005  #if box is smaller than this fraction of image, drop it
        self.movement_threshold = 70 #graylevel out of 255, thresh on createBackgroundSubtractorKnn.apply
        self.ioma_thresh = 0.05 #any overlap of two boxes greater than this causes smaller to get removed
        self.big_enough_contours=[]
        self.frame_no = 0

    def next_frame(self,img_arr):
        self.frame_no += 1
        frame=copy.copy(img_arr)
       # print('frame {}'.format(self.frame_no))
        h,w=frame.shape[0:2]
        area=h*w
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blursize=15
        frame_blur = cv2.GaussianBlur(frame_gray, (blursize,blursize), 0)
        fgmask = self.fgbg.apply(frame_blur)

        thresh = cv2.threshold(fgmask, self.movement_threshold, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
     #   print('cnts:{}'.format(contours))
        # loop over the contours
        self.big_enough_boxes=[]
        for c in contours:
            # if the contour is too small, ignore it
            if cv2.contourArea(c)/area < self.smallest_box_fraction:
                continue
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
          #  print('box frac of image {}'.format(cv2.contourArea(c)/area))
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box = [x,y,w,h]
            print('{} had enough motion'.format(box))
        #check if lots of overlap
            too_much_overlap=False
            for i,det in enumerate(self.big_enough_boxes):
                other_box = det['bbox_xywh']
                print('check overlap bet {} and {}'.format(box,other_box))
                intersection_over_minarea=intersectionOverMinArea(other_box,box)
                if intersection_over_minarea>self.ioma_thresh:
                    too_much_overlap=True
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    combined_box = combine_bbs(box,other_box)
                    self.big_enough_boxes[i]={'bbox_xywh':combined_box}
                    cv2.rectangle(frame, (combined_box[0],combined_box[1]),
                                  (combined_box[0]+combined_box[2],combined_box[1] + combined_box[3]), (0, 0, 255), 2)
                    print('too much overlap, combined {} and {} into {}'.format(other_box,box,combined_box))
                    # area1=box[2]*box[3]
                    # area2=other_box[2]*other_box[3]
                    # if area2>area1: #otherbox is bigger, remove current
                    #     print(' combining new box {}'.format(box))
                    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                    #     combined_box = imutils.combine_bbs(box,other_box)
                    #     self.big_enough_boxes[i]=combined_box
                    #     cv2.rectangle(frame, (combined_box[0],combined_box[1]),
                    #                   (combined_box[0]+combined_box[2],combined_box[1] + combined_box[3]), (100, 100, 255), 2)
                    #     dont_add_this_box=True
                    #     break
                    # else:
                    #     print(' combining old box {}'.format(self.big_enough_boxes[i]))
                    #     x2,y2,w2,h2=self.big_enough_boxes[i]
                    #     cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 255), 2)
                    #     del self.big_enough_boxes[i]
            if not too_much_overlap:
                print('{} did not have too much overlap'.format(box))
                self.big_enough_boxes.append({'bbox_xywh':box})


        if self.visual_output:
            cv2.imshow('motion dets',frame)
#            cv2.imshow('thresh',thresh)
#            cv2.imshow('fgmask2',fgmask)
            k = cv2.waitKey(0) & 0xff
        if self.frame_no<5:
            return([])
        else:
            print('motion_detect found {} boxes:{}'.format(len(self.big_enough_boxes),self.big_enough_boxes))
            return self.big_enough_boxes

def intersectionOverMinArea(r1,r2):
    '''
    r1,r2 in form xywh
    :param r1:
    :param r2:
    :return:
    '''

    intersectionx = int(max(r1[0], r2[0]))
    intersectiony = int(max(r1[1], r2[1]))
    intersectionw = int(min(r1[0] + r1[2], r2[0] + r2[2])) - int(intersectionx)
    if intersectionw < 0:
        intersectionw = 0
    intersectionh = int(min(r1[1] + r1[3], r2[1] + r2[3])) - int(intersectiony)
    if intersectionh < 0:
        intersectionh = 0
        # intersectionh -= intersectiony;
        # print('r1:' + str(r1) + ' r2:' + str(r2) + ' x,y,w,h:' + str(intersectionx) + ',' + str(intersectiony) + ',' + str(
        # intersectionw) + ',' + str(
        # intersectionh))
    min_area=min(r1[2]*r1[3],r2[2]*r2[3])
    intersectionarea = intersectionw * intersectionh
    frac = float(intersectionarea) / float(min_area)
    print('min_area,intarea,frac:' + str(min_area) + ',' + str(intersectionarea) + ',' + str(frac))
    return (frac)

def combine_bbs(bb1_xywh,bb2_xywh):
    minx=min(bb1_xywh[0],bb2_xywh[0])
    maxx=max(bb1_xywh[0]+bb1_xywh[2],bb2_xywh[0]+bb2_xywh[2])
    miny=min(bb1_xywh[1],bb2_xywh[1])
    maxy=min(bb1_xywh[1]+bb1_xywh[3],bb2_xywh[1]+bb2_xywh[3])
    w=maxx-minx
    h=maxy-miny
    return(minx,miny,w,h)


if __name__ == "__main__":
    picdir = '/data/jeremy/image_dbs/variant/viettel_demo/'
    files = [f for f in os.listdir(picdir) if f[-4:]=='.jpg']
 #   fgbg = cv2.createBackgroundSubtractorMOG2()
    fgbg2 = cv2.createBackgroundSubtractorKNN()
 #   fgbg2 = cv2.createBackgroundSubtractorMOG2()
    files.sort()
    img_list = []
    n_for_median = 10
    n=0
    bg_subtractor= bg_sub()
    for f in files:
        full_path = os.path.join(picdir,f)
        if not os.path.isfile(full_path):
            print('not a good file...')
            continue
        print('working on '+full_path)
        frame = cv2.imread(full_path)
        bg_subtractor.next_frame(frame)

    cv2.destroyAllWindows()

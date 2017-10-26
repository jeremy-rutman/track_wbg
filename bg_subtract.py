__author__ = 'jeremy'

import numpy as np
import cv2
from ml_utils import imutils
import os


class bg_sub():
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorKNN()
        self.visual_output=True
        self.smallest_box_fraction=0.004
        self.movement_threshold = 50 #graylevel out of 255, thresh on createBackgroundSubtractorKnn.apply
        self.big_enough_contours=[]

    def next_frame(self,img_arr):
        h,w=frame.shape[0:2]
        area=h*w
        frame_gray = cv2.cvtColor(img_arr,cv2.COLOR_BGR2GRAY)
        blursize=11
        frame_blur = cv2.GaussianBlur(frame_gray, (blursize,blursize), 0)
        fgmask = self.fgbg.apply(frame_blur)

        thresh = cv2.threshold(fgmask, self.movement_threshold, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
     #   print('cnts:{}'.format(contours))
        # loop over the contours
        self.big_enough_contours=[]
        for c in contours:
            # if the contour is too small, ignore it
            if cv2.contourArea(c)/area < self.smallest_box_fraction:
                continue
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            print('frac {}'.format(cv2.contourArea(c)/area))
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.big_enough_contours.append([x,y,w,h])
        #check if lots of overlap
            # for other_c in self.big_enough_contours:
            #     if

        if self.visual_output:
            cv2.imshow('orig',frame)
            cv2.imshow('thresh',thresh)
            cv2.imshow('fgmask2',fgmask)
            k = cv2.waitKey(30) & 0xff
        return contours


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

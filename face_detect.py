__author__ = 'jeremy'

import dlib
import os
import cv2
import time
import face_recognition
import face_recognition_models

from ml_utils import imutils


detector = dlib.get_frontal_face_detector()

def find_face_dlib(image, max_num_of_faces=10):
    faces = detector(image, 1)
    faces = [[rect.left(), rect.top(), rect.width(), rect.height()] for rect in list(faces)]
    print('dlib found {} faces'+str(len(faces)))
    if not len(faces):
        return {'are_faces': False, 'faces': []}
    final_faces = choose_faces(image, faces, max_num_of_faces)
    return {'are_faces': len(final_faces) > 0, 'faces': final_faces}

def find_face_fr(image_file,visual_output=True,threshold=0.2):
    image = face_recognition.load_image_file(image_file)
    faces = face_recognition.face_locations(image)
    output_dict=[]
    for i, d in enumerate(faces):
        print("Detection {}, score: {}, face_type:{}".format(d, scores[i], idx[i]))
        print d
        if 2>threshold:
            output_dict.append({'object':'face','confidence':scores[i],'box_xywh':[d.left(),d.top(),d.left()+d.width(),d.top()+d.height()]})

            cv2.rectangle(image,(d.left(), d.top()), (d.left()+d.width(),d.top()+d.height()),[255,100,0],thickness=2)
            cv2.putText(image,'face'+str(round(scores[i],3)),(d.left()+10,d.top()+10),cv2.FONT_HERSHEY_COMPLEX,0.5,[100,200,100])
    if visual_output:
        cv2.imshow('dets',image)
        cv2.waitKey(30)


def find_face_dlib_with_scores(image, visual_output=True,threshold=0.2):
    '''
    return the full info including scores
    :param image:
    :param max_num_of_faces:
    :return:
    '''
    start=time.time()
    if isinstance(image,basestring):
        image = imutils.get_cv2_img_array(image)
   ## faces, scores, idx = detector.run(image, 1, -1) - gives more results, those that add low confidence percentage ##
    ## faces, scores, idx = detector.run(image, 1, 1) - gives less results, doesn't show the lowest confidence percentage results ##
        ## faces, scores, idx = detector.run(image, 2) - gives more results using more time by scaling ##

    faces, scores, idx = detector.run(image,1)
    print("dlib found {} faces in {} s.".format(len(faces),(time.time() - start)))

    output_dict=[]
    for i, d in enumerate(faces):
        print("Detection {}, score: {}, face_type:{}".format(d, scores[i], idx[i]))
        if scores[i]>threshold:
            output_dict.append({'object':'face','confidence':scores[i],'box_xywh':[d.left(),d.top(),d.left()+d.width(),d.top()+d.height()]})

            cv2.rectangle(image,(d.left(), d.top()), (d.left()+d.width(),d.top()+d.height()),[255,100,0],thickness=2)
            cv2.putText(image,'face'+str(round(scores[i],3)),(d.left()+10,d.top()+10),cv2.FONT_HERSHEY_COMPLEX,0.5,[100,200,100])
            if visual_output:
                cv2.imshow('dets',image)
                cv2.waitKey(30)


    #final_faces = choose_faces(image, faces, max_num_of_faces)

    return(output_dict)

class c_t():
    def __init__(self,initial_image_array,bbox_xywh):
        self.tracker=dlib.correlation_tracker()
        bbox_x1y1x2y2=[long(bbox_xywh[0]),long(bbox_xywh[1]),long(bbox_xywh[0]+bbox_xywh[2]),long(bbox_xywh[1]+bbox_xywh[3])]
        print('tracker being started with box {}'.format(bbox_x1y1x2y2))
##        self.tracker.start_track(initial_image_array,dlib.rectangle(*[1,2,3,4]))
        self.tracker.start_track(initial_image_array,dlib.rectangle(*bbox_x1y1x2y2))
        self.visual_output=True

    def next_frame(self,img_arr):
        self.tracker.update(img_arr)
        rect = self.tracker.get_position()
        pt1 = (int(rect.left()), int(rect.top()))
        pt2 = (int(rect.right()), int(rect.bottom()))
        print "Object tracked at [{}, {}] \r".format(pt1, pt2),
        if self.visual_output:public

            cv2.rectangle(img_arr, pt1, pt2, (255, 0, 255), 3)
            loc = (int(rect.left()), int(rect.top()-20))
            txt = "corrtrack {}, {}".format(pt1, pt2)
            cv2.putText(img_arr, txt, loc , cv2.FONT_HERSHEY_SIMPLEX, .5, (255,0,255), 1)
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Image", img_arr)
            # Continue until the user presses ESC key
            cv2.waitKey(1)
        bbox_xywh = [rect.left(),rect.top(),rect.width,rect.height()]
        return bbox_xywh


if __name__ =="__main__":
    picdir = '/data/jeremy/image_dbs/variant/viettel_demo/'
    files = [f for f in os.listdir(picdir) if f[-4:]=='.jpg']
    files.sort()
    for i,f in enumerate(files):
        full_path = os.path.join(picdir,f)
        if not os.path.isfile(full_path):
            print('not a good file...')
            continue
        print('working on '+full_path)
        start_time = time.time()
        info = find_face_dlib_with_scores(full_path)
        info = find_face_fr(full_path)
        print('info:'+str(info))
        elapsed = time.time()-start_time
   #     raw_input('ret for next frame')
        print('tot elapsed {} for frame {}\n\n'.format(elapsed,i))

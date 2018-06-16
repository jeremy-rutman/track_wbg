__author__ = 'jeremy'

import dlib
import os
import cv2
import time
import face_recognition
import face_recognition_models
import logging

#from ml_utils import imutils


detector = dlib.get_frontal_face_detector()
#detector = dlib.cnn_face_detection_model_v1() #supposed to be better

def find_face_dlib(image, max_num_of_faces=10):
    faces = detector(image, 1)
    faces = [[rect.left(), rect.top(), rect.width(), rect.height()] for rect in list(faces)]
    print('dlib found {} faces'+str(len(faces)))
    if not len(faces):
        return {'are_faces': False, 'faces': []}
    final_faces = choose_faces(image, faces, max_num_of_faces)
    return {'are_faces': len(final_faces) > 0, 'faces': final_faces}

def find_face_fr(image_file,visual_output=False,threshold=0.2):
    image = face_recognition.load_image_file(image_file)
    faces = face_recognition.face_locations(image)
    output_dict=[]
    for i, d in enumerate(faces):
        print("Detection {}, score: {}, face_type:{}".format(d, scores[i], idx[i]))
        print d
        if 2>threshold:
            output_dict.append({'object':'face','confidence':scores[i],'bbox_xywh':[d.left(),d.top(),d.left()+d.width(),d.top()+d.height()]})

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
        image = get_cv2_img_array(image)
   ## faces, scores, idx = detector.run(image, 1, -1) - gives more results, those that add low confidence percentage ##
    ## faces, scores, idx = detector.run(image, 1, 1) - gives less results, doesn't show the lowest confidence percentage results ##
        ## faces, scores, idx = detector.run(image, 2) - gives more results using more time by scaling ##

    faces, scores, idx = detector.run(image,1)
    print("dlib found {} faces in {} s.".format(len(faces),(time.time() - start)))

    output_dict=[]
    for i, d in enumerate(faces):
        print("Detection {}, score: {}, face_type:{}".format(d, scores[i], idx[i]))
        if scores[i]>threshold:
            output_dict.append({'object':'face','confidence':scores[i],'bbox_xywh':[d.left(),d.top(),d.left()+d.width(),d.top()+d.height()]})

            cv2.rectangle(image,(d.left(), d.top()), (d.left()+d.width(),d.top()+d.height()),[255,100,0],thickness=2)
            cv2.putText(image,'face'+str(round(scores[i],3)),(d.left()+10,d.top()+10),cv2.FONT_HERSHEY_COMPLEX,0.5,[100,200,100])
            if visual_output:
                cv2.imshow('dets',image)
                cv2.waitKey(30)


    #final_faces = choose_faces(image, faces, max_num_of_faces)c

    return(output_dict)

class c_t():
    def __init__(self,initial_image_array,bbox_xywh):
        self.tracker=dlib.correlation_tracker()
        bbox_x1y1x2y2=[long(bbox_xywh[0]),long(bbox_xywh[1]),long(bbox_xywh[0]+bbox_xywh[2]),long(bbox_xywh[1]+bbox_xywh[3])]
        print('tracker being started with box {}'.format(bbox_x1y1x2y2))
##        self.tracker.start_track(initial_image_array,dlib.rectangle(*[1,2,3,4]))
        self.tracker.start_track(initial_image_array,dlib.rectangle(*bbox_x1y1x2y2))
        self.visual_output=False

    def next_frame(self,img_arr):
        self.tracker.update(img_arr)
        rect = self.tracker.get_position()
        pt1 = (int(rect.left()), int(rect.top()))
        pt2 = (int(rect.right()), int(rect.bottom()))
        print "Object tracked at [{}, {}] \r".format(pt1, pt2),
        if self.visual_output:
            cv2.rectangle(img_arr, pt1, pt2, (255, 0, 255), 3)
            loc = (int(rect.left()), int(rect.top()-20))
            txt = "corrtrack {}, {}".format(pt1, pt2)
            cv2.putText(img_arr, txt, loc , cv2.FONT_HERSHEY_SIMPLEX, .5, (255,0,255), 1)
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Image", img_arr)
            # Continue until the user presses ESC key
            cv2.waitKey(1)
        bbox_xywh = [int(rect.left()),int(rect.top()),int(rect.width()),int(rect.height())]
        return bbox_xywh

def dlib_speed_test(picdir = '/data/jeremy/image_dbs/variant/longfeld_video/'):
#    export OPENBLAS_NUM_THREADS=1 in the environment variables.
#avg 0.688s with visual_output=True on single cpu
#0.683s with visual_output=False
#0.634s with openblas_um_threads=4, 0.64s n=1

#https://github.com/cmusatyalab/openface/issues/157
# If you are interested in details:
#
# Dlib's face detector has no GPU support
# Detector does not use color information - greyscale images will work faster
# For best performance you can downscale images, because dlib works with small 80x80 faces for detection and use scanner.set_max_pyramid_levels(1) to exclude scanning of large faces. And also face detector has 5 cascades inside - they can be splitted if you need only frontal-looking faces for example;
# GCC-compiled code works much faster than any MSVC. MSVC 2015 works much faster than MSVC 2013
# OpenCV's function cv::equalizeHist will make it find more faces

    files = [f for f in os.listdir(picdir) if f[-4:]=='.jpg']
    files.sort()
    start_time=time.time()
    for i,f in enumerate(files):
        current_start=time.time()
        full_path = os.path.join(picdir,f)
        if not os.path.isfile(full_path):
            print('not a good file...')
            continue
        print('working on '+full_path)
        info = find_face_dlib_with_scores(full_path,visual_output=False)
        print info
        current_elapsed=time.time()-current_start
        all_elapsed=time.time()-start_time
        print('current {} overall {}'.format(current_elapsed,all_elapsed/(i+1)))


def get_cv2_img_array(url_or_path_to_image_file_or_cv2_image_array, convert_url_to_local_filename=False, download=False,
                      download_directory='images', filename=False, replace_https_with_http=True):
    """
    Get a cv2 img array from a number of different possible inputs.

    :param url_or_path_to_image_file_or_cv2_image_array:
    :param convert_url_to_local_filename:
    :param download:
    :param download_directory:
    :return: img_array
    """
    # print('get:' + str(url_or_path_to_image_file_or_cv2_image_array) + ' try local' + str(
    # convert_url_to_local_filename) + ' download:' + str(download))
    got_locally = False
    img_array = None  # attempt to deal with non-responding url

    # first check if we already have a numpy array
    if isinstance(url_or_path_to_image_file_or_cv2_image_array, np.ndarray):
        img_array = url_or_path_to_image_file_or_cv2_image_array

    # otherwise it's probably a string, check what kind
    elif isinstance(url_or_path_to_image_file_or_cv2_image_array, basestring):
        # try getting url locally by changing url to standard name
        if convert_url_to_local_filename:  # turn url into local filename and try getting it again
            # filename = url_or_path_to_image_file_or_cv2_image_array.split('/')[-1].split('#')[0].split('?')[0]
            # jeremy changed this since it didn't work with url -
            # https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcR2oSMcnwErH1eqf4k8fvn2bAxvSdDSbp6voC7ijYJStL2NfX6v
            # TODO: find a better way to create legal filename from url
            filename = \
                url_or_path_to_image_file_or_cv2_image_array.split('/')[-1].split('#')[0].split('?')[-1].split(':')[
                    -1]
            filename = os.path.join(download_directory, filename)
            if filename.endswith('jpg') or filename.endswith('jpeg') or filename.endswith('.bmp') or \
                    filename.endswith('tiff'):
                pass
            else:  # there's no 'normal' filename ending so add .jpg
                filename = filename + '.jpg'
            # print('trying again locally using filename:' + str(filename))
            img_array = get_cv2_img_array(filename, convert_url_to_local_filename=False, download=download,
                                          download_directory=download_directory)
            # maybe return(get_cv2 etc) instead of img_array =
            if img_array is not None:
                # print('got ok array calling self locally')
                return img_array
            else:  # couldnt get locally so try remotely
                # print('trying again remotely since using local filename didnt work, download=' + str( download) + ' fname:' + str(filename))
                return (
                    get_cv2_img_array(url_or_path_to_image_file_or_cv2_image_array, convert_url_to_local_filename=False,
                                      download=download,
                                      download_directory=download_directory))  # this used to be 'return'
        # put images in local directory
        else:
            # get remotely if its a url, get locally if not
            if "://" in url_or_path_to_image_file_or_cv2_image_array:
                if replace_https_with_http:
                    url_or_path_to_image_file_or_cv2_image_array = url_or_path_to_image_file_or_cv2_image_array.replace(
                        "https", "http")
                img_url = url_or_path_to_image_file_or_cv2_image_array
                try:
                    # print("trying remotely (url) ")
                    headers = {'User-Agent': USER_AGENT}
                    response = requests.get(img_url, headers=headers)  # download
                    img_array = imdecode(np.asarray(bytearray(response.content)), 1)
                except ConnectionError:
                    logging.warning("connection error - check url or connection")
                    return None
                except:
                    logging.warning(" error other than connection error - check something other than connection")
                    return None

            else:  # get locally, since its not a url
                # print("trying locally (not url)")
                img_path = url_or_path_to_image_file_or_cv2_image_array
                try:
                    img_array = cv2.imread(img_path)
                    if img_array is not None:
                        # print("success trying locally (not url)")
                        got_locally = True
                    else:
                        # print('couldnt get locally (in not url branch)')
                        return None
                except:
                    # print("could not read locally, returning None")
                    logging.warning("could not read locally, returning None")
                    return None  # input isn't a basestring nor a np.ndarray....so what is it?
    else:
        logging.warning("input is neither an ndarray nor a string, so I don't know what to do")
        return None

    # After we're done with all the above, this should be true - final check that we're outputting a good array
    if not (isinstance(img_array, np.ndarray) and isinstance(img_array[0][0], np.ndarray)):
        print("Bad image coming into get_cv2_img_array - check url/path/array:" + str(
            url_or_path_to_image_file_or_cv2_image_array) + 'try locally' + str(
            convert_url_to_local_filename) + ' dl:' + str(
            download) + ' dir:' + str(download_directory))
        logging.warning("Bad image - check url/path/array:" + str(
            url_or_path_to_image_file_or_cv2_image_array) + 'try locally' + str(
            convert_url_to_local_filename) + ' dl:' + str(
            download) + ' dir:' + str(download_directory))
        return (None)
    # if we got good image and need to save locally :
    if download:
        if not got_locally:  # only download if we didn't get file locally
            if not os.path.isdir(download_directory):
                os.makedirs(download_directory)
            if "://" in url_or_path_to_image_file_or_cv2_image_array:  # its a url, get the bifnocho
                if replace_https_with_http:
                    url_or_path_to_image_file_or_cv2_image_array = url_or_path_to_image_file_or_cv2_image_array.replace(
                        "https", "http")
                filename = \
                    url_or_path_to_image_file_or_cv2_image_array.split('/')[-1].split('#')[0].split('?')[-1].split(':')[
                        -1]
                filename = os.path.join(download_directory, filename)
            else:  # its not a url so use straight
                filename = os.path.join(download_directory, url_or_path_to_image_file_or_cv2_image_array)
            if filename.endswith('jpg') or filename.endswith('jpeg') or filename.endswith('.bmp') or filename.endswith(
                    'tiff'):
                pass
            else:  # there's no 'normal' filename ending
                filename = filename + '.jpg'
            try:  # write file then open it
                # print('filename for local write:' + str(filename))
                write_status = imwrite(filename, img_array)
                max_i = 50  # wait until file is readable before continuing
                gotfile = False
                for i in xrange(max_i):
                    try:
                        with open(filename, 'rb') as _:
                            gotfile = True
                    except IOError:
                        time.sleep(10)
                if gotfile == False:
                    print('Could not access {} after {} attempts'.format(filename, str(max_i)))
                    raise IOError('Could not access {} after {} attempts'.format(filename, str(max_i)))
            except:  # this is prob unneeded given the 'else' above
                print('unexpected error in Utils calling imwrite')
    return img_array



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

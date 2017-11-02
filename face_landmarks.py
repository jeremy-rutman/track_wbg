__author__ = 'jeremy'
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/jeremy/sw/anaconda2/lib/python2.7/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat')

def get_landmarks(image):
    detections = detector(image, 1)
    landmarks=[]
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
            cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame

        for x, y in zip(xlist, ylist): #Store all landmarks in one list in the format x1,y1,x2,y2,etc.
            landmarks.append(x)
            landmarks.append(y)
    print landmarks
    cv2.imshow("image", image) #Display the frame
    cv2.waitKey(0)

    if len(detections) > 0:
        return landmarks
    else: #If no faces are detected, return error message to other function to handle
        landmarks = "error"
        return landmarks


if __name__ == "__main__":
    img_file = '/data/jeremy/image_dbs/variant/longfeld_video/longfeld025.jpg'
    img_arr=cv2.imread(img_file)

    gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)

    get_landmarks(clahe_image)
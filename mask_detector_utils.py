from turtle import width
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
from utils import *

cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
# cascPatheyes = os.path.dirname(
#     cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"
cascPatheyes = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_eye.xml"
cascMouth = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_mcs_mouth.xml"
cascEyes = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_eye.xml"
cascNose = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_mcs_nose.xml"

data = []

faceCascade = cv2.CascadeClassifier(cascPathface)
mouthCascade = cv2.CascadeClassifier(cascMouth)
eyesCascade = cv2.CascadeClassifier(cascEyes)
noseCascade = cv2.CascadeClassifier(cascNose)
model = load_model("masknet.h5")
glasses_model = load_model("glassesnet_test.h5")

mask_label = {0:'MASK',1:'NO MASK'}
dist_label = {0:(0,255,0),1:(255,0,0)}

def detect(frame):

    face_status_list = []

    brightness_status_message = get_brightness_status_message(frame)

    # histogram equalize 
    img_yuv = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # convert to gray 
    gray = cv2.cvtColor(hist_eq, cv2.COLOR_BGR2GRAY)
    new_img = cv2.cvtColor(hist_eq, cv2.COLOR_RGB2BGR)

    # face detect with haar 
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.3,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:

        face_detection = FaceDetection()
        face_item = FaceItem()

        face_frame = frame[y:y+h,x:x+w]
        roi_gray = gray[y:y+h,x:x+w]

        # convert for detecting glasses
        glasses_face = cv2.resize(face_frame, (160,160))
        glasses_face = np.reshape(glasses_face,[1,160,160,3])

        # convert for detecting mask
        face = cv2.cvtColor(face_frame, cv2.COLOR_RGB2BGR)
        mask_face = cv2.resize(face,(128,128))
        mask_face = np.reshape(mask_face,[1,128,128,3])/255.0

        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (224, 224))
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        face_frame =  preprocess_input(face_frame)

        # mouth detection
        mouth_rects = mouthCascade.detectMultiScale(roi_gray, 1.7, 11)
        mouth_area_list = []
        for (mx,my,mw,mh) in mouth_rects:
            my = int(my - 0.15*mh)
            mouth_area = DetectionArea(Poi(mx+x,my+y), Poi(mx+mw+x,my+mh+y))
            mouth_area_list.append(mouth_area)
            break
        face_detection.set_mouth_area(mouth_area_list)
        
        # nose detection 
        nose_rects = noseCascade.detectMultiScale(roi_gray, 1.3, 5)
        nose_area_list = []
        for (nx,ny,nw,nh) in nose_rects:
            nose_area = DetectionArea(Poi(nx+x,ny+y),Poi(nx+nw+x,ny+nh+y))
            nose_area_list.append(nose_area)
            break
        face_detection.set_nose_area(nose_area_list)
        
        # eyes detection
        eyes_rects = eyesCascade.detectMultiScale(roi_gray)
        eyes_area_list = []
        for (ex,ey,ew,eh) in eyes_rects:
            eye_area = DetectionArea(Poi(ex+x,ey+y),Poi(ex+ew+x,ey+eh+y))
            eyes_area_list.append(eye_area)
        face_detection.set_eyes_area(eyes_area_list)
        
        
        # test
        print("face detecction test : ")
        print("has mouth:" ,face_detection.has_mouth())
        print("has nose :", face_detection.has_nose())
        print("has eyes :", face_detection.has_eyes())

        # glasses detection
        predictions = glasses_model.predict(glasses_face)
        predictions = tf.nn.sigmoid(predictions)
        predictions = tf.where(predictions < 0.5, 0, 1)
        predictions_np_array = predictions.numpy()
        prediction_val = predictions_np_array[0][0]
        prediction_val = prediction_val.item()
        print("prediction_val type : ", type(prediction_val))
        has_glasses = prediction_val == 0
        print("has glasses type : ", type(has_glasses))
        face_item.set_glasses(has_glasses)
        
        # mask detection
        mask_result = model.predict(mask_face)
        has_mask = mask_result.argmax() == 0
        face_item.set_mask(has_mask)

        print("face item test : ")
        print("glasses status :", face_item.has_glasses_str())
        print("mask status :", face_item.has_mask_str())

        face_status = FaceStatus(face_detection, face_item)
        face_status_list.append(face_status)
        face_status.set_brightness_status(brightness_status_message)

    return face_status_list
        

def get_brightness_status_message(frame):
    bright_thres = 0.65
    dark_thres = 0.65
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("[INFO] performing histogram equalization...")
    equalized = cv2.equalizeHist(gray)
    dark_part = cv2.inRange(gray, 0, 30)
    bright_part = cv2.inRange(gray, 220, 255)
    total_pixel = np.size(gray)
    dark_pixel = np.sum(dark_part > 0)
    bright_pixel = np.sum(bright_part > 0)
    found = False
    if dark_pixel/total_pixel > bright_thres:
        ans = -1
        found = True
    if bright_pixel/total_pixel > dark_thres:
        ans = 1
        found = True
    if not found:
        ans = 0
    return ans
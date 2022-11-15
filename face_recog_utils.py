from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
from PIL import Image
import tensorflow.compat.v1 as tf
from utils import *
from load_model import faceNet, confidence_threshold


modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
train_img="./train_img"

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        image_size = 182
        input_image_size = 160
        HumanNames = os.listdir(train_img)
        HumanNames.sort()
        print('Loading Model')
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile,encoding='latin1')
        

# def recog_face(frame):
#     face_list = []
#     if frame.ndim == 2:
#         frame = facenet.to_rgb(frame)
    
#     img_yuv = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV)
#     img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
#     hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

#     frame = hist_eq

#     bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
#     faceNum = bounding_boxes.shape[0]
#     if faceNum > 0:
#         print('have a face')
#         det = bounding_boxes[:, 0:4]
#         img_size = np.asarray(frame.shape)[0:2]
#         cropped = []
#         scaled = []
#         scaled_reshape = []
#         for i in range(faceNum):
#             emb_array = np.zeros((1, embedding_size))
#             xmin = int(det[i][0])
#             ymin = int(det[i][1])
#             xmax = int(det[i][2])
#             ymax = int(det[i][3])
#             try:
#                 # inner exception
#                 if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
#                     print('Face is very close!')
#                     continue
#                 cropped.append(frame[ymin:ymax, xmin:xmax,:])
#                 cropped[i] = facenet.flip(cropped[i], False)
#                 scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
#                 scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
#                                         interpolation=cv2.INTER_CUBIC)
#                 scaled[i] = facenet.prewhiten(scaled[i])
#                 scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
#                 feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
#                 emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
#                 predictions = model.predict_proba(emb_array)
#                 best_class_indices = np.argmax(predictions, axis=1)
#                 best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
#                 if best_class_probabilities>0.5:
#                     # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)    #boxing face
#                     for H_i in HumanNames:
#                         if HumanNames[best_class_indices[0]] == H_i:
#                             # result_names = HumanNames[best_class_indices[0]]
#                             print("Predictions : [ name: {} , accuracy: {:.3f} ]".format(HumanNames[best_class_indices[0]],best_class_probabilities[0]))
#                             id = HumanNames[best_class_indices[0]]
#                             accuracy = best_class_probabilities[0]
#                             current_face = FaceRecognize(id, id, accuracy, DetectionArea(Poi(xmin, ymin-20), Poi(xmax, ymin-2)))
#                             face_list.append(current_face)
#                             # cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
#                             # cv2.putText(frame, result_names, (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
#                             #             1, (0, 0, 0), thickness=1, lineType=1)
                            
#                 else :
#                     print('unknow')
#                     id = 'unknow'
#                     accuracy = 0
#                     current_face = FaceRecognize(id ,id , accuracy, DetectionArea(Poi(xmin, ymin-20), Poi(xmax, ymin-2)))
#                     face_list.append(current_face)
#                     # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#                     # cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
#                     # cv2.putText(frame, "?", (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
#                                         # 1, (0, 0, 0), thickness=1, lineType=1)
#             except:   
#                 print("error")
#     return face_list


def recog_face(frame):
    face_list = []
    if frame.ndim == 2:
        frame = facenet.to_rgb(frame)
    
    img_size = np.asarray(frame.shape)[0:2]
    cropped = []
    scaled = []
    scaled_reshape = []

    (hh, ww) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            print('has face')
            box = detections[0, 0, i, 3:7] * np.array([ww, hh, ww, hh])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(ww - 1, endX), min(hh - 1, endY))

            if endX < startX or endY < startY: continue

            xmin = startX
            ymin = startY
            xmax = endX
            ymax = endY

            emb_array = np.zeros((1, embedding_size))

            try:
                # inner exception
                if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                    print('Face is very close!')
                    continue
                cropped.append(frame[ymin:ymax, xmin:xmax,:])
                cropped[i] = facenet.flip(cropped[i], False)
                scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
                scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                        interpolation=cv2.INTER_CUBIC)
                scaled[i] = facenet.prewhiten(scaled[i])
                scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                if best_class_probabilities>0.5:
                    for H_i in HumanNames:
                        if HumanNames[best_class_indices[0]] == H_i:
                            print("Predictions : [ name: {} , accuracy: {:.3f} ]".format(HumanNames[best_class_indices[0]],best_class_probabilities[0]))
                            id = HumanNames[best_class_indices[0]]
                            accuracy = best_class_probabilities[0]
                            current_face = FaceRecognize(id, id, accuracy, DetectionArea(Poi(xmin, ymin-20), Poi(xmax, ymin-2)))
                            face_list.append(current_face)
                            
                else :
                    print('unknow')
                    id = 'unknow'
                    accuracy = 0
                    current_face = FaceRecognize(id ,id , accuracy, DetectionArea(Poi(xmin, ymin-20), Poi(xmax, ymin-2)))
                    face_list.append(current_face)
            except:   
                print("error")
            
            break
    return face_list

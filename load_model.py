import cv2

print("[INFO] loading face detector model...")
prototxtPath = "face_detector/deploy.prototxt.txt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
confidence_threshold = 0.5
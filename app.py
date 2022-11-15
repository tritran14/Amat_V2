from email.mime import base, image
import json
from flask import Flask, redirect, flash, request, jsonify
from flask_cors import CORS, cross_origin
from mask_detector_utils import FaceStatus, detect
import numpy as np
import cv2 as cv2
from face_recog_utils import recog_face
import base64
import operator
import save_file

from utils import ResponseData, UserIdentity


UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = 'super secret key'

cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

################################### HELPER   #############################

totImg = 10
freqAccuracyThreshold = 0.8
totRegisterImg = 5
maskThreshold = 0.5
glassesThreshold = 0.5
eyesThreshold = 0.5
mouthThreshold = 0.5
noseThreshold = 0.5


def convertBase64ToCV2Image(im_b64):
    im_bytes = base64.b64decode(im_b64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img

def recogFace(base64List):
    userIdentity = UserIdentity('-1', 'Unknown', 0.0)
    faceCnt = 0
    dict = {}
    accuracyDict = {}
    for base64 in base64List:
        value = base64['value']
        if value is None: continue
        img = convertBase64ToCV2Image(value)
        face = recog_face(img)
        if len(face) > 0:
            face = face[0]
        else:
            face = None
        if face is None: continue
        else:
            id = face.id
            faceCnt += 1
            if id in dict:
                dict[id]+=1
                accuracyDict[id]+=face.accuracy
            else:
                dict[id]=1
                accuracyDict[id]=face.accuracy
    if dict:
        candidate = max(dict.items(), key=operator.itemgetter(1))
        candidateId = candidate[0]
        candidateCnt = candidate[1]
        candidateAccuracy = accuracyDict[candidateId] / candidateCnt
        if candidateCnt / faceCnt >= freqAccuracyThreshold:
            userIdentity = UserIdentity(candidateId, candidateId, candidateAccuracy)
        print("candidateId : {}, cnt : {}, avg accuracy : {}".format(candidateId, candidateCnt, candidateAccuracy))
        print(json.dumps(dict))
    else:
        print("don't have face")
    return userIdentity

def getResponseData(base64List):
    removeMessagePattern = "Pls remove {} on you face"
    cantDetectMessagePattern = "I can't see your {}"
    message = ""
    goodMessage = "Naice face"

    maskMessage = "mask"
    glassesMessage = "glasses"
    eyesMessage = "eyes"
    mouthMessage = "mouth"
    noseMessage = "nose"

    faceStatusList = []
    cnt = 0
    print('start with ', len(base64List))

    maskCnt = 0
    glassesCnt = 0
    eyesCnt = 0
    mouthCnt = 0
    noseCnt = 0

    validImage = []

    for base64 in base64List:
        value = base64['value']
        if value is None: continue
        img = convertBase64ToCV2Image(value)
        faceStatus = detect(img)
        print('after : .....')
        print('len : ',len(faceStatus))
        faceStatusList.append(faceStatus)
        if len(faceStatus) > 0:
            cnt += 1
            maskCnt += faceStatus[0].face_item.mask == True
            glassesCnt += faceStatus[0].face_item.glasses == True
            eyesCnt += faceStatus[0].face_detection.has_eyes() == True
            mouthCnt += faceStatus[0].face_detection.has_mouth() == True
            noseCnt += faceStatus[0].face_detection.has_nose() == True
            if faceStatus[0].good_face():
                validImage.append(base64)

    maskRatio = maskCnt / cnt
    glassesRatio = glassesCnt / cnt
    eyesRatio = eyesCnt / cnt
    mouthRatio = mouthCnt / cnt
    noseRatio = noseCnt / cnt

    feature = []
    if eyesRatio < eyesThreshold:
        feature.append(eyesMessage)
    if mouthRatio < mouthThreshold:
        feature.append(mouthMessage)
    if noseRatio < noseThreshold:
        feature.append(noseMessage)
    
    item = []

    if maskRatio >= maskThreshold:
        item.append(maskMessage)
    if glassesRatio >= glassesThreshold:
        item.append(glassesMessage)
    
    found = False

    if len(feature) > 0:
        delimiter = ""
        if(len(message) > 0): 
            delimiter = " and "
        message = message + delimiter + cantDetectMessagePattern.format(', '.join(feature))
        found = True

    if len(item) > 0:
        delimiter = ""
        if(len(message) > 0): 
            delimiter = " and "
        message = message + delimiter + removeMessagePattern.format(', '.join(item))
        found = True

    if not found:
        message = goodMessage

    print('cnt : ', cnt)
    print("mask cnt : {}".format(maskCnt))
    print("glasses cnt : {}".format(glassesCnt))
    print("message : {}".format(message))
    return ResponseData(validImage, message)

def train_data():
    import data_proccess
    import train_main

########################################################################################################

@app.route('/')
@cross_origin()
def index():
    return "hello world"

@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_file_endpoint():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        filestr = file.read()
        file_bytes = np.fromstring(filestr, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        face_status_list = detect(img)
        face_status_dict = [dict(x) for x in face_status_list]
        print("dict : ")
        print(face_status_dict)
        return jsonify(face_status_dict), 200

@app.route('/recog_face', methods=['POST'])
@cross_origin()
def recog_face_endpoint():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        filestr = file.read()
        file_bytes = np.fromstring(filestr, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        face_list = recog_face(img)
        face_list_dict = [dict(x) for x in face_list]
        return jsonify(face_list_dict), 200

@app.route('/login', methods=['POST'])
@cross_origin()
def login():
    if request.method == 'POST':
        if request.data is not None:
            base64List = json.loads(request.data)
            faceInfo = recogFace(base64List)
            faceInfoDict = dict(faceInfo)
            return jsonify(faceInfoDict), 200
    return "Bad request", 400

@app.route('/register', methods=['POST'])
@cross_origin()
def register():
    if request.method == 'POST':
        if request.data is not None:
            base64List = json.loads(request.data)
            data = getResponseData(base64List)
            dataDict = dict(data)
            return jsonify(dataDict), 200
    return "Bad request", 400

@app.route('/upload-face', methods=['POST'])
@cross_origin()
def upload_face():
    if request.method == 'POST':
        if request.data is not None:
            userImage = json.loads(request.data)
            userId = userImage["username"]
            base64List = userImage["base64List"]
            print(userId)
            # save_file.create_dir(userId, base64List)
            # train_data()
            
            return {"hello":"world"}, 200
    return "Bad request", 400

app.run(debug=True, host='0.0.0.0', port=5050)
from email.mime import image
import json
from flask import Flask, redirect, flash, request, jsonify
from flask_cors import CORS, cross_origin
from mask_detector_utils import FaceStatus, detect
import numpy as np
import cv2 as cv2
from face_recog_utils import recog_face
import base64
import operator


UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = 'super secret key'

cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

################################### HELPER   #############################

def convertBase64ToCV2Image(im_b64):
    im_bytes = base64.b64decode(im_b64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img

def recogFace(base64List):
    dict = {}
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
            if id in dict:
                dict[id]+=1
            else:
                dict[id]=1
    candidate = max(dict.items(), key=operator.itemgetter(1))
    candidateId = candidate[0]
    candidateCnt = candidate[1]
    print("candidateId : {}, accuracy : {}".format(candidateId, candidateCnt))
    print(json.dumps(dict))

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
            recogFace(base64List)
        return 'login from ai', 200

app.run(debug=True, host='0.0.0.0', port=5050)
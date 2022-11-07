from email.mime import image
import json
from flask import Flask, redirect, flash, request, jsonify
from mask_detector_utils import FaceStatus, detect
import numpy as np
import cv2 as cv2
from face_recog_utils import recog_face

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = 'super secret key'

@app.route('/')
def index():
    return "hello world"

@app.route('/upload', methods=['POST'])
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


app.run(debug=True, host='0.0.0.0', port=5050)
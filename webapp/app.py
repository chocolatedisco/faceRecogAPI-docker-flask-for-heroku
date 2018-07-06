# -*- coding: utf-8 -*-
import cv2
import tensorflow
import numpy as np
import keras
from os import path
from flask import Flask, jsonify, request
from keras.models import load_model
import base64
import io
from PIL import Image

app = Flask(__name__)

@app.route('/')
def hello():
    return cv2.__version__

@app.route('/keras')
def ker():
    return keras.__version__

@app.route('/tf')
def tf():
    return tensorflow.__version__

@app.route("/readf", methods = ['POST'])
def readf():
    try:
        img_b64 = request.form['sample']
        img_binary = base64.decodestring(img_b64)
        img_binarystream = io.BytesIO(img_binary)
        img_pil = Image.open(img_binarystream)
        img_numpy = np.asarray(img_pil)
        img = cv2.cvtColor(img_numpy, cv2.COLOR_RGBA2BGR)
        img_gray = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
        face_list=cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))
        #顔が１つ以上検出された時
        if len(face_list) == 1:
            for rect in face_list:
                x,y,width,height=rect
                img = img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
                if img.shape[0]<64:
                    continue
                img = cv2.resize(img,(64,64))
                img_only_face = img
                pilImg = Image.fromarray(np.uint8(img_only_face))
                buffered = io.BytesIO()
                pilImg.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue())
                b,g,r = cv2.split(img)
                img = cv2.merge([r,g,b])
                img = np.array(img)
                img_array = []
                img_array.append(img)
                img_input=np.array(img_array)
                model = load_model("./FaceRecog.h5")
                ans = model.predict(img_input)
                lis = ["齋藤飛鳥","生田絵梨花","西野七瀬","白石麻衣"]
                iter = np.argmax(ans, axis = None, out = None)
                disp = lis[iter]
            return jsonify({'name' : disp, 'iof': img_str, 'err': '0', 'errme': 'none'})
                #顔が検出されなかった時
        elif len(face_list) > 1:
            return jsonify({'name' : 'none', 'iof': 'none', 'err': '1', 'errme': '認識できる顔が多すぎます'})
        else:
            return jsonify({'name' : 'none', 'iof': 'none', 'err': '1', 'errme': '顔が認識できません'})
    except:
        return jsonify({'name' : 'none', 'iof': 'none', 'err': '1', 'errme': '不明なエラーです（画質が粗い、拡張子がjpgでないなどが考えられます）'})


if __name__ == '__main__':
    app.run(host='0.0.0.0')

#curl -X GET https://protected-lowlands-35272.herokuapp.com/ルート　でアクセス
#curl -X POST -d "hoge=0" http://www.example.com/ルート　でデータhoge=0でPOST

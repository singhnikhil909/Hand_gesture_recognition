from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
import re
import sys 
import os
import base64
import cv2


sys.path.append(os.path.abspath("./model"))
from load import * 


global  model

model = init()

app = Flask(__name__)


@app.route('/')
def home():   
    return render_template('index.html')

bg = None
@app.route('/predict',methods=['GET'])
def predict():
# separate the foreground from background
    def run_avg(image, accumweight):
        global bg
        if bg is None:
            bg = image.copy().astype("float")                              # convert an image to float array
            return
        cv2.accumulateWeighted(image, bg, accumweight)                     # accumweight -- weight of th image, decides the speed of updating


# segment the hand region from the video sequence
    def segmented(image, threshold=25):
        global bg
        difference = cv2.absdiff(bg.astype("uint8"), image)                                        # absolute difference between background and image.
        
        thresholded = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)[1]              # reveals only the hand region

        (cnts, _) = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) == 0:
            return
        else:
            segment = max(cnts, key=cv2.contourArea)                                              # get the maximum contours -- i.e. the hand in this case
            return thresholded, segment



    def getPredictedClass(model):
        image = cv2.imread("Temp.png")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, (50, 50))

        gray_image = gray_image.reshape(-1, 50, 50, 1)
        prediction = model.predict_on_batch(gray_image)

        predicted_class = np.argmax(prediction)
        if predicted_class == 0:
            return "PALM"
        if predicted_class == 1:
            return "L"
        if predicted_class == 2:
            return "FIST"
        if predicted_class == 3:
            return "FIST MOVED"
        if predicted_class == 4:
            return "THUMB"
        if predicted_class == 5:
            return "INDEX"
        if predicted_class == 6:
            return "OKAY"
        if predicted_class == 7:
            return "PALM MOVED"
        if predicted_class == 8:
            return "C"
        if predicted_class == 9:
            return "DOWN"
        
  
    accumweight = 0.5
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 10, 350, 255, 590
    num_frames = 0
    model = load_model("model.h5")

    while True:
        ret, frame = camera.read()
        frame = cv2.resize(frame, (900, 900))
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        height, width = frame.shape[:2]
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
            
        if num_frames < 30:
            run_avg(gray, accumweight)
            if num_frames == 1:
                print("[STATUS] PLEASE WAIT CALIBRATING...")
            elif num_frames == 29:
                print("[STATUS] SUCCESSFUL... CONTINUE...")
        else:
            hand = segmented(gray)
            if hand is not None:
                thresholded, segment = hand
                cv2.drawContours(clone, [segment + (right, top)], -1, (0, 0, 255))
                        
                cv2.imwrite("Temp.png", thresholded)
                predictedClass = getPredictedClass(model)
                cv2.putText(clone, str(predictedClass), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow("Thresholded image", thresholded)

        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
        num_frames = num_frames + 1

        cv2.imshow("VIDEO", clone)

        keypress = cv2.waitKey(1)
        if keypress == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
                        
    
    return render_template('index.html', prediction_text = "HAND GESTURE MADE BY THE USER : {}".format(predictedClass))

if __name__ == '__main__':
    app.run(debug=True, port=8000)
    
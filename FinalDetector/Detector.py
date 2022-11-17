# Import the required libraries for the detection 
import cv2
import numpy as np
from fastai.vision.all import *
from fastai.vision import *
import os
import face_recognition
import glob
import pyttsx3 # Import the required module for text to speech conversion
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


# Test is a directory where captured image will be saved to be classified after
Test_Dinar= os.path.join("Test/")
# loading the model ( a pretrained model cosumed on our data )
# The classifier for classifing 4 type of pices (2000 da, 10000 da, 500 da, 20 da)
model = load_learner('Dinar_classifier1.pkl')
engine= pyttsx3.init()
engine.setProperty("rate", 180)
engine.say("Détection de la monnaie a commencé maintenant")
engine.runAndWait()
engine.say("s'il vous plait tennez le biellet un peu proche")
engine.runAndWait()
process= True

while process == True:
    #  capturing image, saving it on test directory, classifing the image 
    camera = cv2.VideoCapture(0)
    while True:
        voice=""
        ret, image = camera.read()
        #cv2.imshow('test', image)
        imgname = os.path.join(Test_Dinar, 'test.jpg')
            # Write out anchor image
        cv2.imwrite(imgname, image)
        break
    image=f'{Test_Dinar}/test.jpg'
    probabilities = model.predict(item=image)
    # Print the classe that has the highest value probabilitie 
    print("Prediction-**********************************")
    print(probabilities[0])
    voice= str(probabilities[0])

    engine.say (" "+voice+" dinar algérien a été détecté")
    engine.runAndWait()
    engine.say (" Si vous voulez véréfie une autre pièce tappez y, sinon n pour terminer la détection")
    engine.runAndWait()
    response = input ("y/n")
    if(response == "n"):
        engine.say ("La réponse est non. Processus de détection est terminé avec succès")
        engine.runAndWait()
        process= False
    else:
        engine.say ("La réponse est oui. continué la détection")
        engine.runAndWait()
        engine.say("s'il vous plait tennez le biellet un peu proche")
        engine.runAndWait()

    
import sys
sys.path.insert(0,'/home/pi/Zumi_Content/Data/face-recognition')
from recognition import Recognition
from zumi.util.camera import Camera
from zumi.util.screen import Screen
import time
import cv2
import IPython.display
import PIL.Image
import numpy as np
import os


fd = Recognition()
camera = Camera(auto_start=False)
screen = Screen()

def collectPictures():
    camera.start_camera()
    fd.name = input("input name : ")
    print("Capture 50 pictures")
    time.sleep(1)

    while True:
        try:
            image = camera.capture()
            fd.makeDataset(image)
            IPython.display.display(PIL.Image.fromarray(image))
            print("Progress : " + str(fd.cap) + "/50")
            screen.draw_image(PIL.Image.fromarray(fd.streaming_image).convert('1'))
            IPython.display.clear_output(wait=True)

            if fd.cap > 50:
                screen.draw_text_center("Done!")
                camera.close()
                break
        except Exception as e:
            print(e)
            screen.draw_text("")
            camera.close()
            break
    
    fd.cap = 0
    time.sleep(1)
    screen.draw_text("")
    camera.close()

def runModel():
    camera.start_camera()
    print("loading...")
    fd.recognizer.read('../Data/face-recognition/trainer/trainer.yml')
    print("start")

    currentTime = time.time()

    while time.time() - currentTime < 30:
        try:
            image = camera.capture()
            fd.recognition(image)
            IPython.display.display(PIL.Image.fromarray(image))
            screen.draw_image(PIL.Image.fromarray(fd.streaming_image).convert('1'))
            IPython.display.clear_output(wait=True)
        except Exception as e:
            print(e)
            screen.draw_text("")
            camera.close()
            break

def trainModel():    
    print("Training model...")
    fd.trainModel()
    print("Done!") 
    
def deleteOneDataset():
    import pickle

    labels = 0    
    try:
        with open('../Data/face-recognition/labels.pickle', 'rb') as labelFile:
            labels = pickle.load(labelFile)
        names = list(labels.values())
        print(names)
        name = input("Please enter a name: ")
        print("?")
        fd.deleteDataset(name)
        print("??")
    except:
        print("[Error] No dataset")      
        
def deleteAllDatasets():
    fd.deleteAllDatasets()
   
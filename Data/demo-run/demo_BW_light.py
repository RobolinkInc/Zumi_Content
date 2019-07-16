import sys
sys.path.insert(0,'/home/pi/Zumi_Content/Data/knn-classifier')
from zumi.zumi import Zumi
from zumi.util.camera import Camera
from color_classifier import ColorClassifier
from threading import Thread


is_white = False
zumi = Zumi()

def continue_straight():
    global is_white
    global zumi
    
    while is_white:
        zumi.go_straight(5, 0)
    zumi.stop(0)

def run():
    global is_white
    global zumi

    try:
        camera = Camera()
        knn = ColorClassifier(path='/home/pi/Zumi_Content/Data')      
        knn.load_model("demo_BW_light")
        camera.start_camera()
 
        while True:
            if input("Press Enter to read a card, or type q first to exit. ") == "q":
                break
            image = camera.capture()
            predict = knn.predict(image, 'v')
            print(predict)

            if predict == "white":
                is_white = True
                drive_thread = Thread(target=continue_straight)
                drive_thread.start()
            if predict == "black":
                is_white = False
                zumi.stop()
    finally:
        is_white = False
        zumi.stop()
        camera.close()            

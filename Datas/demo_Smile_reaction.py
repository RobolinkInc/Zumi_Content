from zumi.zumi import Zumi
from zumi.util.camera import Camera
from zumi.util.screen import Screen
from zumi.personality import Personality
from zumi.util.image_processor import FaceDetector
import time


def run():
    zumi = Zumi()
    camera = Camera()
    screen = Screen()
    personality = Personality(zumi, screen)
    faceDetector = FaceDetector() 

    camera.start_camera()

    try:    
        while True:
            image = camera.capture()
            face = faceDetector.detect_face(image)
            if face:
                face_roi = faceDetector.get_face_roi()
                smile = faceDetector.detect_smile(face_roi)
                if smile:
                    personality.happy()
                else:
                    personality.angry()
            else:
                screen.close_eyes()
    finally:
        camera.close()
        screen.draw_text("")
import numpy as np
from zumi.util.screen import Screen
import pyzbar.pyzbar as pyzbar
import IPython.display
from PIL import Image
import cv2


class Vision:

    def __init__(self):
        face_xml_path = "/usr/local/lib/python3.5/dist-packages/zumi/util/src/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(face_xml_path)
        
    def screen_display_image(self,frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert it to gray
        small = cv2.resize(gray, (128,64)) # Resize it to fit the screen
        screen = Screen(clear=False)
        screen.draw_image(Image.fromarray(small).convert('1')) # show the picture! 
    
    def show_image(self,frame):
        IPython.display.display(Image.fromarray(frame))
        
    def convert_to_gray(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert it to gray\
        return gray
        

    def clear_output(self):
        IPython.display.clear_output(wait=True)

    def find_face(self,frame, scale=1.05, neighbors=8, minimumSize=(40,40)):
        # face_xml_path = "/usr/local/lib/python3.5/dist-packages/zumi/util/src/haarcascade_frontalface_default.xml"
        # face_cascade = cv2.CascadeClassifier(face_xml_path)

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        signs = self.face_cascade.detectMultiScale(frame, scaleFactor=scale, minNeighbors=neighbors, minSize=minimumSize, flags=cv2.CASCADE_SCALE_IMAGE)
        if len(signs)>0:
            for (x,y,w,h) in signs:
                #draw a bounding box on the stop sign found
                #x is x positon in frame, y is 
                #w is width of area enclosing stop sign and h is height
                color_of_box = (0,0,255) # this is in (R,G,B) 8 bit 2^8=256
                cv2.rectangle(frame, (x,y), (x+w, y+h), color_of_box, 2)

    #will find the largest QR code and return its object
    def find_QR_code(self,frame,draw_color=(255, 85, 255),draw_msg=True, draw_rect=True):
        decodedObjects = pyzbar.decode(frame)
        if len(decodedObjects) > 0:
            if draw_msg == True:
                obj = decodedObjects[0]
                data = obj.data.decode("utf-8")
                cv2.putText(frame,data, (20, 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, draw_color, 2)
            if draw_rect == True:
                obj = decodedObjects[0]
                p1,p2,p3,p4 = obj.polygon
                cv2.line(frame,p1,p2,draw_color,2)
                cv2.line(frame,p2,p3,draw_color,2)
                cv2.line(frame,p4,p3,draw_color,2)
                cv2.line(frame,p4,p1,draw_color,2)
            return decodedObjects[0]
        else:
            return None

    #returns only the message in the qr object
    def get_QR_message(self,QR_object):
        if QR_object is not None: # If the code finds more than one code...
            obj = QR_object
            #print("Found ", obj.type) # Print the type of code (barcode or QR code)
            data = obj.data.decode("utf-8") # Decode the message
            #print("Message: ", data) # Print the message
            return data
        else:
            return None
    #returns the coordinate in the qr object
    def get_QR_center(self,QR_object):
        if QR_object is not None:
            x,y,w,h = QR_object.rect
            return x,y
        else:
            return None

    def get_QR_dimensions(self,QR_object):
        if QR_object is not None:
            obj = QR_object
            x,y,w,h = obj.rect
            return w,h
        else:
            return None

    def get_QR_polygon(self,QR_object):
        if QR_object is not None:
            p1,p2,p3,p4 = QR_object.polygon
            return p1,p2,p3,p4
        else:
            return None

    def warp_frame(self,frame,w_ratio = 0.4,h_ratio = 0.6):
        height,width,channels = frame.shape
        #        location of the coordinates
        #--------------------------------------------\
        #        top_left          top_right         \
        #                                            \
        #                                            \
        #                                            \
        #bottom_left                     bottom_right\
        #--------------------------------------------\

        top_left =  [int(width*w_ratio),        int(height*h_ratio)]
        top_right = [int(width - width*w_ratio),int(height*h_ratio)]

        bottom_left = [0, height]
        bottom_right = [width, height]

        pts1 = np.float32([top_left, top_right, bottom_left, bottom_right])
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(frame, matrix, (int(width), int(height)))
        return result

    def rotate_frame(self,frame,angle):
        height,width,channels = frame.shape
        #does some fancy math to rotate the image depending on the angle
        M = cv2.getRotationMatrix2D((width/2,height/2),angle,1)
        result = cv2.warpAffine(frame,M,(width,height))
        return result

    def track_this_hue(self,image,color):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        filteredFrame = cv2.inRange(hsv, color[0], color[1])
        colorCutout =  cv2.bitwise_and(image, image, mask=filteredFrame)
        return colorCutout, filteredFrame

    def find_blue_object(self,frame,h_range =10,s_range=65,v_range=65,draw_color=(255, 85, 255)):
        colorCutout, filteredFrame = self.blue_filter(frame,h_range ,s_range,v_range)
        contoursArray = cv2.findContours(filteredFrame.copy(), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(contoursArray) > 0:
            #only return one contour
            contour = contoursArray[-1]
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
            return x, y, w, h
        else:
            return None

    def find_green_object(self,frame,h_range =15,s_range=65,v_range=65,draw_color=(255, 85, 255)):
        colorCutout, filteredFrame = self.green_filter(frame,h_range ,s_range,v_range)
        contoursArray = cv2.findContours(filteredFrame.copy(), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(contoursArray) > 0:
            #only return one contour
            contour = contoursArray[-1]
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
            return x, y, w, h
        else:
            return None

    def find_yellow_object(self,frame,h_range =5,s_range=53,v_range=68,draw_color=(255, 85, 255)):
        colorCutout, filteredFrame = self.yellow_filter(frame,h_range ,s_range,v_range)
        contoursArray = cv2.findContours(filteredFrame.copy(), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(contoursArray) > 0:
            #only return one contour
            contour = contoursArray[-1]
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
            return x, y, w, h
        else:
            return None

    def find_orange_object(self,frame,h_range =5,s_range=53,v_range=68,draw_color=(255, 85, 255)):
        colorCutout, filteredFrame = self.orange_filter(frame,h_range ,s_range,v_range)
        contoursArray = cv2.findContours(filteredFrame.copy(), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(contoursArray) > 0:
            #only return one contour
            contour = contoursArray[-1]
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
            return x, y, w, h
        else:
            return None

    def find_red_object(self,frame,h_range =6,s_range=50,v_range=70,draw_color=(255, 85, 255)):
        colorCutout, filteredFrame = self.red_filter(frame,h_range ,s_range,v_range)
        contoursArray = cv2.findContours(filteredFrame.copy(), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(contoursArray) > 0:
            #only return one contour
            contour = contoursArray[-1]
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
            return x, y, w, h
        else:
            return None


    def find_purple_object(self,frame,h_range =5,s_range=113,v_range=98,draw_color=(255, 85, 255)):
        colorCutout, filteredFrame = self.red_filter(frame,h_range ,s_range,v_range)
        contoursArray = cv2.findContours(filteredFrame.copy(), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(contoursArray) > 0:
            #only return one contour
            contour = contoursArray[-1]
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
            return x, y, w, h
        else:
            return None

    # blueLower = (20, 70, 70)
    # blueUpper = (40, 200, 200)
    # blue = [blueLower,blueUpper]

    # greenLower = (40, 70, 70)
    # greenUpper = (70, 200, 200)
    # green = [greenLower,greenUpper]

    # yellowLower = (90, 150, 120)
    # yellowUpper = (100, 255, 255)
    # yellow = [yellowLower,yellowUpper]

    # orangeLower = (100, 150, 120)
    # orangeUpper = (110, 255, 255)
    # orange = [orangeLower,orangeUpper]

    # redLower = (110, 100, 100)
    # redUpper = (130, 255, 255)
    # red = [redLower,redUpper]

    # purpleLower = (140, 30, 60)
    # purpleUpper = (170, 255, 255)
    # purple = [purpleLower,purpleUpper]

    def blue_filter(self,image,h_range =10,s_range=65,v_range=65):
        hue = 30
        sat = 135
        val = 135
        blueLower = (hue-h_range, sat-s_range,val-v_range)
        blueUpper = (hue+h_range, sat+s_range,val+v_range)
        blue = [blueLower,blueUpper]
        return self.track_this_hue(image,blue)

    def green_filter(self,image,h_range =15,s_range=65,v_range=65):
        hue = 55
        sat = 135
        val = 135
        greenLower = (hue-h_range, sat-s_range,val-v_range)
        greenUpper = (hue+h_range, sat+s_range,val+v_range)
        green = [greenLower,greenUpper]
        return self.track_this_hue(image,green)

    def yellow_filter(self,image,h_range =5,s_range=53,v_range=68):
        hue = 95
        sat = 202
        val = 187
        yellowLower = (hue-h_range, sat-s_range,val-v_range)
        yellowUpper = (hue+h_range, sat+s_range,val+v_range)
        yellow = [yellowLower,yellowUpper]
        return self.track_this_hue(image,yellow)

    def orange_filter(self,image,h_range =5,s_range=53,v_range=68):
        hue = 105
        sat = 202
        val = 187
        orangeLower = (hue-h_range, sat-s_range,val-v_range)
        orangeUpper = (hue+h_range, sat+s_range,val+v_range)
        orange = [orangeLower,orangeUpper]
        return track_this_hue(image,orange)

    def red_filter(self,image,h_range =6,s_range=50,v_range=70):
        hue = 120
        sat = 177
        val = 177
        redLower = (hue-h_range, sat-s_range,val-v_range)
        redUpper = (hue+h_range, sat+s_range,val+v_range)
        red = [redLower,redUpper]
        return self.track_this_hue(image,red)

    def purple_filter(self,image,h_range =5,s_range=113,v_range=98):
        hue = 155
        sat = 165
        val = 158
        purpleUpper = (hue-h_range, sat-s_range,val-v_range)
        purpleUpper = (hue+h_range, sat+s_range,val+v_range)
        purple = [greenLower,greenUpper]
        return self.track_this_hue(image,purple)

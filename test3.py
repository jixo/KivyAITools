from __future__ import print_function
from kivy.core.window import Window
#Window.size = (450, 450)

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.gridlayout import GridLayout
from kivy.uix.camera import Camera
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.properties import ObjectProperty
from database import DataBase

import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


from object_recognition_detection.utils import label_map_util

from object_recognition_detection.utils import visualization_utils as vis_util

from PIL import Image
import requests
import os
import numpy as np
import cv2
import argparse
import sys
import os.path
import math
import time

from random import randint
##def fun jix

def invalidLogin():
    content = Button(text='Invalid Email or Password')
    popup = Popup(title='Please try again !!', content=content, auto_dismiss=True, size_hint=(0.5, 0.5))
    content.bind(on_press=popup.dismiss)
    popup.open()

def invalidForm():
    content = Button(text='Please fill in all inputs')
    popup = Popup(title='Invalid Form', content=content, auto_dismiss=True, size_hint=(0.5, 0.5))
    content.bind(on_press=popup.dismiss)
    popup.open()



class MainWindow(Screen):
    email = ObjectProperty(None)
    password = ObjectProperty(None)
    def verify_credential(self):
        # if self.ids["email"].text == "jix@hot.com" and self.ids["password"].text == "jix":
        #     self.manager.current = "second"
        if db.validate(self.email.text, self.password.text):
            SecondWindow.current = self.email.text
            self.reset()
            self.manager.current = "second"
        else:
            invalidLogin()

    def createBtn(self):
        self.reset()
        self.manager.current = "create"

    def reset(self):
        self.email.text = ""
        self.password.text = ""

class CreateAccountWindow(Screen):
    namee = ObjectProperty(None)
    email = ObjectProperty(None)
    password = ObjectProperty(None)

    def submit(self):
        if self.namee.text != "" and self.email.text != "" and self.email.text.count("@") == 1 and self.email.text.count(".") > 0:
            if self.password != "":
                db.add_user(self.email.text, self.password.text, self.namee.text)
                self.reset()
                self.manager.current = "main"
            else:
                invalidForm()
        else:
            invalidForm()
    def login(self):
        self.reset()
        self.manager.current = "main"
    def reset(self):
        self.email.text = ""
        self.password.text = ""
        self.namee.text = ""

class SecondWindow(Screen):
    pass


class ThirdWindow(Screen):

    def FuncBut(self):
        self.start_button = self.ids['start_button']
        self.start_button.disabled = True  # Prevents the user from clicking start again which may crash the program
        self.rec_button = self.ids['rec_button']
        self.rec_button.disabled = True
        self.det_button = sef.ids['det_button']
        self.det_button.disabled = True
        self.wifi_button = sef.ids['wifi_button']
        self.wifi_button.disabled = True
        self.ip_cam = sef.ids['ip_cam']
        self.ip_cam.disabled = True
        self.dvr_cam = sef.ids['dvr_cam']
        self.dvr_cam.disabled = True

##################################### Start CAM ###############################################

    def startCamera(self):

        capture = cv2.VideoCapture(0)
        while True:
            ret, frame = capture.read()
            frame = cv2.flip(frame, 1)  # Flip vertically
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', frame)
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
        capture.release()
        cv2.destroyAllWindows()


##################################### Record CAM ###############################################

    def recordCamera(self):

        capture = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        op = cv2.VideoWriter('savedvideos/Sample1.avi', fourcc, 9.0, (640, 480))
        while True:
            ret, frame = capture.read()
            frame = cv2.flip(frame, 1)  # Flip vertically
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', frame)
            op.write(frame)
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
        op.release()
        capture.release()
        cv2.destroyAllWindows()

##################################### Detect Camera ###############################################

    def detectCamera(self):

        capture = cv2.VideoCapture(0)

        face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt_tree.xml')
        eye_glass = cv2.CascadeClassifier('haar/haarcascade_eye_tree_eyeglasses.xml')

        while True:
            ret, frame = capture.read()
            frame = cv2.flip(frame, 1)  # Flip vertically
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(frame, 'Face', (x + w, y + h), font, 1, (250, 250, 250), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                eye_g = eye_glass.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eye_g:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            # Display the resulting frame
            cv2.imshow('Video', frame)
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

        # When everything is done, release the capture
        capture.release()
        cv2.destroyAllWindows()

##################################### WIFI CAM ###############################################

    def wifiCamera(self):
        url = "http://192.168.1.135:8080/shot.jpg"
        face_cascade = cv2.CascadeClassifier("haar/haarcascade_frontalface_default.xml")
        eye_glass = cv2.CascadeClassifier('haar/haarcascade_eye_tree_eyeglasses.xml')

        while (True):

            img_resp = requests.get(url)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, 1)

            # operations on the frame
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                # To draw a rectangle in a face
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]

                eye_g = eye_glass.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eye_g:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                # Display the resulting frame
                cv2.imshow('PhoneCam', img)

            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

        # When everything done, release the capture
        url.release()
        cv2.destroyAllWindows()

##################################### IP Cam #####################################################

    def ipCamera(self):
        ip_add1 = ('rtsp://admin:admin123@')
        ip_add2 = ('/cam/realmonitor?channel=1&subtype=1')
        self.ipCam = self.ids['ipCam']
        ip_cam = self.ipCam.text

        cap = cv2.VideoCapture(ip_add1 + ip_cam + ip_add2)

        while (True):

            ret, frame = cap.read()
            # frame = cv2.flip(frame, 1)  # Flip vertically
            cv2.imshow('Stream IP camera opencv', frame)

            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

##################################### DVR Camera ################################################

    def dvrCamera(self):
        dvr_add1 = ('rtsp://admin:111111@')
        dvr_add2 = ('/cam/realmonitor?channel=')
        dvr_add3 = ('&subtype=00')
        self.dvrCam = self.ids['dvrCam']
        self.dvrChan = self.ids['dvrChan']
        dvr_cam = self.dvrCam.text
        dvr_chan = self.dvrChan.text

        cap = cv2.VideoCapture()
        cap.open(dvr_add1 + dvr_cam + dvr_add2 + dvr_chan + dvr_add3)  # change subtype to 00 for full resolution

        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Display the resulting frame
            cv2.imshow('img', frame)

            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

##################################### Forth Window ###############################################

class ForthWindow(Screen):
    def FuncButt(self):
        self.face_button = self.ids['face_button']
        self.face_button.disabled = True  # Prevents the user from clicking start again which may crash the program
        self.train_button = self.ids['train_button']
        self.train_button.disabled = True
        self.recg_button = sef.ids['recg_button']
        self.recg_button.disabled = True


##################################### Face CAM ###############################################

    def faceCamera(self):
        self.nub_user = self.ids['nub_user']
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video width
        cam.set(4, 480)  # set video height

        face_detector = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')

        # For each person, enter one numeric face id
        #face_id = input('\n enter user id end press <return> ==>  ')
        face_id = self.nub_user.text

        print("\n [INFO] Initializing face capture. Look the camera and wait ...")
        # Initialize individual sampling face count
        count = 0

        while (True):

            ret, img = cam.read()
            img = cv2.flip(img, 1)  # flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1

                # Save the captured image into the datasets folder
                cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

                cv2.imshow('image', img)

            k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 30:  # Take 30 face sample and stop video
                break

        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()

##################################### Train Face ###############################################

    def trainCamera(self):
        path = 'dataset'

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("haar/haarcascade_frontalface_default.xml");

        # function to get the images and label data
        def getImagesAndLabels(path):

            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faceSamples = []
            ids = []

            for imagePath in imagePaths:

                PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
                img_numpy = np.array(PIL_img, 'uint8')

                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = detector.detectMultiScale(img_numpy)

                for (x, y, w, h) in faces:
                    faceSamples.append(img_numpy[y:y + h, x:x + w])
                    ids.append(id)

            return faceSamples, ids

        print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces, ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
        recognizer.write('trainer/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

        # Print the numer of faces trained and end program
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

##################################### Recognize Face ###############################################

    def recgCamera(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        faceCascade = cv2.CascadeClassifier("haar/haarcascade_frontalface_default.xml")
        # faceCascade = cv2.CascadeClassifier(cascadePath);

        font = cv2.FONT_HERSHEY_SIMPLEX

        # iniciate id counter
        id = 0

        # names related to ids: example ==> Marcelo: id=1,  etc
        names = ['None', 'Jix', 'Eliott', 'Samira', 'Gilles', 'Haytham', 'Elias']

        # Initialize and start realtime video capture
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video widht
        cam.set(4, 480)  # set video height

        # Define min window size to be recognized as a face
        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)

        while True:

            ret, img = cam.read()
            img = cv2.flip(img, 1)  # Flip vertically

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            for (x, y, w, h) in faces:

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                # Check if confidence is less them 100 ==> "0" is perfect match
                if (confidence < 100):
                    id = names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 0, 0), 2)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 0, 255), 2)

            cv2.imshow('camera', img)

            k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break

        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()

##################################### Fifth Window ###############################################

class FifthWindow(Screen):
    def obj(self):

        self.live_detect = self.ids['live_detect']
        self.live_detect.disabled = True
        self.track_mobject = self.ids['track_mobject']
        self.track_mobject.disabled = True
        self.age_gender = self.ids['age_gender']
        self.age_gender.disabled = True
        self.live_detectT = self.ids['live_detectT']
        self.live_detectT.disabled = True




##################################### Live Detect YOLO ###############################################

    def liveDetect(self):
        # Initialize the parameters
        confThreshold = 0.5  # Confidence threshold
        nmsThreshold = 0.4  # Non-maximum suppression threshold
        inpWidth = 416  # Width of network's input image
        inpHeight = 416  # Height of network's input image

        parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
        parser.add_argument('--image', help='Path to image file.')
        parser.add_argument('--video', help='Path to video file.')
        args = parser.parse_args()

        # Load names of classes
        classesFile = "ObjectDetection-YOLO/darkflow/data/coco.names"
        classes = None
        with open(classesFile, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

        # Give the configuration and weight files for the model and load the network using them.
        modelConfiguration = "ObjectDetection-YOLO/darkflow/cfg/yolov3.cfg"
        modelWeights = "ObjectDetection-YOLO/darkflow/yolov3.weights"

        net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Get the names of the output layers
        def getOutputsNames(net):
            # Get the names of all the layers in the network
            layersNames = net.getLayerNames()
            # Get the names of the output layers, i.e. the layers with unconnected outputs
            return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # Draw the predicted bounding box
        def drawPred(classId, conf, left, top, right, bottom):
            # Draw a bounding box.
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

            label = '%.2f' % conf

            # Get the label for the class name and its confidence
            if classes:
                assert (classId < len(classes))
                label = '%s:%s' % (classes[classId], label)

            # Display the label at the top of the bounding box
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            cv2.rectangle(frame, (left, top - round(1.5 * labelSize[1])),
                         (left + round(1.5 * labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

        # Remove the bounding boxes with low confidence using non-maxima suppression
        def postprocess(frame, outs):
            frameHeight = frame.shape[0]
            frameWidth = frame.shape[1]

            # Scan through all the bounding boxes output from the network and keep only the
            # ones with high confidence scores. Assign the box's class label as the class with the highest score.
            classIds = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > confThreshold:
                        center_x = int(detection[0] * frameWidth)
                        center_y = int(detection[1] * frameHeight)
                        width = int(detection[2] * frameWidth)
                        height = int(detection[3] * frameHeight)
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)
                        classIds.append(classId)
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])

            # Perform non maximum suppression to eliminate redundant overlapping boxes with
            # lower confidences.
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
            for i in indices:
                i = i[0]
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

        # Process inputs
        winName = 'Deep learning object detection in OpenCV'
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)

        outputFile = "ObjectDetection-YOLO/yolo_out_py.avi"
        if (args.image):
            # Open the image file
            if not os.path.isfile(args.image):
                print("Input image file ", args.image, " doesn't exist")
                sys.exit(1)
            cap = cv2.VideoCapture(args.image)
            outputFile = args.image[:-4] + 'ObjectDetection-YOLO/_yolo_out_py.jpg'
        elif (args.video):
            # Open the video file
            if not os.path.isfile(args.video):
                print("Input video file ", args.video, " doesn't exist")
                sys.exit(1)
            cap = cv2.VideoCapture(args.video)
            outputFile = args.video[:-4] + 'ObjectDetection-YOLO/_yolo_out_py.avi'
        else:
            # Webcam input
            cap = cv2.VideoCapture(0)

        # Get the video writer initialized to save the output video
        if (not args.image):
            vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (
            round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while cv2.waitKey(1) < 0:

            # get frame from the video
            hasFrame, frame = cap.read()

            # Stop the program if reached end of video
            if not hasFrame:
                print("Done processing !!!")
                print("Output file is stored as ", outputFile)
                cv2.waitKey(3000)
                # Release device
                cap.release()
                break

            # Create a 4D blob from a frame.
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

            # Sets the input to the network
            net.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = net.forward(getOutputsNames(net))

            # Remove the bounding boxes with low confidence
            postprocess(frame, outs)

            # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
            t, _ = net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
            cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            # Write the frame with the detection boxes
            if (args.image):
                cv2.imwrite(outputFile, frame.astype(np.uint8))
            else:
                vid_writer.write(frame.astype(np.uint8))

            cv2.imshow(winName, frame)

##################################### Track Multi-Object CAM ###############################################

    def trackMobject(self):
        trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

        def createTrackerByName(trackerType):
            # Create a tracker based on tracker name
            if trackerType == trackerTypes[0]:
                tracker = cv2.TrackerBoosting_create()
            elif trackerType == trackerTypes[1]:
                tracker = cv2.TrackerMIL_create()
            elif trackerType == trackerTypes[2]:
                tracker = cv2.TrackerKCF_create()
            elif trackerType == trackerTypes[3]:
                tracker = cv2.TrackerTLD_create()
            elif trackerType == trackerTypes[4]:
                tracker = cv2.TrackerMedianFlow_create()
            elif trackerType == trackerTypes[5]:
                tracker = cv2.TrackerGOTURN_create()
            elif trackerType == trackerTypes[6]:
                tracker = cv2.TrackerMOSSE_create()
            elif trackerType == trackerTypes[7]:
                tracker = cv2.TrackerCSRT_create()
            else:
                tracker = None
                print('Incorrect tracker name')
                print('Available trackers are:')
                for t in trackerTypes:
                    print(t)

            return tracker

        if __name__ == '__main__':

            print("Default tracking algoritm is CSRT \n"
                  "Available tracking algorithms are:\n")
            for t in trackerTypes:
                print(t)

            trackerType = "CSRT"

            # Set video to load
            self.video_track = self.ids['video_track']
            videofile = self.video_track.text
            #videoPath = 'videofile'

            # Create a video capture object to read videos
            cap = cv2.VideoCapture(videofile)

            # Read first frame
            success, frame = cap.read()
            # quit if unable to read the video file
            if not success:
                print('Failed to read video')
                sys.exit(1)

            ## Select boxes
            bboxes = []
            colors = []

            # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
            # So we will call this function in a loop till we are done selecting all objects
            while True:
                # draw bounding boxes over objects
                # selectROI's default behaviour is to draw box starting from the center
                # when fromCenter is set to false, you can draw box starting from top left corner
                bbox = cv2.selectROI('multiTracker/MultiTracker', frame)
                bboxes.append(bbox)
                colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
                print("Press q to quit selecting boxes and start tracking")
                print("Press any other key to select next object")
                k = cv2.waitKey(0) & 0xFF
                if (k == 113):  # q is pressed
                    break

            print('Selected bounding boxes {}'.format(bboxes))

            ## Initialize MultiTracker
            # There are two ways you can initialize multitracker
            # 1. tracker = cv2.MultiTracker("CSRT")
            # All the trackers added to this multitracker
            # will use CSRT algorithm as default
            # 2. tracker = cv2.MultiTracker()
            # No default algorithm specified

            # Initialize MultiTracker with tracking algo
            # Specify tracker type

            # Create MultiTracker object
            multiTracker = cv2.MultiTracker_create()

            # Initialize MultiTracker
            for bbox in bboxes:
                multiTracker.add(createTrackerByName(trackerType), frame, bbox)

            # Process video and track objects
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # get updated location of objects in subsequent frames
                success, boxes = multiTracker.update(frame)

                # draw tracked objects
                for i, newbox in enumerate(boxes):
                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

                # show frame
                cv2.imshow('multiTracker/MultiTracker', frame)

                # quit on ESC button
                k = cv2.waitKey(5) & 0xFF
                if k == 27:
                    break

############################################## Age Gender #################################################

    def ageGender(self):
        def getFaceBox(net, frame, conf_threshold=0.7):
            frameOpencvDnn = frame.copy()
            frameHeight = frameOpencvDnn.shape[0]
            frameWidth = frameOpencvDnn.shape[1]
            blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

            net.setInput(blob)
            detections = net.forward()
            bboxes = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > conf_threshold:
                    x1 = int(detections[0, 0, i, 3] * frameWidth)
                    y1 = int(detections[0, 0, i, 4] * frameHeight)
                    x2 = int(detections[0, 0, i, 5] * frameWidth)
                    y2 = int(detections[0, 0, i, 6] * frameHeight)
                    bboxes.append([x1, y1, x2, y2])
                    cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
            return frameOpencvDnn, bboxes

        parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
        parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')

        args = parser.parse_args()

        faceProto = "AgeGender/opencv_face_detector.pbtxt"
        faceModel = "AgeGender/opencv_face_detector_uint8.pb"

        ageProto = "AgeGender/age_deploy.prototxt"
        ageModel = "AgeGender/age_net.caffemodel"

        genderProto = "AgeGender/gender_deploy.prototxt"
        genderModel = "AgeGender/gender_net.caffemodel"

        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        genderList = ['Male', 'Female']

        # Load network
        ageNet = cv2.dnn.readNet(ageModel, ageProto)
        genderNet = cv2.dnn.readNet(genderModel, genderProto)
        faceNet = cv2.dnn.readNet(faceModel, faceProto)

        # Open a video file or an image file or a camera stream
        cap = cv2.VideoCapture(args.input if args.input else 0)
        padding = 20
        while cv2.waitKey(1) < 0:
            # Read frame
            t = time.time()
            hasFrame, frame = cap.read()
            if not hasFrame:
                cv2.waitKey()
                break

            frameFace, bboxes = getFaceBox(faceNet, frame)
            if not bboxes:
                print("No face Detected, Checking next frame")
                continue

            for bbox in bboxes:
                # print(bbox)
                face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                       max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                # print("Gender Output : {}".format(genderPreds))
                print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                print("Age Output : {}".format(agePreds))
                print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

                label = "{},{}".format(gender, age)
                cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                           cv2.LINE_AA)
                cv2.imshow("Age Gender Demo", frameFace)
                # cv2.imwrite("out-{}".format(args.input),frameFace)
            print("time : {:.3f}".format(time.time() - t))

#################################### Live Detect Tensorflow #####################################

    def liveDetectT(self):

        # # Model preparation
        # Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
        # By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

        # What model to download.
        MODEL_NAME = 'object_recognition_detection/ssd_mobilenet_v1_coco_2018_01_28'
        # MODEL_FILE = MODEL_NAME + '.tar.gz'
        # DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join('object_recognition_detection/data', 'mscoco_label_map.pbtxt')

        NUM_CLASSES = 90

        # ## Download Model

        if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
            print('Downloading the model')
            opener = urllib.request.URLopener()
            opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
            tar_file = tarfile.open(MODEL_FILE)
            for file in tar_file.getmembers():
                file_name = os.path.basename(file.name)
                if 'frozen_inference_graph.pb' in file_name:
                    tar_file.extract(file, os.getcwd())
            print('Download complete')
        else:
            print('Model already exists')

        # ## Load a (frozen) Tensorflow model into memory.

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # ## Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        # intializing the web camera device

        import cv2
        cap = cv2.VideoCapture(0)

        # Running the tensorflow session
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                ret = True
                while (ret):
                    ret, image_np = cap.read()
                    image_np = cv2.flip(image_np, 1)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    #      plt.figure(figsize=IMAGE_SIZE)
                    #      plt.imshow(image_np)
                    cv2.imshow('image', cv2.resize(image_np, (1024, 768)))
                    k = cv2.waitKey(5) & 0xFF
                    if k == 27:
                        cv2.destroyAllWindows()
                        cap.release()
                        break



###################################Sixth Window ###################################################
class SixthWindow(Screen):
    def funcButt(self):
        self.find_image = self.ids['find_image']
        self.find_image.disabled = True
        self.ppl_count = self.ids['ppl_count']
        self.ppl_count.disabled = True
##################################### Find In Image ###############################################

    def findImage(self):

        self.find_imageup = self.ids['find_imageup']
        self.find_imagedn = self.ids['find_imagedn']

        filename1 = self.find_imageup.text
        filename2 = self.find_imagedn.text

        image = cv2.imread(filename1)
        cv2.imshow('people', image)
        cv2.waitKey(0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        template = cv2.imread(filename2, 0)

        w, h = template.shape[::-1]

        # result of template matching of object over an image
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
        sin_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        top_left = max_loc
        # increasing the size of bounding rectangle by 50 pixels
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 5)

        cv2.imshow('object found', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

############################################## People Count #########################################

    def pplCount(self):
        face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')

        self.count_image = self.ids['count_image']
        coutimage = self.count_image.text

        image = cv2.imread(coutimage)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(grayImage)

        print(faces)

        if len(faces) == 0:
            print("No faces found")

        else:
            print(faces)
            print(faces.shape)
            print("Number of faces detected: " + str(faces.shape[0]))

            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

            cv2.rectangle(image, ((0, image.shape[0] - 25)), (270, image.shape[0]), (255, 255, 255), -1)
            cv2.putText(image, "Number of faces detected: " + str(faces.shape[0]), (0, image.shape[0] - 10),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1)

            cv2.imshow('Image with faces', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

################################################### Face Swap #####################################################

    def faceSwap(self):
        def readPoints(path):
            # Create an array of points.
            points = [];

            # Read points
            with open(path) as file:
                for line in file:
                    x, y = line.split()
                    points.append((int(x), int(y)))

            return points

        # Apply affine transform calculated using srcTri and dstTri to src and
        # output an image of size.
        def applyAffineTransform(src, srcTri, dstTri, size):

            # Given a pair of triangles, find the affine transform.
            warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

            # Apply the Affine Transform just found to the src image
            dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            return dst

        # Check if a point is inside a rectangle
        def rectContains(rect, point):
            if point[0] < rect[0]:
                return False
            elif point[1] < rect[1]:
                return False
            elif point[0] > rect[0] + rect[2]:
                return False
            elif point[1] > rect[1] + rect[3]:
                return False
            return True

        # calculate delanauy triangle
        def calculateDelaunayTriangles(rect, points):
            # create subdiv
            subdiv = cv2.Subdiv2D(rect);

            # Insert points into subdiv
            for p in points:
                subdiv.insert(p)

            triangleList = subdiv.getTriangleList();

            delaunayTri = []

            pt = []

            for t in triangleList:
                pt.append((t[0], t[1]))
                pt.append((t[2], t[3]))
                pt.append((t[4], t[5]))

                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])

                if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
                    ind = []
                    # Get face-points (from 68 face detector) by coordinates
                    for j in range(0, 3):
                        for k in range(0, len(points)):
                            if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                                ind.append(k)
                                # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
                    if len(ind) == 3:
                        delaunayTri.append((ind[0], ind[1], ind[2]))

                pt = []

            return delaunayTri

        # Warps and alpha blends triangular regions from img1 and img2 to img
        def warpTriangle(img1, img2, t1, t2):

            # Find bounding rectangle for each triangle
            r1 = cv2.boundingRect(np.float32([t1]))
            r2 = cv2.boundingRect(np.float32([t2]))

            # Offset points by left top corner of the respective rectangles
            t1Rect = []
            t2Rect = []
            t2RectInt = []

            for i in range(0, 3):
                t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
                t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
                t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

            # Get mask by filling triangle
            mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

            # Apply warpImage to small rectangular patches
            img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
            # img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)

            size = (r2[2], r2[3])

            img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

            img2Rect = img2Rect * mask

            # Copy triangular region of the rectangular patch to the output image
            img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                        (1.0, 1.0, 1.0) - mask)

            img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect

        if __name__ == '__main__':

            # Make sure OpenCV is version 3.0 or above
            (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

            if int(major_ver) < 3:
                print >> sys.stderr, 'ERROR: Script needs OpenCV 3.0 or higher'
                sys.exit(1)

            def show_load1(self):
                content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
                self._popup = Popup(title="Load file", content=content, size_hint=(0.9, 0.9))
                self._popup.open()

            # Read images
            # filename1 = 'FaceSwap/ted_cruz.jpg'
            # filename2 = 'FaceSwap/donald_trump.jpg'

            self.swap_imageup = self.ids['swap_imageup']
            self.swap_imagedn = self.ids['swap_imagedn']

            filename1 = self.swap_imageup.text
            filename2 = self.swap_imagedn.text

            img1 = cv2.imread(filename1);
            img2 = cv2.imread(filename2);
            img1Warped = np.copy(img2);

            # Read array of corresponding points
            points1 = readPoints(filename1 + '.txt')
            points2 = readPoints(filename2 + '.txt')

            # Find convex hull
            hull1 = []
            hull2 = []

            hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

            for i in range(0, len(hullIndex)):
                hull1.append(points1[int(hullIndex[i])])
                hull2.append(points2[int(hullIndex[i])])

            # Find delanauy traingulation for convex hull points
            sizeImg2 = img2.shape
            rect = (0, 0, sizeImg2[1], sizeImg2[0])

            dt = calculateDelaunayTriangles(rect, hull2)

            if len(dt) == 0:
                quit()

            # Apply affine transformation to Delaunay triangles
            for i in range(0, len(dt)):
                t1 = []
                t2 = []

                # get points for img1, img2 corresponding to the triangles
                for j in range(0, 3):
                    t1.append(hull1[dt[i][j]])
                    t2.append(hull2[dt[i][j]])

                warpTriangle(img1, img1Warped, t1, t2)

            # Calculate Mask
            hull8U = []
            for i in range(0, len(hull2)):
                hull8U.append((hull2[i][0], hull2[i][1]))

            mask = np.zeros(img2.shape, dtype=img2.dtype)

            cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

            r = cv2.boundingRect(np.float32([hull2]))

            center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))

            # Clone seamlessly.
            output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)

            cv2.imshow("Face Swapped", output)
            cv2.waitKey(0)

            cv2.destroyAllWindows()
############################################



####################################################### Seventh Window ###################################

# class LoadDialog(FloatLayout):
#     load = ObjectProperty(None)
#     cancel = ObjectProperty(None)


class SeventhWindow(Screen):
    pass
    # loadfile = ObjectProperty(None)
    # text_input = ObjectProperty(None)
    # cancel = ObjectProperty(None)
    #
    # def dismiss_popup(self):
    #     self._popup.dismiss()
    #
    # def show_load(self):
    #     content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
    #     self._popup = Popup(title="Load file", content=content,
    #                         size_hint=(0.9, 0.9))
    #     self._popup.open()
    #
    # def load(self, path, filename):
    #     with open(os.path.join(path, filename[0])) as stream:
    #         self.text_input.text = stream.read()
    #
    #     self.dismiss_popup()


##################################################################################################


########################################### Eighth Window ####################################

class EighthWindow(Screen):
    def build(self):
        v = root.ids.fc
        if len(sys.argv) > 1:
            v.path = sys.argv[1]

        v.bind(selection=lambda *x: pprint("selection: %s" % x[1:]))
        v.bind(path=lambda *x: pprint("path: %s" % x[1:]))


class WindowManager(ScreenManager):
    pass

kv = Builder.load_file("test3.kv")

db = DataBase("users.txt")


class test3(App):
    def build(self):
        return kv

if __name__ == "__main__":
    test3().run()
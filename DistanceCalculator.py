import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import csv
from collections import Counter
import time


class Calci:
    
    face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.75)
    
    def something(self):
        time.sleep(30000)

    def calculate_distance(self,distance_pixel, distance_cm):
        # get corrlation coffs
        coff = np.polyfit(distance_pixel, distance_cm, 2)
        outfile = open('C:/Users/nandi/final_project/test/Distance/myfile.txt','w')
        # For webcam input:
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        while( int(time.time() - start_time) < 30 ):
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(image)
            bbox_list, eyes_list = [], []
            if results.detections:
                for detection in results.detections:

                    # get bbox data
                    bboxc = detection.location_data.relative_bounding_box
                    ih, iw, ic = image.shape
                    bbox = int(bboxc.xmin*iw), int(bboxc.ymin *
                                                   ih), int(bboxc.width*iw), int(bboxc.height*ih)
                    bbox_list.append(bbox)
                    left_eye = detection.location_data.relative_keypoints[0]
                    right_eye = detection.location_data.relative_keypoints[1]
                    eyes_list.append([(int(left_eye.x*iw), int(left_eye.y*ih)),
                                      (int(right_eye.x*iw), int(right_eye.y*ih))])
                    
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for bbox, eye in zip(bbox_list, eyes_list):
                dist_between_eyes = np.sqrt(
                    (eye[0][1]-eye[1][1])**2 + (eye[0][0]-eye[1][0])**2)

                #calculate distance in cm
                a, b, c = coff
                distance_cm = a*dist_between_eyes**2+b*dist_between_eyes+c
                #print(distance_cm)
                outfile.write("%d \n" %distance_cm)
                print(distance_cm)
            cv2.imshow('webcam', image)
            if cv2.waitKey(1) & 0xFF == ord('k'):
                break
        cap.release()
        
if __name__ == '__main__':
    distance_df = pd.read_csv('C:/Users/nandi/final_project/test/Distance/distance_xy.csv')
    eye_screen_distance = Calci()
    eye_screen_distance.calculate_distance(distance_df['distance_pixel'], distance_df['distance_cm'])
    
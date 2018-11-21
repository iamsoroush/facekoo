# -*- coding: utf-8 -*-
"""Main module for online face recognition.
"""
# Author: Soroush Moazed <soroush.moazed@gmail.com>


import cv2
import dlib
import pandas as pd
import numpy as np
import os
from collections import OrderedDict
from face_recognition import FaceRecognizer
from face_alignment import FaceAligner
import atexit

from helpers import draw_border, resize_image, drop_high_correlated_images, one_to_many_high_correlated_indices


if __name__ == '__main__':
    shape_predictor_path = 'models/shape_predictor/shape_predictor_5_face_landmarks.dat'
    stream = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)
    FACIAL_LANDMARKS_5_IDXS = OrderedDict([
        ("right_eye", (2, 3)),
        ("left_eye", (0, 1)),
        ("nose", (4,))
    ])
    fa = FaceAligner(predictor=predictor, landmarks_idxs=FACIAL_LANDMARKS_5_IDXS,
                     desired_face_width=182)
    fr = FaceRecognizer()
    while True:
        (grabbed, frame) = stream.read()
        if not grabbed:
            break
        resized_frame = resize_image(frame, out_pixels_wide=1000)
        overlay = resized_frame.copy()
        output = resized_frame.copy()
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)  # Detect faces in the gray scale frame
        if rects:
            aligned = fa.align(resized_frame, gray, rects[0])
            name = 'None'
            if fr.classifier is not None:
                name, prob = fr.predict(img=aligned)
                if prob < 0.8:
                    print('Unknown')
                    name = 'Unknown'
                else:
                    print(name, ' : %', round(prob * 100, 2))
            d = rects[0]
            alpha = 0.5
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, \
                d.bottom() + 1, d.width(), d.height()
            draw_border(overlay, (x1, y1), (x2, y2), (162, 255, 0),
                        5, 10, 10, name)  # Draw a fancy border around the faces
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)  # Make semi-transparent bounding box
        cv2.imshow("Face Recognition", output)
        key = cv2.waitKey(1)

        # press q to break out of the loop
        if key == ord("q"):
            break
        elif key == ord("c"):
            name = input("Please enter your name: ")
            print("Every 0.5 seconds an image will be captured, when you are done, press 'c' again.")
            aligned_images = capture_pics(stream, detector, fa, class_name=name)
            print("retraining the classifier ...")
            fr.retrain(images=aligned_images, class_name=name)

    # cleanup
    fr.clean()
    cv2.destroyAllWindows()
    stream.release()


@atexit.register
def exit_handler():
    print('Goodbye :)')

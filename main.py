
import cv2
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
from timeit import default_timer
from court_detector import CourtDetector
from tennis_court_model import TennisCourtModel
import numpy as np

court_detector = CourtDetector()
tennis_court_model = TennisCourtModel()


def detect_court():
    img_paths = glob("court_images/*")


    for idx, img_path in enumerate(img_paths):
        start = default_timer()
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        _, court_line_img = court_detector.court_detect_from_image(img)

        #im_rgb = cv2.cvtColor(court_line_img, cv2.COLOR_BGR2RGB)
        temp = img_path.split("\\")[-1]
        filename = f"detected_lines/detected_{temp}"
        cv2.imwrite(filename, court_line_img)
        
        end = default_timer() - start
        print(f"ELAPSED TIME: {end:.2f} seconds")

        print(f"Detected Image saved to: {filename}")
        #plt.imshow(im_rgb)
        #plt.show()

    


if __name__ == '__main__':
    detect_court()
    #evaluate_court()


    


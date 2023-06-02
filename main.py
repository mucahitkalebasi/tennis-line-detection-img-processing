

import cv2
from glob import glob
from timeit import default_timer
from tennis_line_detector import CourtLineDetector
from tennis_court_model import TennisCourtModel


tennis_line_detector = CourtLineDetector()
tennis_court_model = TennisCourtModel()


def detect_court():
    img_paths = glob("court_images/*.jpg")

    try:
        for idx, img_path in enumerate(img_paths):
            start = default_timer()
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            _, court_line_img = tennis_line_detector.court_detect_from_image(img)

            temp = img_path.split("\\")[-1]
            filename = f"detected_lines/detected_{temp}"
            cv2.imwrite(filename, court_line_img)
            
            end = default_timer() - start
            print(f"ELAPSED TIME: {end:.2f} seconds")

            print(f"Detected Image saved to: {filename}")
            #plt.imshow(im_rgb)
            #plt.show()
    except:
        print(f"File: {filename}")

    


if __name__ == '__main__':
    detect_court()


    


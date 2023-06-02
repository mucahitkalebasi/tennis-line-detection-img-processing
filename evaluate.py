
import cv2
#from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
from timeit import default_timer
from court_detector import CourtLineDetector
from tennis_court_model import TennisCourtModel
import numpy as np


court_detector = CourtLineDetector()
tennis_court_model = TennisCourtModel()


def create_dataset():
    ground_truth_dataset = tennis_court_model.get_ground_truth_dataset()
    prediction_data = tennis_court_model.get_raw_prediction_dataset()

    img_paths = ["court_images\\court_reference1.jpg", "court_images\\court_reference2.jpg", "court_images\\court_reference3.jpg"]

    for idx, img_path in enumerate(img_paths):
        start = default_timer()
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        lines, _ = court_detector.court_detect_from_image(img)

        temp_dict = dict(zip(tennis_court_model.points, lines)) # for removing unnecessary coordinates
        prediction_data[f"image_{idx}"] = temp_dict.values()
        
        end = default_timer() - start
        print(f"ELAPSED TIME: {end:.2f} seconds")

        #plt.imshow(im_rgb)
        #plt.show()

    prediction_data = tennis_court_model.preprocess_dataset(prediction_data, False)
    prediction_data.to_csv("prediction_data.csv")
    ground_truth_dataset.to_csv("ground_truth_data.csv")


def evaluate_court():
    ground_truth_dataset = pd.read_csv("ground_truth_data.csv", index_col=0)
    prediction_dataset = pd.read_csv("prediction_data.csv", index_col=0)
    tol = 5 # tolerance
    results = court_detector.calculate_scores(ground_truth_dataset, prediction_dataset, tol)


    labels = list(results.keys())
    precision_vals = [val['Precision'] for val in results.values()]
    recall_vals = [val['Recall'] for val in results.values()]
    f1_vals = [val['F1-score'] for val in results.values()]

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots(figsize=(16, 9))
    rects1 = ax.bar(x - width, precision_vals, width, label='Precision')
    rects2 = ax.bar(x, recall_vals, width, label='Recall')
    rects3 = ax.bar(x + width, f1_vals, width, label='F1-score')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by image and metric')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()

    plt.show()
    #print(results)





if __name__ == '__main__':
    #create_dataset()
    evaluate_court()
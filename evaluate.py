#%%

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from timeit import default_timer
from tennis_line_detector import CourtLineDetector
from tennis_court_model import TennisCourtModel
import numpy as np

tennis_line_detector = CourtLineDetector()
tennis_court_model = TennisCourtModel()


def create_dataset():
    ground_truth_dataset = tennis_court_model.get_ground_truth_dataset()
    prediction_data = tennis_court_model.get_raw_prediction_dataset()

    img_paths = ["evaluation_images\\court_reference01.jpg", "evaluation_images\\court_reference02.jpg", "evaluation_images\\court_reference03.jpg"]

    for idx, img_path in enumerate(img_paths):
        start = default_timer()
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        lines, _ = tennis_line_detector.court_detect_from_image(img)

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
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
            


    ground_truth_dataset = pd.read_csv("ground_truth_data.csv", index_col=0)
    prediction_dataset = pd.read_csv("prediction_data.csv", index_col=0)
    tol = 5 # tolerance
    results = tennis_line_detector.calculate_scores(ground_truth_dataset, prediction_dataset, tol)


    labels = list(results.keys())
    
    acc_vals = [val['Accuracy'] for val in results.values()]
    mse_vals = [val['MSE'] for val in results.values()]

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots(figsize=(16, 9))
    rects1 = ax.bar(x - width, acc_vals, width, label='Accuracy', color = 'orange')
    

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Accuracy of the Images')
    ax.set_xticks(x - width/2)  #  adjust position of x-axis labels
    ax.set_xticklabels(labels)
    ax.legend()

 
    autolabel(rects1)
    fig.tight_layout()
    plt.show()

    
    fig, ax = plt.subplots(figsize=(16, 9))
    rects2 = ax.bar(x - width, mse_vals, width, label='MSE', color = 'orange')
    

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Mean Squared Error')
    ax.set_xticks(x - width/2)  #  adjust position of x-axis labels
    ax.set_xticklabels(labels)
    ax.legend()
    
    autolabel(rects2)
    fig.tight_layout()
    plt.show()

    #print(results)


if __name__ == '__main__':
    #create_dataset()
    evaluate_court()
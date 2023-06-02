import numpy as np
import cv2
from sympy.geometry import Line
from itertools import combinations
from tennis_court_model import TennisCourtModel
import ast
import math

class CourtLineDetector:

  def __init__(self):
    self.colour_threshold = 200
    self.dist_tau = 3
    self.intensity_threshold = 40
    self.tennis_court_model = TennisCourtModel()
    self.v_width = 0
    self.v_height = 0
    self.frame = None
    self.gray = None
    self.court_warp_matrix = []
    self.game_warp_matrix = []


  
  def court_detect_from_image(self, img):
    tuple_lines = []
    lines = self._detect(img) # send the img array

    for i in range(0, len(lines), 4):
        first_point, second_point =  (int(lines[i]),int(lines[i+1])), (int(lines[i+2]), int(lines[i+3]))
        tuple_lines.append(first_point)
        tuple_lines.append(second_point)

        cv2.line(img, first_point, second_point, (255,153,204), 2)
        cv2.circle(img, first_point, 2, (0, 0, 255), -1)  # Circle for point 1
        cv2.circle(img, second_point, 2, (0, 0, 255), -1)  # Circle for point 2

 
    return tuple_lines, img
  

  def string_to_tuple(self, s):
    # Convert a string-formatted tuple to an actual tuple
    return tuple(map(int, s.strip('()').split(',')))
  



  

  def calculate_distances(self, truth_df, pred_df, image, tol):
    tp, fp, mse, n = 0, 0, 0, 0
    for _, truth_row in truth_df.iterrows():
        for _, pred_row in pred_df.iterrows():
            if truth_row['point_name'] == pred_row['point_name']:
                #print(f"Truth: {truth_row[image]}, Pred: {pred_row[image]}")  # Add this line
                
                # TP, FP calculation 
                a = ast.literal_eval(truth_row[image])
                b = ast.literal_eval(pred_row[image])

                dx = b[0] - a[0]
                dy = b[1] - a[1]
                distance = math.sqrt(dx*dx + dy*dy)

                # MSE calculation
                squared_distance = dx*dx + dy*dy
                mse += squared_distance
                n+=1

                if distance <= tol:
                    tp += 1
                else:
                    fp += 1

    
    
    return tp, fp, mse, n
  
  def calculate_scores(self, truth_df, pred_df, tol):
    images = [col for col in truth_df.columns if col not in ['point_name']]
    results = {}
    for image in images:
        tp, fp, mse, n = self.calculate_distances(truth_df, pred_df, image, tol)
        
        accuracy = tp / (tp + fp) if (tp + fp) > 0 else 0
        mse = mse / n if n > 0 else 0

        print(image.split("_")[0], image.split("_")[1], ":\t" ,"tp:", tp, "fp: ",fp)
        print(f"Accuracy: {accuracy}")
        print(f"Mean Squared Error: {mse}")
        results[image] = {"Accuracy": accuracy, "MSE": mse}
    return results
  
  def _detect(self, frame):
    """
    Detecting the court in the frame
    """
    self.frame = frame
    self.v_height, self.v_width = frame.shape[:2]
    # Get binary image from the frame
    
    self.gray = self._threshold(frame)
    
    # Filter pixel using the court known structure
    filtered = self._filter_pixels(self.gray)
    
    # Detect lines using Hough transform
    horizontal_lines, vertical_lines = self._detect_lines(filtered)
    
    # Find transformation from reference court to frame`s court
    court_warp_matrix, game_warp_matrix = self._find_homography(horizontal_lines, vertical_lines)

    self.court_warp_matrix.append(court_warp_matrix)
    self.game_warp_matrix.append(game_warp_matrix)
    
    # Find important lines location on frame
    return self.find_lines_location()

  def _threshold(self, frame):
    """
    Simple thresholding for white pixels
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    return gray

  def _filter_pixels(self, gray):
    """
    Filter pixels by using the court line structure
    """
    for i in range(self.dist_tau, len(gray) - self.dist_tau):
      for j in range(self.dist_tau, len(gray[0]) - self.dist_tau):
        if gray[i, j] == 0:
          continue
        if (gray[i, j] - gray[i + self.dist_tau, j] > self.intensity_threshold and gray[i, j] - gray[i - self.dist_tau, j] > self.intensity_threshold):
          continue
        if (gray[i, j] - gray[i, j + self.dist_tau] > self.intensity_threshold and gray[i, j] - gray[i, j - self.dist_tau] > self.intensity_threshold):
          continue
        gray[i, j] = 0
    return gray

  def _detect_lines(self, gray):
        """
        Finds all line in frame using Hough transform
        """
        minLineLength = 100
        maxLineGap = 20
        # Detect all lines
        lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 80, minLineLength=minLineLength, maxLineGap=maxLineGap)
        lines = np.squeeze(lines)
 
         # Classify the lines using their slope
        horizontal, vertical = self._classify_lines(lines)
        
        # Merge lines that belong to the same line on frame
        horizontal, vertical = self._merge_lines(horizontal, vertical)
        return horizontal, vertical

  def _classify_lines(self, lines):
        """
        Classify line to vertical and horizontal lines
        """
        horizontal = []
        vertical = []
        highest_vertical_y = np.inf
        lowest_vertical_y = 0
        for line in lines:
            x1, y1, x2, y2 = line
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            if dx > 2 * dy:
                horizontal.append(line)
            else:
                vertical.append(line)
                highest_vertical_y = min(highest_vertical_y, y1, y2)
                lowest_vertical_y = max(lowest_vertical_y, y1, y2)

        # Filter horizontal lines using vertical lines lowest and highest point
        clean_horizontal = []
        h = lowest_vertical_y - highest_vertical_y
        lowest_vertical_y += h / 15
        highest_vertical_y -= h * 2 / 15
        for line in horizontal:
            x1, y1, x2, y2 = line
            if lowest_vertical_y > y1 > highest_vertical_y and lowest_vertical_y > y1 > highest_vertical_y:
                clean_horizontal.append(line)
        return clean_horizontal, vertical

  
  def _merge_lines(self, horizontal_lines, vertical_lines):
        """
        Merge lines that belongs to the same frame`s lines
        """

        # Merge horizontal lines
        horizontal_lines = sorted(horizontal_lines, key=lambda item: item[0])
        mask = [True] * len(horizontal_lines)
        new_horizontal_lines = []
        for i, line in enumerate(horizontal_lines):
            if mask[i]:
                for j, s_line in enumerate(horizontal_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        dy = abs(y3 - y2)
                        if dy < 10:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[0])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False
                new_horizontal_lines.append(line)

        # Merge vertical lines
        vertical_lines = sorted(vertical_lines, key=lambda item: item[1])
        xl, yl, xr, yr = (0, self.v_height * 6 / 7, self.v_width, self.v_height * 6 / 7)
        mask = [True] * len(vertical_lines)
        new_vertical_lines = []
        for i, line in enumerate(vertical_lines):
            if mask[i]:
                for j, s_line in enumerate(vertical_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        xi, yi = line_intersection(((x1, y1), (x2, y2)), ((xl, yl), (xr, yr)))
                        xj, yj = line_intersection(((x3, y3), (x4, y4)), ((xl, yl), (xr, yr)))

                        dx = abs(xi - xj)
                        if dx < 10:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[1])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False

                new_vertical_lines.append(line)
        return new_horizontal_lines, new_vertical_lines

  def _find_homography(self, horizontal_lines, vertical_lines):
        """
        Finds transformation from reference court to frame`s court using 4 pairs of matching points
        """
        max_score = -np.inf
        max_mat = None
        max_inv_mat = None
        # Loop over every pair of horizontal lines and every pair of vertical lines
        for horizontal_pair in list(combinations(horizontal_lines, 2)):
            for vertical_pair in list(combinations(vertical_lines, 2)):
                h1, h2 = horizontal_pair
                v1, v2 = vertical_pair
                # Finding intersection points of all lines
                i1 = line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v1[0:2]), tuple(v1[2:])))
                i2 = line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v2[0:2]), tuple(v2[2:])))
                i3 = line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v1[0:2]), tuple(v1[2:])))
                i4 = line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v2[0:2]), tuple(v2[2:])))

                intersections = [i1, i2, i3, i4]
                intersections = sort_intersection_points(intersections)

                for i, configuration in self.tennis_court_model.prepare_court_configs().items():
                    # Find transformation
                    matrix, _ = cv2.findHomography(np.float32(configuration), np.float32(intersections), method=0)
                    inv_matrix = cv2.invert(matrix)[1]
                    # Get transformation score
                    confi_score = self._get_confi_score(matrix)

                    if max_score < confi_score:
                        max_score = confi_score
                        max_mat = matrix
                        max_inv_mat = inv_matrix
                        self.best_conf = i
   

        return max_mat, max_inv_mat

  def _get_confi_score(self, matrix):
        """
        Calculate transformation score
        """
        court = cv2.warpPerspective(self.tennis_court_model.court_img, matrix, self.frame.shape[1::-1])
        court[court > 0] = 1
        gray = self.gray.copy()
        gray[gray > 0] = 1
        correct = court * gray
        wrong = court - correct
        c_p = np.sum(correct)
        w_p = np.sum(wrong)
        return c_p - 0.5 * w_p


  def add_court_overlay(self, frame, homography=None, overlay_color=(255, 255, 255), frame_num=-1):
        """
        Add overlay of the court to the frame
        """
        if homography is None and len(self.court_warp_matrix) > 0 and frame_num < len(self.court_warp_matrix):
            homography = self.court_warp_matrix[frame_num]
        court = cv2.warpPerspective(self.tennis_court_model.court_img, homography, frame.shape[1::-1])
        frame[court > 0, :] = overlay_color
        return frame

  def find_lines_location(self):
        """
        Finds important lines location on frame
        """
        self.p = np.array(self.tennis_court_model.get_court_lines(), dtype=np.float32).reshape((-1, 1, 2))
        self.lines = cv2.perspectiveTransform(self.p, self.court_warp_matrix[-1]).reshape(-1)

        
        return self.lines

  def get_warped_court(self):
        """
        Returns warped court using the reference court and the transformation of the court
        """
        court = cv2.warpPerspective(self.tennis_court_model.court_img, self.court_warp_matrix[-1], self.frame.shape[1::-1])
        court[court > 0] = 1
        return court



def line_intersection(line1, line2):
    """
    Find 2 lines intersection point
    """
    l1 = Line(line1[0], line1[1])
    l2 = Line(line2[0], line2[1])
    
    intersection = l1.intersection(l2)
    
    
    return intersection[0].coordinates

def sort_intersection_points(intersections):
    """
    sort intersection points from top left to bottom right
    """
    y_sorted = sorted(intersections, key=lambda x: x[1])
    p12 = y_sorted[:2]
    p34 = y_sorted[2:]
    p12 = sorted(p12, key=lambda x: x[0])
    p34 = sorted(p34, key=lambda x: x[0])
    return p12 + p34
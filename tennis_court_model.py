import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TennisCourtModel:
    """
    Tennis Court reference model
    """
    def __init__(self):
        # Initializing reference points (in (x, y) format)
        self.top_line = ((286, 561), (1379, 561))             # Top boundary of the court
        self.bottom_line = ((286, 2935), (1379, 2935))        # Bottom boundary of the court
        self.center_net = ((286, 1748), (1379, 1748))         # Net line (center line across the width)
        self.left_court_border = ((286, 561), (286, 2935))    # Left boundary of the court
        self.right_court_border = ((1379, 561), (1379, 2935)) # Right boundary of the court
        self.left_inner_border = ((423, 561), (423, 2935))    # Left service border (inner boundary)
        self.right_inner_border = ((1242, 561), (1242, 2935)) # Right service border (inner boundary)
        self.center_line = ((832, 1110), (832, 2386))         # Center service line
        self.top_inner_border = ((423, 1110), (1242, 1110))   # Top service border
        self.bottom_inner_border = ((423, 2386), (1242, 2386))# Bottom service border
        self.top_extension = (832.5, 580)                     # An extra top point (for extension)
        self.bottom_extension = (832.5, 2910)                 # An extra bottom point (for extension)


        # Preparing court configurations
        self.prepare_court_configs()

        self.line_thickness = 1
        self.court_w = 1117
        self.court_h = 2408
        self.top_bottom_padding = 549
        self.right_left_padding = 274
        self.total_court_w = self.court_w + self.right_left_padding * 2
        self.total_court_h = self.court_h + self.top_bottom_padding * 2

        self.points = ["top_left", "top_right", "bottom_left", "bottom_right", "middle_left", "middle_right", "top_left", 
        "bottom_left", "top_right", "bottom_right", "left_middle_top", "left_middle_bottom", "right_middle_top", 
        "right_middle_bottom", "middle_top", "middle_bottom", "second_middle_left", "second_middle_right", "middle_bottom_left", 
        "middle_bottom_right"]

        self.raw_dataframe_points = [
            "top_left", "top_right", "bottom_left", "bottom_right", "middle_left", 
            "middle_right", "left_middle_top", "left_middle_bottom", "right_middle_top", 
            "right_middle_bottom", "middle_top", "middle_bottom", "second_middle_left", 
            "second_middle_right", "middle_bottom_left", "middle_bottom_right"
        ]

        self.court_img = cv2.cvtColor(cv2.imread('court_configurations/court_reference.png'), cv2.COLOR_BGR2GRAY)

    def prepare_court_configs(self):
        """
        Prepare a dictionary of court configurations,
        each configuration is a set of 4 points (or more)
        that define different areas or lines on the court.
        """
        self.court_configurations = {1: [*self.top_line, *self.bottom_line],
                                     2: [self.left_inner_border[0], self.right_inner_border[0], 
                                         self.left_inner_border[1], self.right_inner_border[1]],
                                     3: [self.left_inner_border[0], self.right_court_border[0], 
                                         self.left_inner_border[1], self.right_court_border[1]],
                                     4: [self.left_court_border[0], self.right_inner_border[0], 
                                         self.left_court_border[1], self.right_inner_border[1]],
                                     5: [*self.top_inner_border, *self.bottom_inner_border],
                                     6: [*self.top_inner_border, self.left_inner_border[1], 
                                         self.right_inner_border[1]],
                                     7: [self.left_inner_border[0], self.right_inner_border[0], 
                                         *self.bottom_inner_border],
                                     8: [self.right_inner_border[0], self.right_court_border[0], 
                                         self.right_inner_border[1], self.right_court_border[1]],
                                     9: [self.left_court_border[0], self.left_inner_border[0], 
                                         self.left_court_border[1], self.left_inner_border[1]],
                                     10: [self.top_inner_border[0], self.center_line[0], 
                                          self.bottom_inner_border[0], self.center_line[1]],
                                     11: [self.center_line[0], self.top_inner_border[1], 
                                          self.center_line[1], self.bottom_inner_border[1]],
                                     12: [*self.bottom_inner_border, self.left_inner_border[1], 
                                          self.right_inner_border[1]]}
        

        return self.court_configurations
    

    def get_ground_truth_dataset(self):
        """
        Create a DataFrame containing ground truth (x, y) pixel locations
        for various named points on the tennis court. Then preprocess it.
        """
        ground_truth_pixels = {

            "top_left" :            [(172,84), (101,85), (11,175)],
            "top_right" :           [(468,84), (503,86), (199,89)],
            "bottom_left" :         [(78,331), (15,401), (510,393)],
            "bottom_right" :        [(562,331), (618,401), (616,199)],
            "middle_left" :         [(136,176), (66,212), (197,259)],
            "middle_right" :        [(503,176), (548,212), (372,134)],
            "left_middle_top" :     [(209,84), (150,85), (41,162)],
            "left_middle_bottom" :  [(138,332), (91,401), (529,359)],
            "right_middle_top" :    [(431,84), (452,86), (181,97)],
            "right_middle_bottom" : [(502,331), (543,402), (607,216)],
            "middle_top" :          [(320,122), (305,137), (191,151)],
            "middle_bottom" :       [(320,250), (313,303), (431,231)],
            "second_middle_left" :  [(198,122), (141,137), (115,193)],
            "second_middle_right" : [(442,122), (467,138), (252,117)],
            "middle_bottom_left" :  [(161,250), (109,309), (369,296)],
            "middle_bottom_right" : [(479,249), (515,303), (479,180)]
        }
        ground_truth_dataset = pd.DataFrame(ground_truth_pixels)
        ground_truth_dataset = self.preprocess_dataset(ground_truth_dataset)
        
        return ground_truth_dataset
    

    def preprocess_dataset(self, dataset, transpose = True):
        """
        Preprocess a DataFrame of point coordinates.
        If 'transpose' is True, pivot the DataFrame so that each row 
        corresponds to a point_name and columns correspond to images.
        Otherwise, rename columns using 'self.raw_dataframe_points'.
        """
        if transpose:
            dataset = dataset.T
            dataset = dataset.reset_index()
            dataset.rename(columns = {'index':'point_name'}, inplace = True)
            dataset.rename(columns={0: "image_0", 1: "image_1",
                                    2: "image_2"}, inplace=True)

        else: 
            dataset = dataset.reset_index()
            dataset.rename(columns={0: "point_name"}, inplace=True)
            dataset["point_name"].replace(to_replace=dataset["point_name"].values,
                                           value=self.raw_dataframe_points,
                                           inplace=True)


        return dataset



    def get_raw_prediction_dataset(self):
        """
        Create a blank DataFrame for raw predictions, 
        with row indexes set to the names of the points.
        """
        dictionary = {
            "top_left",           
            "top_right",         
            "bottom_left",        
            "bottom_right",       
            "middle_left",        
            "middle_right",     
            "left_middle_top",  
            "left_middle_bottom", 
            "right_middle_top", 
            "right_middle_bottom",
            "middle_top",       
            "middle_bottom",  
            "second_middle_left", 
            "second_middle_right",
            "middle_bottom_left", 
            "middle_bottom_right"
        }

        dataset = pd.DataFrame.from_dict(dictionary)
        dataset = dataset.set_index([0])

        return dataset
        


    def construct_court_model(self):
        """
        Construct the reference court model image by drawing the court lines
        on a blank (zeros) image, then dilate the lines and save it.
        """
        court_img = np.zeros((self.court_h + 2 * self.top_bottom_padding, 
                              self.court_w + 2 * self.right_left_padding), dtype=np.uint8)
        self.draw_lines(court_img)
        court_img = cv2.dilate(court_img, np.ones((5, 5), dtype=np.uint8))
        plt.imsave('court_configurations/court_reference.png', court_img, cmap='gray')
        self.court_img = court_img
        return court_img

    def draw_lines(self, img):
        """
        Draw the court lines on the provided image using the stored line coordinates.
        """
        cv2.line(img, *self.top_line, 1, self.line_thickness)
        cv2.line(img, *self.bottom_line, 1, self.line_thickness)
        cv2.line(img, *self.top_inner_border, 1, self.line_thickness)
        cv2.line(img, *self.bottom_inner_border, 1, self.line_thickness)
        cv2.line(img, *self.left_court_border, 1, self.line_thickness)
        cv2.line(img, *self.right_court_border, 1, self.line_thickness)
        cv2.line(img, *self.left_inner_border, 1, self.line_thickness)
        cv2.line(img, *self.right_inner_border, 1, self.line_thickness)
        cv2.line(img, *self.center_line, 1, self.line_thickness)

    def get_court_lines(self):
        """
        Returns a list of all line endpoints in the court model.
        """
        all_lines = [*self.top_line, *self.bottom_line, *self.center_net, *self.left_court_border, 
                     *self.right_court_border, *self.left_inner_border, *self.right_inner_border, 
                     *self.center_line, *self.top_inner_border, *self.bottom_inner_border]
        return all_lines

    def get_extension_parts(self):
        """
        Return the two extension points defined for the court model.
        """
        return [self.top_extension, self.bottom_extension]

    def save_court_configs(self):
        """
        For each configuration (4 or more points), draw red circles on
        the court reference image and save them as separate PNG files.
        """
        for i, config in self.court_configurations.items():
            img = cv2.cvtColor(255 - self.court_img, cv2.COLOR_GRAY2BGR)
            for point in config:
                img = cv2.circle(img, point, 15, (0, 0, 255), 30)
            cv2.imwrite(f'court_configurations/court_conf_{i}.png', img)

    def get_court_mask(self, mask_type=0):
        """
        Generate and return a mask based on the selected mask_type:
          0: full court mask
          1: bottom half only
          2: top half only
          3: court-only (no margins)
        """
        mask = np.ones_like(self.court_img)
        if mask_type == 1:  # Bottom half court
            mask[:self.center_net[0][1] - 1000, :] = 0
        elif mask_type == 2:  # Top half court
            mask[self.center_net[0][1]:, :] = 0
        elif mask_type == 3: # court without margins
            mask[:self.top_line[0][1], :] = 0
            mask[self.bottom_line[0][1]:, :] = 0
            mask[:, :self.left_court_border[0][0]] = 0
            mask[:, self.right_court_border[0][0]:] = 0
        return mask


if __name__ == '__main__':
    # Instantiate the TennisCourtModel, then construct and save the reference court image.
    tennis_court = TennisCourtModel()
    tennis_court.construct_court_model()

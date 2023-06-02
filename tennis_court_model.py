import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TennisCourtModel:
    """
    Tennis Court reference model
    """
    def __init__(self):
        # initialising points for image configuration
        self.top_line = ((286, 561), (1379, 561))
        self.bottom_line = ((286, 2935), (1379, 2935))
        self.center_net = ((286, 1748), (1379, 1748))
        self.left_court_border = ((286, 561), (286, 2935))
        self.right_court_border = ((1379, 561), (1379, 2935))
        self.left_inner_border = ((423, 561), (423, 2935))
        self.right_inner_border = ((1242, 561), (1242, 2935))
        self.center_line = ((832, 1110), (832, 2386))
        self.top_inner_border = ((423, 1110), (1242, 1110))
        self.bottom_inner_border = ((423, 2386), (1242, 2386))
        self.top_extension = (832.5, 580)
        self.bottom_extension = (832.5, 2910)

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

        self.raw_dataframe_points = ["top_left",
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
                                    "middle_bottom_right"]

        self.court_img = cv2.cvtColor(cv2.imread('court_configurations/court_reference.png'), cv2.COLOR_BGR2GRAY)

    def prepare_court_configs(self):
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
        ground_truth_pixels = {

            "top_left" :            [(161,82), (596,237), (145,235)],
            "top_right" :           [(450,82), (1373,239), (428,105)],
            "bottom_left" :         [(70,323), (276,788), (896,563)],
            "bottom_right" :        [(542,323), (1689,798), (1054,271)],
            "middle_left" :         [(126,173), (484,432), (425,362)],
            "middle_right" :        [(485,173), (1482,435), (688,172)],
            "left_middle_top" :     [(198,82), (693,238), (188,215)],
            "left_middle_bottom" :  [(128,324), (455,789), (924,512)],
            "right_middle_top" :    [(414,82), (1275,240), (400,118)],
            "right_middle_bottom" : [(483,323), (1506,796), (1041,296)],
            "middle_top" :          [(306,119), (982,318), (416,198)],
            "middle_bottom" :       [(306,244), (980,596), (777,318)],
            "second_middle_left" :  [(187,119), (660,316), (301,261)],
            "second_middle_right" : [(425,119), (1306,318), (508,148)],
            "middle_bottom_left" :  [(151,244), (539,594), (683,417)],
            "middle_bottom_right" : [(461,244), (1423,599), (849,242)]
        }
        ground_truth_dataset = pd.DataFrame(ground_truth_pixels)
        ground_truth_dataset = self.preprocess_dataset(ground_truth_dataset)
        
        return ground_truth_dataset
    

    def preprocess_dataset(self, dataset, transpose = True):
        # if ground truth icin
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
        # Creating the court reference image based on the line positions
        court_img = np.zeros((self.court_h + 2 * self.top_bottom_padding, 
                              self.court_w + 2 * self.right_left_padding), dtype=np.uint8)
        self.draw_lines(court_img)
        court_img = cv2.dilate(court_img, np.ones((5, 5), dtype=np.uint8))
        plt.imsave('court_configurations/court_reference.png', court_img, cmap='gray')
        self.court_img = court_img
        return court_img

    def draw_lines(self, img):
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
        Returns all lines of the court
        """
        all_lines = [*self.top_line, *self.bottom_line, *self.center_net, *self.left_court_border, 
                     *self.right_court_border, *self.left_inner_border, *self.right_inner_border, 
                     *self.center_line, *self.top_inner_border, *self.bottom_inner_border]
        return all_lines

    def get_extension_parts(self):
        return [self.top_extension, self.bottom_extension]

    def save_court_configs(self):
        """
        Create all configurations of 4 points on court reference
        """
        for i, config in self.court_configurations.items():
            img = cv2.cvtColor(255 - self.court_img, cv2.COLOR_GRAY2BGR)
            for point in config:
                img = cv2.circle(img, point, 15, (0, 0, 255), 30)
            cv2.imwrite(f'court_configurations/court_conf_{i}.png', img)

    def get_court_mask(self, mask_type=0):
        """
        Get mask of the court
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
    tennis_court = TennisCourtModel()
    tennis_court.construct_court_model()

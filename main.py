from regularisation import read_csv, plot, part_wise_plot
from regularisation import detect_straight_line
from regularisation import detect_rectangle
from regularisation import detect_circle_or_ellipse
from regularisation import detect_star_shape
from regularisation import detect_regular_polygon

import numpy as np
import matplotlib.pyplot as plt

from occlusion import reflect_curve
from occlusion import calculate_symmetry_score
from occlusion import check_continuity_and_derivatives
from occlusion import mirror_points
from occlusion import find_intersections
from occlusion import shift_symmetry_line

import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"



# parsing
parser = argparse.ArgumentParser(description="Adobe")
parser.add_argument('--task', type=str, default='regularisation')
parser.add_argument('--path', type=str, default='dataset\\isolated.csv')


args = parser.parse_args()
task = args.task
path = args.path

outputfile = np.empty((0, 4), dtype=np.float64)
shape_num = 0

if task=='regularisation':
    paths_XYs = read_csv(path)
    plot(paths_XYs)
    final_shapes = []
    for XYs in paths_XYs:
        for XY in XYs:
            XYs = np.array(XY).squeeze()
            x = XYs[:, 0]
            y = XYs[:, 1]
            range_val = max(np.ptp(x), np.ptp(y))
            print(f"Currently values range is {range_val}")
            
            if(detect_straight_line(x, y, threshold=0.1*range_val)[0]==True):
                print("The file contain a straight line!!")
                value = detect_straight_line(x, y, threshold=0.1*range_val)
                loss = value[1]
                x = value[2]
                y = value[3]
            
            elif(detect_circle_or_ellipse(x, y, threshold=0.1*range_val)[0]==True):
                print("The file contains an ellipse!!")
                value = detect_circle_or_ellipse(x, y, threshold=0.1*range_val)
                loss = value[1]
                x = value[2]
                y = value[3]
            
            elif(detect_rectangle(x, y, threshold=0.6*range_val, tolerance=0.1*range_val)[0]==True):
                print("The file contains a rectangle!!")
                value = detect_rectangle(x, y, threshold=0.6*range_val, tolerance=0.1*range_val)
                loss = value[1]
                x = value[2]
                y = value[3]

            elif(detect_star_shape(x, y, threshold=0.1*range_val)[0]==True):
                print("The file contains a star!!")
                value = detect_star_shape(x, y, threshold=0.1*range_val)
                loss = value[1]
                x = value[2]
                y = value[3]

            elif(detect_regular_polygon(x, y, threshold=1000)[0]==True):
                print("The file contains a regular polygon!!")
                value = detect_regular_polygon(x, y, threshold=1000)
                loss = value[1]
                x = value[2]
                y = value[3]
                
            # appending the values in final_shapes    
            final_shape = np.column_stack((x, y))
            final_shapes.append([final_shape])

        # print(len(final_shape))
        zeros_column = np.zeros((len(final_shape), 1), dtype=np.float64)
        integers_column = np.full((len(final_shape), 1), float(shape_num), dtype=np.float64)
        shape_num += 1
        stacked_array = np.hstack((integers_column, zeros_column, np.array(final_shape, dtype=np.float64)))
        outputfile = np.vstack((outputfile, stacked_array), dtype=np.float64)

    # plotting the graph
    plot(final_shapes)  
    
    
if task == 'occlusion':
    paths_XYs = read_csv(path)
    plot(paths_XYs)
    final_shapes = []
    for XYs in paths_XYs:
        for XY in XYs:
            XYs= np.array(XY).squeeze()
            x =XY[:, 0]
            y =XY[:, 1]
            
            
            # calculating the centroid
            centroid_x, centroid_y = np.mean(x), np.mean(y)
            
            # generating the variable theta for finding best angle of symmetry
            angles = np.linspace(0, np.pi, 180)
            
            # finding symmetry axis
            best_axis = None
            best_score = np.inf
            
            for theta in angles:
                m = np.tan(theta)
                c = centroid_y - m*centroid_x
                
                # reflecting the curve for comparison
                x_reflected, y_reflected = reflect_curve(x, y, m, c)
                
                score = calculate_symmetry_score(x, y, x_reflected, y_reflected)
                if score < best_score:
                    best_score = score
                    best_axis = (m, c)
            
            m, c = best_axis
            
            
            '''
            Shifitng the symmetry line so that we can find the best line about which
            symmetry can be applied and then mirror the non occluded part
            '''
            # shifting the line
            m, c = shift_symmetry_line(x, y, m, c)
            
            # Divide the points into left and right based on the symmetry line
            left_side = []
            right_side = []

            for xi, yi in zip(x, y):
                y_line = m * xi + c
                if yi < y_line:
                    left_side.append((xi, yi))
                else:
                    right_side.append((xi, yi))
                    
            # Check continuity and derivatives on each side
            left_gaps, left_derivs = check_continuity_and_derivatives(left_side)
            right_gaps, right_derivs = check_continuity_and_derivatives(right_side)
            
            # Deciding which side is occluded
            if len(left_gaps) > len(right_gaps) or len(left_derivs) > len(right_derivs):
                occluded_side = left_side
                non_occluded_side = right_side
            else:
                occluded_side = right_side
                non_occluded_side = left_side
            
            mirrored_points = mirror_points(non_occluded_side, m, c)
            
            # Combine the mirrored points with the non-occluded side to complete the curve
            completed_curve_x = np.concatenate([np.array(non_occluded_side)[:, 0], mirrored_points[:, 0]])
            completed_curve_y = np.concatenate([np.array(non_occluded_side)[:, 1], mirrored_points[:, 1]])
            
            # appending the values in final_shapes    
            final_shape = np.column_stack((completed_curve_x, completed_curve_y))
            final_shapes.append([final_shape])

        # print(final_shapes.shape)
        zeros_column = np.zeros((len(final_shape), 1), dtype=np.float64)
        integers_column = np.full((len(final_shape), 1), float(shape_num), dtype=np.float64)
        shape_num += 1
        stacked_array = np.hstack((integers_column, zeros_column, np.array(final_shape, dtype=np.float64)))
        outputfile = np.vstack((outputfile, stacked_array), dtype=np.float64)
    # plotting the graph
    plot(final_shapes)

    

    from evaluation import polylines2svg
    polylines2svg(final_shapes, "output.svg")

csv_file_path = 'output\\outputfile.csv'  
np.savetxt(csv_file_path, outputfile, delimiter=',', fmt='%f', comments='')

print(f"Output file saved as {csv_file_path}")
path_out = read_csv(csv_file_path)
plot(path_out)
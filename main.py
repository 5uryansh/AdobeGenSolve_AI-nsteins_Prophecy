from regularisation import read_csv, plot, part_wise_plot
from regularisation import detect_straight_line
from regularisation import detect_rectangle
from regularisation import detect_circle_or_ellipse
from regularisation import detect_star_shape
from regularisation import detect_regular_polygon

import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style

from occlusion import reflect_curve
from occlusion import calculate_symmetry_score
from occlusion import check_continuity_and_derivatives
from occlusion import mirror_points
from occlusion import find_intersections
from occlusion import shift_symmetry_line

from fragmented import update_frag_with_segments
from fragmented import process_segments
from fragmented import angle_based_line_check
from fragmented import process_and_merge_segments_with_graph
from fragmented import find_cycles_in_graph
from fragmented import process_and_merge_corners
from fragmented import find_all_cycle_combinations
from fragmented import seperate_cycles
from fragmented import defaultdict

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
print(Fore.YELLOW, "Reading the input file....")
print(Style.RESET_ALL)

if task=='regularisation':
    paths_XYs = read_csv(path)
    final_shapes = []
    for XYs in paths_XYs:
        for XY in XYs:
            XYs = np.array(XY).squeeze()
            x = XYs[:, 0]
            y = XYs[:, 1]
            range_val = max(np.ptp(x), np.ptp(y))
            
            if(detect_straight_line(x, y, threshold=0.1*range_val)[0]==True):
                print(Fore.GREEN, "The file contain a straight line!!")
                value = detect_straight_line(x, y, threshold=0.1*range_val)
                loss = value[1]
                x = value[2]
                y = value[3]
            
            elif(detect_circle_or_ellipse(x, y, threshold=0.1*range_val)[0]==True):
                print(Fore.GREEN, "The file contains an ellipse!!")
                value = detect_circle_or_ellipse(x, y, threshold=0.1*range_val)
                loss = value[1]
                x = value[2]
                y = value[3]
            
            elif(detect_rectangle(x, y, threshold=0.6*range_val, tolerance=0.1*range_val)[0]==True):
                print(Fore.GREEN, "The file contains a rectangle!!")
                value = detect_rectangle(x, y, threshold=0.6*range_val, tolerance=0.1*range_val)
                loss = value[1]
                x = value[2]
                y = value[3]

            elif(detect_star_shape(x, y, threshold=0.1*range_val)[0]==True):
                print(Fore.GREEN, "The file contains a star!!")
                value = detect_star_shape(x, y, threshold=0.1*range_val)
                loss = value[1]
                x = value[2]
                y = value[3]

            elif(detect_regular_polygon(x, y, threshold=1000)[0]==True):
                print(Fore.GREEN, "The file contains a regular polygon!!")
                value = detect_regular_polygon(x, y, threshold=1000)
                loss = value[1]
                x = value[2]
                y = value[3]
                
            ## printing symmetry lines
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
            print(Fore.BLUE, f"The symmetry line 'y=m*x+c' has m as {m} and c as {c}")
            if(abs(m)>=0 and abs(m)<0.577):
                print("Horizontal line of symmetry.")
            if(abs(m)>=0.577 and abs(m)<1.732):
                print("Diagonal line of symmetry.")
            if(abs(m)>=1.732):
                print("Vertical line of symmetry.")
            print(Style.RESET_ALL)
            
            # appending the values in final_shapes    
            final_shape = np.column_stack((x, y))
            final_shapes.append([final_shape])

        # print(len(final_shape))
        zeros_column = np.zeros((len(final_shape), 1), dtype=np.float64)
        integers_column = np.full((len(final_shape), 1), float(shape_num), dtype=np.float64)
        shape_num += 1
        stacked_array = np.hstack((integers_column, zeros_column, np.array(final_shape, dtype=np.float64)))
        outputfile = np.vstack((outputfile, stacked_array), dtype=np.float64)

    
    
if task == 'occlusion':
    paths_XYs = read_csv(path)
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
            print(Fore.BLUE, f"The symmetry line 'y=m*x+c' has m as {m} and c as {c}")
            if(abs(m)>=0 and abs(m)<0.577):
                print("Horizontal line of symmetry.")
            if(abs(m)>=0.577 and abs(m)<1.732):
                print("Diagonal line of symmetry.")
            if(abs(m)>=1.732):
                print("Vertical line of symmetry.")
            print(Style.RESET_ALL)
            
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
            if len(left_gaps) < len(right_gaps) or len(left_derivs) < len(right_derivs):
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

            zeros_column = np.zeros((len(final_shape), 1), dtype=np.float64)
            integers_column = np.full((len(final_shape), 1), float(shape_num), dtype=np.float64)
            shape_num = shape_num + 1
            stacked_array = np.hstack((integers_column, zeros_column, np.array(final_shape, dtype=np.float64)))
            outputfile = np.vstack((outputfile, stacked_array), dtype=np.float64)



if task == 'fragmented':
    isolated = read_csv(path)
    newfrag = update_frag_with_segments(isolated, curvature_threshold=0.2)
    new_frag_with_lines = process_segments(newfrag)
    is_straight_line = [angle_based_line_check(segment[0], 0.1, 0.1) for segment in newfrag]
    # Process the new fragments to merge and convert to straight lines where applicable
    merged_frag_with_lines = process_and_merge_segments_with_graph(new_frag_with_lines, is_straight_line)
    # Process and merge corners
    merged_frag_with_lines = process_and_merge_corners(merged_frag_with_lines, is_straight_line)

    # Step 1: Maintain a map for all the segments between two coordinate points.
    segment_map = defaultdict(list)
    for idx, segment in enumerate(merged_frag_with_lines):
        start = tuple(segment[0][0])
        end = tuple(segment[0][-1])
        segment_map[(start, end)].append(idx)
        # segment_map[(end, start)].append(idx)  # Include reverse for undirected graph

    # Create a mapping for nodes to 1-based indices
    nodes = set()
    for start, end in segment_map:
        nodes.add(start)
        nodes.add(end)

    node_list = list(nodes)
    node_index = {node: idx for idx, node in enumerate(node_list)}  # 1-based index

    # Prepare the graph representation
    N = len(node_list)  # Number of nodes
    graph = [[] for _ in range(N)]  # 1-based indexing

    for (start, end), indices in segment_map.items():
        u = node_index[start]
        v = node_index[end]
        graph[u].append(v)
        graph[v].append(u)

    cycles = find_cycles_in_graph(graph, N)
    # Print all cycle combinations
    cycle_combinations = find_all_cycle_combinations(cycles, segment_map, N, node_list)

    final_coordinates = seperate_cycles(cycle_combinations,merged_frag_with_lines, is_straight_line)
    # print(final_coordinates)
    for i in range(0, len(final_coordinates)):
        final_shape = final_coordinates[i]
        zeros_column = np.zeros((len(final_shape), 1), dtype=np.float64)
        integers_column = np.full((len(final_shape), 1), float(i), dtype=np.float64)
        stacked_array = np.hstack((integers_column, zeros_column, np.array(final_shape, dtype=np.float64)))
        outputfile = np.vstack((outputfile, stacked_array), dtype=np.float64)
        


csv_file_path = 'output\\outputfile.csv'  
np.savetxt(csv_file_path, outputfile, delimiter=',', fmt='%f', comments='')

print(Fore.RED, f"Output file saved as {csv_file_path}")
print(Style.RESET_ALL)

#plotting the input image
plot(read_csv(path), "Input Figure")

# plotting the final output
plot(read_csv(csv_file_path), "Output Figure")

if task == 'occlusion':
    from evaluation import polylines2svg
    polylines2svg(final_shapes, "output.svg")
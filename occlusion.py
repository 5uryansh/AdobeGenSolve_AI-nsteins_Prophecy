import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs


def plot(paths_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.show()
    

# Reflect each point (x, y) across a line y = mx + c
def reflect_curve(x, y, m, c):
    x_reflected = []
    y_reflected = []

    for xi, yi in zip(x, y):
        # Reflection formula across a line y = mx + c
        d = (xi + (yi - c) * m) / (1 + m ** 2)
        x_ref = 2 * d - xi
        y_ref = 2 * d * m - yi + 2 * c
        x_reflected.append(x_ref)
        y_reflected.append(y_ref)

    return np.array(x_reflected), np.array(y_reflected)


# Calculate symmetry score (mean squared distance between original and reflected points)
def calculate_symmetry_score(x, y, x_reflected, y_reflected):
    distances = np.sqrt((x - x_reflected) ** 2 + (y - y_reflected) ** 2)
    return np.mean(distances)


# Function to check continuity and derivative on one side
def check_continuity_and_derivatives(points):
    points = np.array(points)
    distances = np.sqrt(np.diff(points[:, 0])**2 + np.diff(points[:, 1])**2)
    derivatives = np.diff(points[:, 1]) / np.diff(points[:, 0])
    
    avg_distance = np.mean(distances)
    large_gaps = np.where(distances > avg_distance * 1.5)[0]  # 1.5 is a threshold
    large_derivatives = np.where(np.abs(np.diff(derivatives)) > 1.0)[0]  # 1.0 is a threshold
    
    return large_gaps, large_derivatives


# Mirror the non-occluded side
def mirror_points(points, m, c):
    mirrored_points = []
    for xi, yi in points:
        d = (xi + (yi - c) * m) / (1 + m**2)
        x_mirrored = 2*d - xi
        y_mirrored = 2*d*m - yi + 2*c
        mirrored_points.append((x_mirrored, y_mirrored))
    return np.array(mirrored_points)


def find_intersections(x, y, m, c):
    """ Find intersections between the curve and the symmetry line """
    intersections = []
    for i in range(len(x)-1):
        x1, y1 = x[i], y[i]
        x2, y2 = x[i+1], y[i+1]
        
        # Line equation for curve segment: y = m_curve * x + b_curve
        if x2 - x1 == 0:
            continue
        m_curve = (y2 - y1) / (x2 - x1)
        b_curve = y1 - m_curve * x1
        
        # Find intersection with symmetry line y = m*x + c
        if m_curve - m != 0:
            x_intersect = (c - b_curve) / (m_curve - m)
            y_intersect = m * x_intersect + c
            if min(x1, x2) <= x_intersect <= max(x1, x2):
                intersections.append((x_intersect, y_intersect))
    
    return intersections


def shift_symmetry_line(x, y, m, c, step=1, max_shifts=100):
    max_distance = 0
    final_m, final_c = m, c
    
    # Explore shifts in both directions: positive and negative
    for direction in [-1, 1]:
        for i in range(max_shifts):
            # Shift the line by modifying the intercept
            c_shifted = c + direction * i * step
            
            # Find the points where the curve intersects the shifted symmetry line
            intersections = find_intersections(x, y, m, c_shifted)
            
            if len(intersections) < 2:
                continue
            
            # Calculate the distance between the two extreme intersecting points
            dist = euclidean(intersections[0], intersections[-1])
            
            if dist > max_distance:
                max_distance = dist
                final_m, final_c = m, c_shifted
            else:
                # If the distance starts to decrease, stop further shifting in this direction
                break
    
    return final_m, final_c
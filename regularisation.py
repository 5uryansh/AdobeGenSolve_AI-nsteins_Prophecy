import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import minimize
   

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    # print(np_path_XYs)
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs


def plot(paths_XYs, title_for_graph):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.title(title_for_graph)
    plt.show()
    

def part_wise_plot(index, size=4):
    index = np.array(index).squeeze()
    x = index[:, 0]
    y = index[:, 1]
    
    # Plotting the points
    plt.figure(figsize=(size, size))
    plt.plot(x, y, linestyle='-')
    plt.title('Plot of Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    

# Code for error calculation
def calculate_error(params, x, y, vertical=False):
    if vertical:
        # If the line is vertical, the error is based on the difference in x-values
        c = params[0]
        return np.mean((x - c) ** 2)
    else:
        # For non-vertical lines, the error is based on y-values
        m, c = params
        y_fit = m * x + c
        return np.mean((y - y_fit) ** 2)

# code for detection of straight line
def detect_straight_line(x, y, threshold=1.0, epsilon=1e-8):
    delta_x = x[-1] - x[0]
    delta_y = y[-1] - y[0]
    
    if delta_x == 0:
        delta_x += epsilon
    
    m_initial = delta_y / delta_x
    c_initial = y[0] - m_initial * x[0]
    
    if m_initial > 11.4:
        avg_x = (x[0]+x[-1])/2
        new_x = []
        for i in range(0, len(x)):
            new_x.append(avg_x)
        
        new_x = np.array(new_x)
        loss = np.mean((x-new_x) ** 2)
        loss = loss ** 0.5
        return loss<threshold, loss, new_x, y

    # If the slope is very large (close to infinity), we treat it as a vertical line
    if abs(m_initial) > 1e6:
        vertical = True
        result = minimize(calculate_error, [x[0]], args=(x, y, vertical))
        c_optimized = result.x[0]
        final_error = calculate_error([c_optimized], x, y, vertical)
        y_fit = np.full_like(y, c_optimized)  # y-fit doesn't matter in the vertical line case
        x_fit = np.full_like(x, c_optimized)  # x is constant for the vertical line
    else:
        vertical = False
        result = minimize(calculate_error, [m_initial, c_initial], args=(x, y, vertical))
        m_optimized, c_optimized = result.x
        y_fit = m_optimized * x + c_optimized
        x_fit = x
        final_error = calculate_error(result.x, x, y, vertical)

    # Calculate the final error (loss)
    loss = calculate_error(result.x, x, y)

    # Check if the final error is below the threshold'
    return loss<threshold, loss, x, y_fit
 


## Checking for circle and ellipse
def detect_circle_or_ellipse(x, y, threshold=9.0):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    distances = np.sqrt((x - x_mean) ** 2 + (y - y_mean) ** 2)
    radius_mean = np.mean(distances)
    loss = np.mean((distances - radius_mean) ** 2)

    if loss < threshold:
        angles = np.arctan2(y - y_mean, x - x_mean)
        x = x_mean + radius_mean * np.cos(angles)
        y = y_mean + radius_mean * np.sin(angles)
    return loss < threshold, loss, x, y


## Checking for local maximas
def find_maximums(nums):
    n = len(nums)
    local_maximas = []
    for i in range(1, n-1):
        if nums[i] > nums[i-1] and nums[i] > nums[i+1]:
            local_maximas.append(i)

    if (nums[1] - nums[0]) * (nums[n-1] - nums[n-2]) < 0:
        if nums[0] > nums[n-1]:
            local_maximas = [0] + local_maximas
        else:
            local_maximas = local_maximas + [n-1]

    return local_maximas

## Checking for local minimas
def find_minimas(nums):
    n = len(nums)
    local_minimas = []

    for i in range(1, n-1):
        if nums[i] <= nums[i-1] and nums[i] <= nums[i+1]:
            local_minimas.append(i)

    if (nums[1] - nums[0]) >=0 and (nums[n-1] - nums[n-2]) <= 0:
        if nums[0] < nums[n-1]:
            local_minimas = [0] + local_minimas
        else:
            local_minimas = local_minimas + [n-1]

    return local_minimas


## Checking for regular polygon
def detect_regular_polygon(x, y, threshold=0.1):
    # Calculate centroid
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)
    
    # Calculate radii
    radii = np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
    
    # Detect peaks (potential vertices of the polygon)
    peaks = find_maximums(radii)
    
    # Convert peaks to a NumPy array for advanced indexing
    peaks = np.array(peaks)
    
    # Calculate angles from centroid to each peak
    angles = np.arctan2(y[peaks] - centroid_y, x[peaks] - centroid_x)
    angles_sorted_indices = np.argsort(angles)
    peaks = peaks[angles_sorted_indices]
    
    # Calculate the expected angles for a regular polygon
    num_sides = len(peaks)
    expected_angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    
    # Calculate expected vertices positions based on centroid and average radius
    avg_radius = np.mean(radii[peaks])
    expected_x = centroid_x + avg_radius * np.cos(expected_angles)
    expected_y = centroid_y + avg_radius * np.sin(expected_angles)
    
    # Calculate loss as the sum of distances between detected and expected vertices
    loss = np.sum(np.sqrt((x[peaks] - expected_x)**2 + (y[peaks] - expected_y)**2))
    
    # Normalize loss by the number of sides
    loss /= num_sides
    
    # Regularize the original polygon if loss is below the threshold
    if loss < threshold:
        new_x = []
        new_y = []
        
        for i in range(num_sides):
            next_index = (i + 1) % num_sides
            t = np.linspace(0, 1, len(x)//num_sides)
            new_x[i*len(t):(i+1)*len(t)] = x[peaks[i]] * (1 - t) + x[peaks[next_index]] * t
            new_y[i*len(t):(i+1)*len(t)] = y[peaks[i]] * (1 - t) + y[peaks[next_index]] * t
        
        return loss < threshold, loss, new_x, new_y
    
    return loss < threshold, loss, x, y


# Detect star shape
def detect_star_shape(x, y, threshold=2.0):
    # Calculate centroid
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)

    # Calculate radii
    radii = np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)

    # Find peaks (local maxima) and troughs (local minima)
    peaks = find_maximums(radii)
    troughs = find_minimas(radii)
    # print(peaks)
    # print(troughs)

    # Check for alternating pattern and calculate loss
    peak_diffs = np.diff(peaks)
    trough_diffs = np.diff(troughs)

    # A simple heuristic for loss: sum of absolute differences in peak and trough distances
    loss = np.sum(np.abs(peak_diffs - np.mean(peak_diffs))) + np.sum(np.abs(trough_diffs - np.mean(trough_diffs)))

    # Normalize loss
    loss /= (len(peaks) + len(troughs))

    # Count peaks and troughs
    num_peaks = len(peaks)
    num_troughs = len(troughs)

    # Regularize the star shape if loss is below the threshold
    if loss < threshold:
        # Combine peaks and troughs and sort them by angle from the centroid
        combined = list(peaks) + list(troughs)
        angles = np.arctan2(y[combined] - centroid_y, x[combined] - centroid_x)
        sorted_indices = np.argsort(angles)
        sorted_combined = np.array(combined)[sorted_indices]

        # Initialize new coordinates
        new_x = np.zeros(len(sorted_combined))
        new_y = np.zeros(len(sorted_combined))

        # Connect peaks and troughs alternately
        for i in range(len(sorted_combined)):
            current_index = sorted_combined[i]
            if i % 2 == 0:  # If current point is a peak
                min_dist = float('inf')
                next_index = current_index
                for trough_index in sorted_combined:
                    if trough_index != current_index:
                        dist = np.sqrt((x[current_index] - x[trough_index])**2 + (y[current_index] - y[trough_index])**2)
                        if dist < min_dist:
                            min_dist = dist
                            next_index = trough_index
            # If current point is a trough
            else:
                min_dist = float('inf')
                next_index = current_index
                for peak_index in sorted_combined:
                    if peak_index != current_index:
                        dist = np.sqrt((x[current_index] - x[peak_index])**2 + (y[current_index] - y[peak_index])**2)
                        if dist < min_dist:
                            min_dist = dist
                            next_index = peak_index

            # Fit x and y values to the corresponding line (star side)
            t = np.linspace(0, 1, len(x))
            new_x[i] = x[current_index] * (1 - t[i]) + x[next_index] * t[i]
            new_y[i] = y[current_index] * (1 - t[i]) + y[next_index] * t[i]

        # Ensure closure by connecting the last point to the first point
        last_index = sorted_combined[-1]
        first_index = sorted_combined[0]
        new_x[-1] = x[last_index]
        new_y[-1] = y[last_index]
        new_x = np.append(new_x, new_x[0])
        new_y = np.append(new_y, new_y[0])

        return loss < threshold, loss, new_x, new_y

    return loss < threshold, loss, x, y


# Detection of rectangle
def detect_rectangle(x, y, threshold=100.0, tolerance=10.0):
    # Check for the number of points
    if len(x) != len(y) or len(x) < 4:
        return False, float('inf')

    # Find the bounding box
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    # Calculate the expected side lengths
    width = max_x - min_x
    height = max_y - min_y

    # Check if all points are on the bounding box edges
    is_on_edge = np.logical_or(
        np.logical_or(np.isclose(x, min_x, atol=tolerance), np.isclose(x, max_x, atol=tolerance)),
        np.logical_or(np.isclose(y, min_y, atol=tolerance), np.isclose(y, max_y, atol=tolerance))
    )

    if not np.all(is_on_edge):
        return False, float('inf')

    # Calculate the loss as the maximum deviation from the expected side lengths
    loss = np.max([
        np.max(np.abs(x[np.isclose(y, min_y)] - min_x)),
        np.max(np.abs(x[np.isclose(y, max_y)] - min_x)),
        np.max(np.abs(y[np.isclose(x, min_x)] - min_y)),
        np.max(np.abs(y[np.isclose(x, max_x)] - min_y)),
    ])

    if loss <= threshold:
        new_x = np.copy(x)
        new_y = np.copy(y)
        new_x[np.isclose(x, min_x, atol=tolerance)] = min_x
        new_x[np.isclose(x, max_x, atol=tolerance)] = max_x
        new_y[np.isclose(y, min_y, atol=tolerance)] = min_y
        new_y[np.isclose(y, max_y, atol=tolerance)] = max_y
        return loss <= threshold, loss, new_x, new_y

    return loss <= threshold, loss



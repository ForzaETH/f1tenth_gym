import os
import numpy as np
from pyglet.gl import GL_POINTS
from pyglet import shapes
import yaml
import subprocess

def read_config(path):
    with open(path) as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
    return conf

def downsample_points_distance_based(points, min_distance):
    """
    Downsamples points based on a minimum distance criterion.

    Parameters:
    points (numpy.ndarray): Array of points with shape (N_points, 2).
    min_distance (float): Minimum distance between consecutive points.

    Returns:
    numpy.ndarray: Downsampled array of points.
    """
    if len(points) == 0:
        return np.array([])

    # Start with the first point
    downsampled_points = [points[0]]

    for point in points:
        if np.linalg.norm(point - downsampled_points[-1]) >= min_distance:
            downsampled_points.append(point)

    return np.array(downsampled_points)

def downsample_points_simple(points, interval):
    """
    Downsamples points by selecting every nth point.

    Parameters:
    points (list of tuples): List of (x, y) points.
    interval (int): Interval for downsampling (every nth point).

    Returns:
    list of tuples: Downsampled list of points.
    """
    return points[::interval]

def ensure_absolute_path(path):
    if not path.startswith(os.sep):
        path = os.sep + path
    return path

def render_callback(env_renderer):
    # custom extra drawing function
    e = env_renderer

    # update camera to follow car
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.score_label.x = left
    e.score_label.y = top - 700
    e.left = left - 800
    e.right = right + 800
    e.top = top + 800
    e.bottom = bottom - 800

def render_single_point(env, point_coordinates, color_rgb_list):
    point_coordinates_scaled = 50.*point_coordinates
    env.batch.add(1, GL_POINTS, None, ('v3f/stream', [point_coordinates_scaled[0], point_coordinates_scaled[1], 0.]),
                                ('c3B/stream', color_rgb_list))

def moving_average(values, window):
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid")

def get_package_location(package_name):
    # Run pip show and capture the output
    result = subprocess.run(['pip', 'show', package_name], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error:", result.stderr)
        return None

    # Parse the output to find the location
    lines = result.stdout.split('\n')
    location_line = next((line for line in lines if line.startswith('Location:')), None)
    if location_line:
        return location_line.split(':', 1)[1].strip()

    return None

class Traj:
    def __init__(self, N, batch, clr=(255, 100, 222), r=5) -> None:
        self.points = [shapes.Circle(0, 0, r, color=clr, batch=batch) for i in range(N)]

    def set_points(self, points):
        for point, tr_point in zip(self.points, points):
            point.x = 50 * tr_point[0]
            point.y = 50 * tr_point[1]
            point.draw()
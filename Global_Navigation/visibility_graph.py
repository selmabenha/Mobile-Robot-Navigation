# Import necessary libraries
import os
import math
import pyvisgraph as vg
import numpy as np
from PIL import Image, ImageDraw
import time
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Record the start time
start_time = time.time()

# Set the script directory as the current working directory
script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
print(f"Current working directory: {os.getcwd()}")


# Function to convert polygon list format
def convert_polys_list(poly_list):
    out_poly = []
    for i in range(len(poly_list)):
        out_poly.append([])  # Append a new list for each polygon
        for j in range(len(poly_list[i])):
            out_poly[i].append(vg.Point(poly_list[i][j][0], poly_list[i][j][1]))
    print(f"\nPOLYGONS:\n{out_poly}\n---------------------")
    return out_poly

# Function to convert path to list format
def convert_path_to_list(list, start, goal):
    path = [start]
    for i in range(0, len(list) - 1):
        path.append((list[i].x, list[i].y))
    print(f"---------------------\nConverting path to list:\n{path}\n")
    path.append(goal)
    return path

# Function to calculate distance between two points
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to calculate "incenter" of an obstacle
def incenter(vertices):
    x_coords, y_coords = zip(*vertices)
    num_points = len(vertices)

    # Calculate lengths of sides opposite each vertex
    side_lengths = [distance(vertices[i], vertices[(i + 1) % num_points]) for i in range(num_points)]

    # Calculate incenter coordinates
    incenter_x = sum(side_lengths[i] * x_coords[i] for i in range(num_points)) / sum(side_lengths)
    incenter_y = sum(side_lengths[i] * y_coords[i] for i in range(num_points)) / sum(side_lengths)

    return incenter_x, incenter_y

# Function to resize polygons based on robot dimensions
def resize_polys(polys_list, robot_d):
    incenters = []
    polys_resized = []
    for i in range(0, len(polys_list)):
        polygon = polys_list[i]
        inc_x, inc_y = incenter(polygon)
        incenters.append((inc_x, inc_y))
        polys_resized.append([])
        for j in range(0, len(polygon)):
            point = polygon[j]
            dx = incenters[i][0] - point[0]
            dy = incenters[i][1] - point[1]
            d = distance(point, incenters[i])
            xp = point[0] - robot_d * (dx/d)
            yp = point[1] - robot_d * (dy/d)
            polys_resized[i].append((xp, yp))
    print(f"----------------\nRESIZED POLYGONS:\n{polys_resized}\n")
    return polys_resized, incenters

def merge_overlapping_polygons(polygons):
    # Convert the list of vertex coordinates to Shapely Polygon objects
    shapely_polygons = [Polygon(vertices) for vertices in polygons]

    # Initialize an empty list to store the merged polygons
    merged_polygons = []

    # Iterate through each polygon and check for overlaps
    for i in range(len(shapely_polygons)):
        current_polygon = shapely_polygons[i]
        merged = False

        # Check for overlaps with previously merged polygons
        for j in range(len(merged_polygons)):
            if current_polygon.intersects(merged_polygons[j]):
                # If there is an overlap, merge the polygons
                current_polygon = unary_union([current_polygon, merged_polygons[j]])
                merged_polygons[j] = current_polygon
                merged = True
                break

        # If no overlap was found, add the polygon to the list of merged polygons
        if not merged:
            merged_polygons.append(current_polygon)

    # Convert the final merged polygons back to the list of vertices
    result = [list(p.exterior.coords) for p in merged_polygons]

    return result

def calculate_angle(point1, point2):
    #Calculate angle (in degrees) between two points
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]
    return math.degrees(math.atan2(delta_y, delta_x))

def analyze_path(path):
    #Create a dictionary for each segment of the path with distance and angle
    path_data = []

    for i in range(len(path) - 1):
        segment_name = f"seg{i + 1}"
        point1, point2 = path[i], path[i + 1]
        angle = calculate_angle(point1, point2)
        if i == 0:
            path_data.append([(point2[0], point2[1]), 0])
        else:
            path_data.append([(point2[0], point2[1]), angle])
    print(f"-----------------\nDATA ABOUT THE PATH {segment_name}:\n{path_data}\n")
    return path_data

def global_navigation(polys_list, robot_dimension, start, goal):
    #PATH CALCULATION
    #Input
    #polys_list = list of the coordinates in  pixels of the vertices of each object
    #robot_dimension = dimension of the robot in pixels
    #Output
    #shortest_list: list of points the robot needs to pass by
    #path_data: it stores the angle and the distance of each segment of the path
    #polys_list_resized_merged: list of the vartices of the combined obstacles
    #incenters_list: list of the incenters of the obstacles
    polys_list_resized, incenters_list = resize_polys(polys_list, robot_dimension+15) #Considering the size of the robot
    polys_list_resized_merged = merge_overlapping_polygons(polys_list_resized) #Considering overlapping resized polygons
    polys = convert_polys_list(polys_list_resized_merged)

    # Build visibility graph
    g = vg.VisGraph()
    g.build(polys)

    # Find the shortest path
    shortest = g.shortest_path(vg.Point(start[0], start[1]), vg.Point(goal[0], goal[1]))
    
    #Converting the results to correct format
    shortest_list = convert_path_to_list(shortest, start, goal)
    path_data = analyze_path(shortest_list)

    # Record the end time
    end_time = time.time()
    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds\nAvaible frequency: {round(1/elapsed_time)} Hz\n")
    return path_data, polys_list_resized_merged, incenters_list

#Converts the polygons' dictionary into a list|
def convert_polygons(polys_dict):
    polys_list = []
    for key in polys_dict:
        polys_list.append(polys_dict[key]['Vertices'])
    return polys_list
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sci
import random as rd
import seaborn as sns

# Creating 100 random points
x = []
y = []
for _ in range(100):
    x.append(rd.randint(0, 101))
    y.append(rd.randint(0, 101))

# Plotting the points
sns.scatterplot(x = x, y = y)

# Creating the list of points
rnd_points = [(xx, yy) for xx, yy in zip(x,y)]

# We set the number of points
N_CENTROIDS = 3


# Creating random centroids 
centroid_x = []
centroid_y = []
for _ in range(N_CENTROIDS):
    centroid_x.append(rd.randint(0, 101))
    centroid_y.append(rd.randint(0, 101))


# Saving centroids information in a dictionary
def centroid_info(x_centroids, y_centroids, N_c):
    centroid_coords = {(x_c, y_c): i for i, x_c, y_c in zip(range(N_c), x_centroids, y_centroids)}

    return centroid_coords


rnd_centroids = centroid_info(centroid_x, centroid_y, N_CENTROIDS)


# Function to calculate the distance between two points of x-y coordinates
def distance(point_coords, centroid_coords): #_coords is a tuple of length 2
    sq_dist = 0
    for p, c in zip(point_coords, centroid_coords):
        sq_dist += (p - c)**2
    
    dist = np.sqrt(sq_dist)
    return dist


# Clusterizing the points to the random centroids 
def clusterize(points_list, centroids):
    clusterization = {}
    for point in points_list:
        distances = []
        for centroid in centroids.keys():
            distances.append(distance(point, centroid))
        
        centroid_label = distances.index(max(distances))
        clusterization = {**clusterization, **{point: centroid_label}}
    
    
    return clusterization


cluster = clusterize(rnd_points, rnd_centroids)


# 

sns.scatterplot(x = x, y = y, hue = cluster.values())


















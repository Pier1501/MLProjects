import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sci
import random as rd
import seaborn as sns

x = []
y = []
for _ in range(100):
    x.append(rd.randint(0, 101))
    y.append(rd.randint(0, 101))

sns.scatterplot(x = x, y = y)

rnd_points = [(xx, yy) for xx, yy in zip(x,y)]

# Number of centroids we want
N_CENTROIDS = 3

centroid_x = []
centroid_y = []
for _ in range(N_CENTROIDS):
    centroid_x.append(rd.randint(0, 101))
    centroid_y.append(rd.randint(0, 101))

def centroid_info(x_centroids, y_centroids, N_c):
    centroid_coords = {i: (x_c, y_c) for i, x_c, y_c in zip(range(N_c), x_centroids, y_centroids)}

    return centroid_coords

rnd_centroids = centroid_info(centroid_x, centroid_y, N_CENTROIDS)


def distance(point_coords, centroid_coords): #_coords is a tuple of length 2
    sq_dist = 0
    for p, c in zip(point_coords, centroid_coords):
        sq_dist += (p - c)**2
    
    dist = np.sqrt(sq_dist)
    return dist

def clusterize(points_list, centroids):
    for point in points_list:
        distances = []
        for centroid in centroids:
            distances.append(distance(point, centroid))
        
        centroid_label = distances.index(max(distances))
        clusterization = {point: centroid_label}
    
    return clusterization

cluster = clusterize(rnd_points, rnd_centroids)

number_of_points_of_cluster = {k : cluster.count(k) for k in cluster.values()}

sns.scatterplot(x = x, y = y, hue = cluster.values())


















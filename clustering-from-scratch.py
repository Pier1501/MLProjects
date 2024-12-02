import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sci
import random as rd
import seaborn as sns

# Defining the values for the clusterization
ERR = 10**(-5)
N_CENTROIDS = 3


# Creating the random points
x = []
y = []
for _ in range(100):
    x.append(rd.randint(0, 101))
    y.append(rd.randint(0, 101))

sns.scatterplot(x = x, y = y)


# We create the list of points
rnd_points = [(xx, yy) for xx, yy in zip(x,y)]


# We randomly choose the centroids
rnd_centroids = rd.sample(rnd_points, N_CENTROIDS)
x_centroids = [centroid[0] for centroid in rnd_centroids]
y_centroids = [centroid[1] for centroid in rnd_centroids]


# Function to calculate the distance between two points in the plane
def distance(xx, yy, cx, cy): 
    sq_dist = (xx - cx)**2 + (yy - cy)**2
    dist = np.sqrt(sq_dist)
    return dist


# Function to calculate the coordinates (weighted mean)
def find_centroid(coords):
    return sum(coords)/len(coords)


# Function to define the clusterization 
def clusterize(points_list, centroids):
    clusterization = []
    return clusterization

cluster = clusterize(rnd_points, rnd_centroids)

number_of_points_of_cluster = {k : cluster.count(k) for k in cluster.values()}

sns.scatterplot(x = x, y = y, hue = cluster.values())


















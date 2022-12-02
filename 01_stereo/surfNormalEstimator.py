#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:43:32 2022

This script will estimate the normal vectors to a point cloud by taking the PCA of points in batches of k size
that are close to eachother, then taking the least eigenvector regarding each point.

Input: Mx3 point cloud of x y z points separated by spaces
Output: ply_header + Mx6 point cloud of x y z nx ny nz points separated by spaces 

@author: daniel
"""
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits.mplot3d import Axes3D
import math
from math import pi
import os
import pandas as pd

# Header file to be attached to the beginning of a ply file
ply_header = """ply
format ascii 1.0
comment VCGLIB generated
element vertex 15171
property float x
property float y
property float z
property float nx
property float ny
property float nz
element face 0
property list uchar int vertex_indices
end_header
"""

def normaldefinition_3D_real(void_data, k):
    dist = distance.squareform(distance.pdist(void_data)) # Alternative (and more direct) form to calculate the distances
    closest = np.argsort(dist, axis = 1) # Axis = 1 because we are sorting the columns
    
    # Extraction of normals and centroids for each point 
    
    total_pts=np.size(closest,0)
    planes=np.zeros((total_pts,6)) # The three first columns contian the coordinates of the normals. The 3 last columns contain the coordinates of the centroid of the plane

    for i in range(total_pts):
        normal_vect, xmn,ymn,zmn, knn_pt_coord = tangentplane_3D_real(closest[i,:],void_data,k) #Obtention of the normal and centroid (and other parametres) for each point in the ellipsoid
        planes[i,0:3] = normal_vect #Keep the coordinates of the normal vectors
        planes[i,3:6] = np.array([xmn, ymn, zmn]) #Keep the coordinates of the centroid

    planes_consist = normalconsistency_3D_real(planes)
    
    return  planes, planes_consist
    
def tangentplane_3D_real(closest_pt,ellipsoid_data,k):
    knn_pt_id = closest_pt[0:k] # Retain only the indexes of the k-closest points
    nb_points = np.size(knn_pt_id)
    knn_pt_coord = np.zeros((nb_points,3)) 
    
    for i in range(nb_points):
        point_i = knn_pt_id[i]
        knn_pt_coord[i,:] = ellipsoid_data[point_i,:]

    xmn = np.mean(knn_pt_coord[:,0])
    ymn = np.mean(knn_pt_coord[:,1])
    zmn = np.mean(knn_pt_coord[:,2])
    
    c=np.zeros((np.size(knn_pt_coord,0),3))
    
    c[:,0] = knn_pt_coord[:,0]-xmn
    c[:,1] = knn_pt_coord[:,1]-ymn
    c[:,2] = knn_pt_coord[:,2]-zmn
    
    # Covariance matrix
    cov=np.zeros((3,3))    
    
    cov[0,0] = np.dot(c[:,0],c[:,0])
    cov[0,1] = np.dot(c[:,0],c[:,1])
    cov[0,2] = np.dot(c[:,0],c[:,2])
    
    cov[1,0] = cov[0,1]
    cov[1,1] = np.dot(c[:,1],c[:,1])
    cov[1,2] = np.dot(c[:,1],c[:,2])
    
    cov[2,0] = cov[0,2]
    cov[2,1] = cov[1,2]
    cov[2,2] = np.dot(c[:,2],c[:,2])
   
    u,s,vh = np.linalg.svd(cov) # U contains the orthonormal eigenvectors and S contains the eigenvectors    
    minevindex = np.argmin(s)
    normal_vect = u[:,minevindex]

    return normal_vect, xmn, ymn,zmn,knn_pt_coord

def normalconsistency_3D_real(planes):
    nbnormals = np.size(planes, 0)
    planes_consist=np.zeros((nbnormals,6))
    planes_consist[:, 3:6] = planes[:, 3:6] # We just copy the columns corresponding to the coordinates of the centroids (from 3th to 5th)
    
    sensorcentre=np.array([0,0,0])
    
    for i in range(nbnormals):    
        p1 = (sensorcentre - planes[i,3:6]) / np.linalg.norm(sensorcentre - planes[i,3:6]) # Vector from the centroid to the centre of the ellipsoid (here the sensor is placed)
        p2 = planes[i,0:3]
        
        angle = math.atan2(np.linalg.norm(np.cross(p1,p2)), np.dot(p1,p2) ) # Angle between the centroid-sensor and plane normal
       
        if (angle >= -pi/2 and angle <= pi/2): # (angle >= -pi/2 and angle <= pi/2):     
            planes_consist[i,0] = -planes[i,0]
            planes_consist[i,1] = -planes[i,1]
            planes_consist[i,2] = -planes[i,2]  
            
        else:
            planes_consist[i,0] = planes[i,0]
            planes_consist[i,1] = planes[i,1]
            planes_consist[i,2] = planes[i,2]
         
    return planes_consist


def process_file(directory, filename, k, keep_points):
    to_open = directory + '/' + filename
    df = pd.read_csv(to_open, sep=' ', header=None)
    
    dflen = df.shape[0]
    
    ratio = 1
    if(keep_points < dflen):
        ratio = keep_points / dflen 
        
    mask = np.random.choice(a=[True, False], size=(dflen, 1), p=[ratio, 1-ratio])
    
    df = df[mask] # Sample the points randomly based on the mask and ratio
    
    print('Total number of points: ' + str(dflen))
    print('Remaining points: ' + str(df.shape[0]))
    
    void_data = np.array(df)
    planes, planes_consist = normaldefinition_3D_real(void_data, k)
    
    # Construct dataframe from normals found
    ply = pd.DataFrame(planes)
    ply = ply[[3,4,5,0,1,2]] # Swap the column order
    ply_name = 'oriented_clouds/' + filename.split('.')[0] + '.ply'
    ply.to_csv(ply_name, header = False, index = False, sep = ' ') # Write to csv
    
    # Write header to csv
    f = open(ply_name,'r+')
    lines = f.readlines() 
    f.seek(0) 
    f.write(ply_header)
    for line in lines:
        f.write(line)
    f.close()


if(__name__ == '__main__'):
    ##~~ Parameters ~~##
    k = 3
    keep_points = 20000
    directory = '/home/daniel/source/repos/3dSensing/01_stereo/results'
    file_to_process = ''
    
    if(file_to_process == ''): ##~~ Warning: this will eat your processing power ~~##
        for filename in os.listdir(directory): # Iterate over all files in the directory and assign normals to the points
            if(filename.endswith(".xyz")):
                print('processing: ' + filename)
                process_file(directory, filename, k, keep_points)
                print('Done.')
                # break # just in case to not have massive exec time on our hands
    else:
        filename = file_to_process
        process_file(filename, k, keep_points)
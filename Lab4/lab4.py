# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:07:20 2020

@author: Vlad
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import math

class KnnClassifier:
    
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels 
        
    def classify_image(self, test_image, num_neighbors = 3, metric = 'l2'):
        aux_list = []   
        k_neighbors = []
        aux = []
        if metric is 'l1':
                return "nu"
        else:
           for i in range(len(self.train_images)):      
               dist = np.linalg.norm(self.train_images[i]-test_image)
               aux_list.append([dist, self.train_labels[i]])
               aux.append(dist)
           aux = np.argsort(aux)
           for i in range(num_neighbors):
               k_neighbors.append(aux_list[aux[i]][1])
           mx = 0
           result = 0
           while(len(k_neighbors) > 0):
               indices = np.where(k_neighbors == k_neighbors[0])
               
               n_of_indices = len(indices[0])
               if mx < n_of_indices:
                   mx = n_of_indices
                   result = k_neighbors[0]
               k_neighbors = np.delete(k_neighbors, indices)
               
           return result
                
                
train_images = np.loadtxt('train_images.txt') # incarcam imaginile
train_labels = np.loadtxt('train_labels.txt', 'int') # incarcam etichetele avand
test_images = np.loadtxt('test_images.txt')
test_labels = np.loadtxt('test_labels.txt', 'int')

knn = KnnClassifier(train_images, train_labels)
k = 1
x = []
y = []
for j in range(5):
    corect = 0;
    for i in range(len(test_images)):
        if knn.classify_image(test_images[i], num_neighbors = k) == test_labels[i]:
            corect += 1
    print(corect/len(test_labels))
    y.append(corect/len(test_labels))
    x.append(k)
    k += 2
        
plt.plot(x, y)

plt.xlabel("number of neighbors")
plt.ylabel("accuracy")
plt.show()
        
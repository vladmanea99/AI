# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:14:11 2020

@author: Vlad
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

train_images = np.loadtxt('train_images.txt') # incarcam imaginile
train_labels = np.loadtxt('train_labels.txt', 'int') # incarcam etichetele avand
test_images = np.loadtxt('test_images.txt')
test_labels = np.loadtxt('test_labels.txt', 'int')
 # tipul de date int
image = train_images[0, :] # prima imagine
image = np.reshape(image, (28, 28))
plt.imshow(image.astype(np.uint8), cmap='gray')
plt.show()

bins = np.linspace(start=0, stop=255, num=5) # returneaza intervalele
x_to_bins = np.digitize(train_images, bins)

naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(x_to_bins - 1, train_labels)


print(naive_bayes_model.predict(test_data))

test_data = np.digitize(test_images, bins)

print(naive_bayes_model.score(test_data - 1, test_labels))

print(naive_bayes_model)
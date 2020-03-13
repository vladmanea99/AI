# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:38:54 2020

@author: Vlad
"""

import numpy as np
from collections import Counter

train_sentences = np.load('training_sentences.npy', allow_pickle=True)
train_labels = np.load('training_labels.npy') 
test_sentences = np.load('test_sentences.npy', allow_pickle=True)
test_labels = np.load('test_labels.npy')

dictionary = {}

i = 0;
for sentence in train_sentences:
    for word in sentence:
        if not word in dictionary:
            dictionary[word] = i
            i += 1
            
result = np.zeros((len(train_sentences), len(dictionary)))
i = 0
for sentence in train_sentences:
    for word in sentence:
        result[i][dictionary[word]] += 1
    i += 1
print(result)
            
            

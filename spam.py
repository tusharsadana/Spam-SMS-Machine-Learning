# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 01:41:48 2017

@author: Tushar
"""

#spam sms

import pandas as pd
import numpy as np

df = pd.read_csv("spam.csv")

from sklearn.model_selection import train_test_split

# split into train and test
data_train, data_test, labels_train, labels_test = train_test_split(
    df.v2,
    df.v1, 
    test_size=0.2, 
    random_state=0) 

print (data_train.head())
print (labels_train.head())

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

data_train_count = vectorizer.fit_transform(data_train)
data_test_count  = vectorizer.transform(data_test)

import matplotlib.pyplot as plt

word_freq_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'occurrences':data_train_count.toarray().sum(axis=0)})
word_freq_df['frequency'] = word_freq_df['occurrences']/np.sum(word_freq_df['occurrences'])
plt.plot(word_freq_df.occurrences)
plt.show()

print (data_train_count.shape, labels_train.shape, data_test_count.shape)
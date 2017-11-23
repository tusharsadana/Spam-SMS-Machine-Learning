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




from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(data_train_count,labels_train)

y_pred2 = classifier.predict(data_test_count)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,y_pred2)

sc = classifier.score(data_test_count,labels_test)

    
check = pd.DataFrame(labels_test)
check2 = pd.DataFrame(y_pred2)

from sklearn.neighbors import KNeighborsClassifier
classifier1 = KNeighborsClassifier(n_neighbors =5, metric ='minkowski', p=2)
classifier1.fit(data_train_count,labels_train)

y_pred3 = classifier1.predict(data_test_count)

check3 = pd.DataFrame(y_pred3)

cm1 = confusion_matrix(labels_test,y_pred3)

sc1 = classifier1.score(data_test_count,labels_test)

from sklearn.tree import DecisionTreeClassifier
classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state =0)

classifier2.fit(data_train_count,labels_train)

y_pred4 = classifier2.predict(data_test_count)
check4 = pd.DataFrame(y_pred4)

cm2 = confusion_matrix(labels_test,y_pred4)
sc2 = classifier2.score(data_test_count,labels_test)


from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier(n_estimators = 10,criterion = 'entropy', random_state =0)

classifier3.fit(data_train_count,labels_train)

y_pred5 = classifier3.predict(data_test_count)
check5 = pd.DataFrame(y_pred5)

cm3 = confusion_matrix(labels_test,y_pred4)
sc3 = classifier3.score(data_test_count,labels_test)


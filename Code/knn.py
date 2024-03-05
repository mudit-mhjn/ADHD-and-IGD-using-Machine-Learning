
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 00:07:21 2022

@author: Mudit Mahajan
"""


# coding: utf-8

# In[57]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import pprint


# In[58]:


train = pd.read_csv('pubg_final_data_labels.csv')

train.Gender[train.Gender == 'Male'] = 1
train.Gender[train.Gender == 'Female'] = 0
# train = train['Gender'].replace(0, 'Female')
# train = train['Gender'].replace(1, 'Male')


# In[59]:


from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN' ,strategy='mean')
imputer = imputer.fit(train[['Do you have high self-esteem?']])
train[['Do you have high self-esteem?']] = imputer.transform(train[['Do you have high self-esteem?']])


# In[60]:


train.drop(['Unnamed: 0','Country','Do you feel preoccupied with your gaming behaviour?','Do you feel more irritability, anxiety or even sadness when you try to either reduce or stop your gaming activity?','Do you feel the need to spend increasing amount of time engaged gaming in order to achieve satisfaction or pleasure?','Do you systematically fail when trying to control or cease your gaming activity?','Have you lost interests in previous hobbies and other entertainment activities as a result of your engagement with the game?','Have you continued your gaming activity despite knowing it was causing problems between you and other people?','Have you deceived any of your family members, therapists or others because the amount of your gaming activity?','Do you play in order to temporarily escape or relieve a negative mood (e.g., helplessness, guilt, anxiety)?','Have you jeopardised or lost an important relationship, job or an educational or career opportunity because of your gaming activity?','How often do you have trouble wrapping up the final details of a project, once the challenging parts have been done?','How often do you have difficulty getting things in order when you have to do a task that requires organization?','How often do you have problems remembering appointments or obligations?','When you have a task that requires a lot of thought, how often do you avoid or delay getting started?','How often do you fidget or squirm with your hands or feet when you have to sit down for a long time','How often do you feel overly active and compelled to do things, like you were driven by a motor?',	'Do you feel nervous, anxious or on edge?','Are you not being able to stop or control worrying?','Do you worry too much about different things?','Do you have trouble relaxing?','Do you ever feel so restless that it is hard to sit still?','Do you become easily annoyed or irritable?','Do you feel afraid as if something awful might happen?'],axis=1,inplace=True)
train.drop(['solo_headshotKills','solo_kills','solo_longestTimeSurvived','solo_roundsPlayed','solo_roundMostKills','solo_top10s','solo_wins','solo_AvgSurvival','solo_Top10%','solo_WinRatio','duo_headshotKills','duo_kills','duo_longestTimeSurvived','duo_roundsPlayed','duo_roundMostKills','duo_top10s','duo_wins','duo_AvgSurvival','duo_Top10%','duo_WinRatio','squad_headshotKills','squad_kills','squad_longestTimeSurvived','squad_roundsPlayed','squad_roundMostKills','squad_top10s','squad_wins','squad_AvgSurvival','squad_Top10%','squad_WinRatio'],axis=1,inplace=True)

#This is for calculating result for ADHD. 

train.drop(['username','IGD_score','anxiety_result','IGD_result','anxiety_score'],axis=1,inplace=True)

#This is for calculating result for anxiety result.

#train.drop(['username','IGD_score','ADHD_result','IGD_result','anxiety_score'],axis=1,inplace=True)

#This is for calculating result for IGD result

#train.drop(['username','IGD_score','ADHD_result','anxiety_result','anxiety_score'],axis=1,inplace=True)


# In[61]:


print(train.columns)


# In[62]:


print(train.head)


# In[63]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('ADHD_result',axis=1), 
                                                    train['ADHD_result'], test_size=0.20, 
                                                    random_state=101)


# In[64]:


from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)


# In[66]:


print(X_train)


# In[67]:

from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train)  

y_pred = classifier.predict(X_test)  
# In[69]:
print(y_pred)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
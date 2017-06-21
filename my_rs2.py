# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:56:33 2017

@author: Priya
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

r_cols = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_csv('D:\Python\ml-100k\u.data', sep ='\t', names= r_cols)

print ratings.head()

ratings.drop(['timestamp'], 1, inplace= True)
print ratings.head()
ratings

num_users = ratings.user_id.unique().shape[0]
print num_users
num_items = ratings.movie_id.unique().shape[0]
print num_items

"""
User-based Recommendation
------------------------------------------------
"""

train_data, test_data = train_test_split(ratings, test_size=0.25) #train-test set split
print train_data.head()
print test_data.head()

num_ratings_train = train_data.rating.shape[0]
num_ratings_test = test_data.rating.shape[0]
#user_data = defaultdict(list)
#add = defaultdict(list)
#print type(train_data)

user_data= {}
for row in train_data.itertuples():
    user_data.setdefault(row[1],{})
    user_data[row[1]].setdefault(row[2],row[3])

'''
u1 = train_data.set_index('user_id', drop=True).to_dict(orient='index')
user_data= u1.set_index(['movie_id', 'rating'], drop=False ).to_dict(orient='list')
print u1
'''
'''
with open(train_data) as f1:
    one = f1.read()
    with open(train_data) as f2:
       two = f2.read()
       with open(train_data) as f3:
           three = f3.read()
           mydict ={column[1]: {column[2]:column[3]} for column in one}

'''

#print user_data

'''
for key in user_data:
    while (key == (train_data.user_id)):
        user_data[key].update({train_data.movie_id: train_data.rating})
        
'''

#k-neighbors
def knn_func(user1, k):
    w = {}
    for u2 in user_data.keys():
        if (u2==user1):
            continue
        else:
            w.update({u2: float(msd_sim(user1,u2))})
    
    sorted_w = sorted(w.items(), key=lambda x:x[1], reverse=True)
    if (len(sorted_w)<k):
        k= len(sorted_w)
    sorted_w = sorted_w[:k]
    return sorted_w

#computing similarity
def msd_sim(user1, user2):
    I= 0
    sqd_diff = 0.0
    for i in user_data[user1].keys():
        if ( i in user_data[user2].keys()):
            I= I+1
            sqd_diff = sqd_diff + float((user_data[user1][i] - user_data[user2][i])**2)
    if (sqd_diff == 0):
        return 1
    else:
        msd = float(I/sqd_diff)
        return msd
w ={}

#w = knn_func(21, 7)
#print w

import math    
pred_items = []
def recomm(user):
    for i in user_data[user].keys():
        pred = pred_ratings(user, i)
        if ( pred >= 3.8):
          pred_items.append(i)
    return pred_items
     

def pred_ratings(user1, item):
    w1= {}
    w1.update(knn_func(user1,12))
    mul=0
    div=0
    for v in w1.keys():
        #print v
        #print user_data[v].keys()
        if (item in user_data[v].keys()):
            #print v
            #print user_data[v].keys()
            #print w1[v]
            #print user_data[v][item]
            mul = mul + (w1[v] * user_data[v][item])
            div = div + math.fabs(w1[v])
    if (div==0):
        ratings=0
    else:
        ratings = mul/div
    return ratings

def accuracy(user):
    sums=0
    num_ratings = num_ratings_train
    for i in user_data[user].keys():
        sums = sums + math.fabs(pred_ratings(user,i) - user_data[user][i])
    mae = float(sums/ num_ratings)
    accuracy = (1-mae)*100
    return float(accuracy)

#print pred_ratings(21,8)

#print accuracy(21,17)

def precision(user):
    corr_items = 0
    total_recomm = len(recomm(user))
    for i in recomm(user):
        if (user_data[user][i]>3.8):
            corr_items= corr_items +1
    precision_u= corr_items/total_recomm
    return precision_u
user = input("Enter the user id")
print recomm(int(user))
print accuracy(user) 
print precision(user)
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:56:33 2017

@author: Priya
"""

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

r_cols = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_csv('D:\Python\ml-100k\u.data', sep ='\t', names= r_cols)

#print ratings.head()

ratings.drop(['timestamp'], 1, inplace= True)
#print ratings.head()


num_users = ratings.user_id.unique().shape[0]
print "number of users: ",num_users
num_items = ratings.movie_id.unique().shape[0]
print "number of items: ",num_items

"""
User-based Recommendation
------------------------------------------------
"""

train_data, test_data = train_test_split(ratings, test_size=0.25) #train-test set split
#print train_data.head()
#print test_data.head()

"""ratings_data= {}
for row in test_data.itertuples():
    user_data_test.setdefault(row[1],{})
    user_data_test[row[1]].setdefault(row[2],row[3])
"""

user_data= {}
for row in train_data.itertuples():
    user_data.setdefault(row[1],{})
    user_data[row[1]].setdefault(row[2],row[3])
    
user_data_test= {}
for row in test_data.itertuples():
    user_data_test.setdefault(row[1],{})
    user_data_test[row[1]].setdefault(row[2],row[3])

num_ratings_train = train_data.rating.shape[0]
num_ratings_test = test_data.rating.shape[0]
#user_data = defaultdict(list)
#add = defaultdict(list)
#print type(train_data)



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
def r_average(user):
    sums=0
    for item in user_data[user].keys():
        sums = sums + user_data[user][item]
    no_of_ratings= len(user_data[user].keys())
    average = float(sums/no_of_ratings)
    return average


#k-neighbors
#in the form (usr2:msd)
def knn_func(user1, k):
    w = {}
    for u2 in user_data.keys():
        if (u2==user1):
            continue
        else:
            w.update({u2: pc_sim(user1,u2)})
    
    sorted_w = sorted(w.items(), key=lambda x:x[1], reverse=True)
    if (len(sorted_w)<k):
        k= len(sorted_w)
    sorted_w = sorted_w[:k]
    return sorted_w

#computing similarity
#msd = |I[u][v]|/summation((r[u][i]- r[v][i])**2)
def pc_sim(user1, user2):
    top=0
    bottom1=0
    bottom2=0
    r_u_avg= r_average(user1)
    r_v_avg= r_average(user2)
    for i in user_data[user1].keys():
        if (i in user_data[user2].keys()):
            top= top + (user_data[user1][i] - r_u_avg)*(user_data[user2][i] - r_v_avg)
            bottom1= bottom1 + (user_data[user1][i] - r_u_avg)**2
            bottom2 = bottom2 + (user_data[user2][i] - r_v_avg)**2
    div = float(math.sqrt(bottom1*bottom2))
    if (div != 0):
        pc= float(top/div)
    else:
        pc = 1
    return pc

w ={}

#w = knn_func(21, 7)
#print w
#item is predicted if predicted rating of the item is greater than 3.8
   
pred_items = []

def recomm(user):
    dict_pred ={}
    for i in ratings.movie_id.unique():
        if (i in user_data[user].keys()):
            continue
        else:
            pred = pred_ratings(user, i)
            if ( pred >= 3.8):
                dict_pred.update({i:pred})
    sorted_dict = sorted(dict_pred.items(), key=lambda x:x[1], reverse=True)
    for k in sorted_dict:
        pred_items.append(k[0]) 
    return pred_items

#pred_ratings = summation(w[u][v] * r[v][i])/summation(|w[u][v]|)
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

#mae = summation(|predicted_ratings - actual_ratings|)
#accuracy = 1-mae
def mean_absolute_error(user):
    sums=0
    num_ratings = num_ratings_train
    for i in user_data_test[user].keys():
        sums = sums + math.fabs(pred_ratings(user,i) - user_data_test[user][i])
    mae = float(sums/ num_ratings)
    return mae

#print pred_ratings(21,8)

#print accuracy(21,17)

#precision = total correctly predicted items/ total recommended items
def precision(user, recommend):
    corr_items = 0
    total_recomm = len(recommend)
    for i in user_data_test[user].keys():
        if (i in recommend and user_data_test[user][i]>3.8):
            corr_items= corr_items +1
    precision_u= float(corr_items/total_recomm)
    return precision_u

user = input("Enter the user id")
recommend = recomm(int(user))
print "recommendations: ", recommend
print "mean absolute error: ", mean_absolute_error(int(user))
print "precision: ",precision(int(user), recommend)



"""

for row in test_data.itertuples():
    user = row[1]
    print "for the user_id, " + str(user) + " recommended item_ids are:"
    print recomm(user)
    
"""
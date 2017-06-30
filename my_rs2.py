# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:56:33 2017

@author: Priya
"""

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

mae=[]
prec=[]
rec=[]
rmse=[]

r_cols = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_csv('D:\Python\ml-100k\u.data', sep ='\t', names= r_cols)

ratings.drop(['timestamp'], 1, inplace= True)

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
    
#num_ratings_train = train_data.rating.shape[0]
#num_ratings_test = test_data.rating.shape[0]

'''
with open(train_data) as f1:
    one = f1.read()
    with open(train_data) as f2:
       two = f2.read()
       with open(train_data) as f3:
           three = f3.read()
           mydict ={column[1]: {column[2]:column[3]} for column in one}

'''

#computing average
def r_average(user):
    sums=0
    for item in user_data[user].keys():
        sums = sums + user_data[user][item]
    no_of_ratings= len(user_data[user].keys())
    average = float(sums/no_of_ratings)
    return average


#k-neighbors
#in the form {(usr2:msd)}
def knn_func(user1,item,k):
    w = {}
    for u2 in user_data.keys():
        if (u2==user1):
            continue
        if(item in user_data[u2].keys()):
            w.update({u2: pc_sim(user1,u2)})
    
    sorted_w = sorted(w.items(), key=lambda x:x[1], reverse=True)
    if (len(sorted_w)<k):
        k= len(sorted_w)
    sorted_w = sorted_w[:k]
    return sorted_w

#computing similarity
#pc = (summation(rui-r_u_avg)(rvi-r_v_avg))/sqrt((summation(rui-r_u_avg)sq)(summation(rvi-r_v_avg)sq))
def pc_sim(user1, user2):
    top=0.0
    bottom1=0.0
    bottom2=0.0
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

#item is predicted if predicted rating of the item is greater than 3.8

def recomm(user):
    dict_pred ={}
    for i in user_data_test[user].keys():
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
    w1.update(knn_func(user1,item,50))
    mul=0
    div=0
    for v in w1.keys():
        if (item in user_data[v].keys()):
            mul = mul + (w1[v] * (user_data[v][item]- r_average(v)))
            div = div + math.fabs(w1[v])
    if (div==0):
        ratings=0
    else:
        ratings =r_average(user1)+ mul/div
    return ratings

#mae = summation(|predicted_ratings - actual_ratings|)/ R_test
def mean_absolute_error(user):
    sums=0
    sq=0
    num_ratings = len(user_data_test[user].keys())
    for i in user_data_test[user].keys():
        p = pred_ratings(user,i)
        u= user_data_test[user][i]
        sums = sums + math.fabs(p-u)
        sq= sq + (p- u)**2
    err = float(sums/ num_ratings)
    mae.append(err)
    rm= math.sqrt(float(sq/num_ratings))
    rmse.append(rm)

#precision = total correctly predicted items/ total recommended items
#recall = total correctly predicted items/total liked items
def precision(user, recommend):
    tp=0.0
    fp=0.0
    fn=0.0
    for i in user_data_test[user].keys():
        if (i in recommend and user_data_test[user][i]>3.8):
            tp=tp+1
        elif(i in recommend and user_data_test[user][i]<3.8):
            fp=fp+1
        elif(i not in recommend and user_data_test[user][i]>3.8):
            fn=fn+1
    if(tp+fp==0):
        precision=0
    else:
        precision = tp/(tp+fp)
    if (tp+fn==0):
        recall=0
    else:
        recall = tp/(tp+fn)
    prec.append(precision)
    rec.append(recall)
n=0

for x in user_data_test.keys():
    pred_items=[]
    recommend = recomm(x)
    print "Recommendations for user id ", x, ": ", recommend
    precision(x, recommend)
    mean_absolute_error(x)
    n=n+1
    if(n>=100):
        break;
        
print "precision: ", sum(prec)/n
print "recall: ", sum(rec)/n
print "mean absolute error: ", sum(mae)/n
print "rmse: ", sum(rmse)/n

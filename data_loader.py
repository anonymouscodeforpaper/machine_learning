#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utilities import traite_train_test,traite_train_test_movie
import json 
import numpy as np
import pandas as pd
from operator import itemgetter



def read(name):
    if name == 'dbook2014':
        hehe_test = pd.read_csv('dbook2014/train.csv')
        df_empty = pd.read_csv('dbook2014/test.csv')
        hehe_test = traite_train_test(hehe_test)
        df_empty = traite_train_test(df_empty)
        hehe = hehe_test.append(df_empty)
        n_users = max(hehe['user'])+1 ## This is the number of users
        n_attribute_types = 2
        n_attribute_types = 1712
        return hehe_test, df_empty, n_users, n_attributes,n_attribute_types
        
        
        
        
    
    else:
        trainset = pd.read_csv(str(name) + 'train.csv')
        testset = pd.read_csv(str(name) + 'test.csv')
        hehe_test = traite_train_test(trainset)
        df_empty = traite_train_test(testset)
        df_empty['user_id'] = df_empty['user_id'].astype('int')
        df_empty['user_rating'] = df_empty['user_rating'].astype('float')
        df_empty['movie'] = df_empty['movie'].astype('int')
        hehe_test.index = range(len(hehe_test))
        df_empty.index = range(len(df_empty))
        hehe = hehe_test.append(df_empty)
        n_users = max(hehe['user_id']) + 1
        n_attribute_types = 3
        if name == 'NETFLIX':
            n_attribute_types = 1154
        if name == 'MovieLenssmall':
            n_attribute_types = 22428
        if name == 'MovieLens100k':
            n_attribute_types = 4328
    
        return hehe_test, df_empty, n_users, n_attributes,n_attribute_types

        

        

        
        
        





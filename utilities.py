#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from sklearn.preprocessing import LabelEncoder
import torch
loss_func = torch.nn.MSELoss()
from sklearn.metrics import f1_score, roc_auc_score,accuracy_score
from concurrent.futures import ProcessPoolExecutor
import json
THREADS = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def traite_train_test_movie(df):
    df['actors'] = df['actors'].apply(lambda x: json.loads(x))
    df['director'] = df['director'].apply(lambda x: json.loads(x))
    df['genre'] = df['genre'].apply(lambda x: json.loads(x))
    return df

def traite_train_test(df):
    df['authors'] = df['authors'].apply(lambda x: json.loads(x))
    df['genres'] = df['genres'].apply(lambda x: json.loads(x))
    df['user'] = df['user'].astype('int')
    df['rating'] = df['rating'].astype('float')
    df['item'] = df['item'].astype('int')
    df.index = range(len(df))
    df.index = range(len(df))
    return df

def RMSE(data, model,rate,name):  ### this function returns RMSE, MAE, precicion, recall, f1 score, accuracy, and the rating before averaging
    if name == 'book':
        users_index = data.iloc[:, 0].values
        users = torch.LongTensor(users_index).to(DEVICE)
        actors_id = data.iloc[:, 2]
        directors_id = data.iloc[:, 3]
        rating = torch.FloatTensor(
        data.iloc[:, 4].values).to(DEVICE)
        prediction,scores,contribute_actors,contribute_directors,cnm = model(users,actors_id, directors_id,rate)
        rmse = loss_func(prediction, rating)
        mae = torch.nn.L1Loss()(prediction, rating)
        return rmse ** 0.5,mae,cnm
        
    else:
        users_index = data.iloc[:, 0].values
        users = torch.LongTensor(users_index).to(DEVICE)
        actors_id = data.iloc[:, 2]
        directors_id = data.iloc[:, 3]
        genres_id = data.iloc[:, 4]
        rating = torch.FloatTensor(data.iloc[:, 5].values).to(DEVICE)
        prediction,scores,contribute_actors,contribute_directors,contribute_genres,cnm = model(users,actors_id, directors_id, genres_id)
        rmse = loss_func(prediction, rating)
        mae = torch.nn.L1Loss()(prediction, rating)
        return rmse ** 0.5,mae





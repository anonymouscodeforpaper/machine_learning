#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from concurrent.futures import ProcessPoolExecutor
THREADS = 16
def map_aspect_values_to_movies(x):
    (film, meta), aspect = x
    aspects = dict()
    if aspect == "director" and type(meta[aspect]) is str:
        aspects[meta[aspect]] = 1
    else:
        for g in meta[aspect]:
            aspects[g] = 1
    return film, meta, aspects


def dict_movie_aspect(paper_films, aspect):
    paper_films_aspect_prepended = map(
        lambda e: (e, aspect), list(paper_films.items()))
    aspect_dict = dict()
    with ProcessPoolExecutor(max_workers=THREADS) as executor:
        results = executor.map(map_aspect_values_to_movies,
                               paper_films_aspect_prepended)
    for film, meta, aspects in results:
        aspect_dict[film] = aspects

    return aspect_dict


# In[ ]:





import torch
loss_func = torch.nn.MSELoss()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def RMSE(data, model,batch_size):
    users_index = data.iloc[:, 0].values
    users = torch.LongTensor(users_index).to(DEVICE)
#     items_index = data.iloc[:, 0].values
#     items = torch.LongTensor(items_index).to(DEVICE)
    actors_id = data.iloc[:, 2]
    directors_id = data.iloc[:, 3]
    rating = torch.FloatTensor(
        data.iloc[:, 4].values).to(DEVICE)
    prediction,scores,contribute_actors,contribute_directors,cnm = model(users,actors_id, directors_id)
    rmse = loss_func(prediction, rating)
    mae = torch.nn.L1Loss()(prediction, rating)
    return rmse ** 0.5,mae, cnm



from model import aspect_augumentation
import time
from tqdm import tqdm
def train(lr, dim, reg, batch_size, num_epochs, data, test):
    model = aspect_augumentation(5576, 1712,2, dim).to(DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr,weight_decay=reg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, threshold_mode='abs',threshold = 0.005)
    for epoch in range(num_epochs):
        model.train()
        t1 = time.time()
        num_example = len(data)
        indices = list(range(num_example))
        for i in tqdm(range(0, num_example, batch_size)):
            optimizer.zero_grad()
            indexs = indices[i:min(i+batch_size, num_example)]
            users_index = data.iloc[:, 0].loc[indexs].values
            users = torch.LongTensor(users_index).to(DEVICE)
            items_index = data.iloc[:, 1].values
            items = torch.LongTensor(items_index).to(DEVICE)
            actors_id = data.iloc[:, 2].loc[indexs]
            actors_id.index = range(len(actors_id))
            directors_id = data.iloc[:, 3].loc[indexs]
            directors_id.index = range(len(directors_id))
            rating = torch.FloatTensor(
                data.iloc[:, 4].loc[indexs].values).to(DEVICE)
            prediction, scores, contribute_actors, contribute_directors,cnm = model(
                users, actors_id, directors_id)

            err = loss_func(prediction, rating) 
            err.backward()
            optimizer.step()
        t2 = time.time()
        rmse, mae,cnm = RMSE(test, model,batch_size,items)
        scheduler.step(rmse)
        print("Epoch: ", epoch, " Loss: ", err, " RMSE in test set:",
              rmse, "MAE in test set: ", mae)
        #print("Accuracy in test set is: ", accuracy, "Precision in test set:",
         #     p, "Recall in test set: ", r, "F1 scores in test set is:", f)
        print("Time consumed is:", t2-t1)
    return rmse, mae, model, cnm


import json
def traite_train_test(df):
    df['authors'] = df['authors'].apply(lambda x: json.loads(x))
    df['genres'] = df['genres'].apply(lambda x: json.loads(x))
    df['user'] = df['user'].astype('int')
    df['rating'] = df['rating'].astype('float')
    df['item'] = df['item'].astype('int')
    df.index = range(len(df))
    df.index = range(len(df))
    return df
''' functions to get movie aspects, compute similarity, get ratings to test '''


import time
import scipy
import numpy as np
import pandas as pd
from collections import Counter
import sklearn.preprocessing as pp
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor


THREADS = cpu_count() - 1


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


def viewed_matrix(ratings_cold_start, all_films, data_origin):
    user_ids = ratings_cold_start["userID"]
    item_ids = ratings_cold_start["itemID"]
    train_ratings = ratings_cold_start["rating"]

    assert len(user_ids) == len(item_ids) == len(train_ratings)

    movies_watched = dict()
    for uid in all_films.keys():
        movies_watched[uid] = dict()

    for i in range(len(item_ids)):
        current_user_id = user_ids[i]
        current_item_id = item_ids[i]
        if data_origin == 'netflix':
            current_rating = int(train_ratings[i])
        elif data_origin == 'small':
            current_rating = float(train_ratings[i])
        elif data_origin == '100k':
            current_rating = int(train_ratings[i])

        try:
            movies_watched[current_item_id][current_user_id] = current_rating
        except Exception:
            # possibly the movies lacking info such as actors which are discarded
            print('item id missing %s' % current_item_id)

    return movies_watched


def get_movies_aspect_matrix(films, aspect_type):
    aspects_associated_to_movies = dict_movie_aspect(films, aspect_type)
    movies_all_aspects_matrix = pd.DataFrame.from_dict(
        aspects_associated_to_movies, dtype='int64', orient='index')
    movies_all_aspects_matrix = movies_all_aspects_matrix.replace(np.nan, 0)
    aspects_in_db = movies_all_aspects_matrix.keys()
    print('We have %d %s (an example is %s)' %
          (len(aspects_in_db), aspect_type, aspects_in_db[0]))
    return aspects_in_db, movies_all_aspects_matrix


def preprocessing(df_all,hehe_test, df_empty, data_origin):
    start = time.time()
    df_sum = df_all.copy()
    movie_set = set(df_sum['movie'])
    movie = df_sum[['movie','actors','director','genre']].loc[df_sum[['movie','actors','director','genre']].astype(str).drop_duplicates().index]
    movie.index = range(len(movie))
    movie = movie[['movie','actors','director','genre']].loc[movie[['movie','actors','director','genre']].astype(str).drop_duplicates().index]
    
    
    films = dict()
    test_eva_dict = movie.to_dict('records')
    for row in test_eva_dict[:]:
        if row['movie'] not in films:
            films[row['movie']] = dict()
        films[row['movie']]['director'] = row['director']
        films[row['movie']]['genre'] = row['genre']
        films[row['movie']]['actors'] = row['actors']
        
        
        
        
    genres_in_db, movies_all_genres_matrix = get_movies_aspect_matrix(
    films, "genre")
    directors_in_db, movies_all_directors_matrix = get_movies_aspect_matrix(
    films, "director")
    actors_in_db, movies_all_actors_matrix = get_movies_aspect_matrix(
    films, "actors")
    print('We have %d total aspects' %
      (len(genres_in_db)+len(directors_in_db)+len(actors_in_db)))
    train_ratings_dict = dict()
    ratings_dict = dict()
    train_ratings_dict["userID"] = []
    train_ratings_dict["itemID"] = []
    train_ratings_dict["rating"] = []
    compressed_test_ratings_dict = dict()
    
    test_eva_dict = hehe_test.to_dict('records')
    for row in test_eva_dict[:]:
        tuple_key = (row['user_id'],row['movie'])
        ratings_dict[tuple_key] = int(row['user_rating'])
        train_ratings_dict["userID"].append(row['user_id'])
        train_ratings_dict["itemID"].append(row['movie'])
        train_ratings_dict["rating"].append(row['user_rating'])
    
    test_eva_dict = df_empty.to_dict('records')
    for row in test_eva_dict[:]:
        if row['user_id'] not in compressed_test_ratings_dict:
            compressed_test_ratings_dict[row['user_id']] = []
        compressed_test_ratings_dict[row['user_id']].append((row['movie'],row['user_rating']))
    

# compute similarity
    movies_watched = viewed_matrix(train_ratings_dict, films, data_origin)
    movies_watched = pd.DataFrame.from_dict(movies_watched, dtype='int64', orient='index').T
    movies_watched = movies_watched.replace(np.nan, 0)
    user_ids_in_matrix = movies_watched.index.values

# normalize vectors and then calculate cosine values by determining the matrix product
    movies_watched = movies_watched.T
    movies_watched = scipy.sparse.csc_matrix(movies_watched.values)
    normalized_matrix_by_column = pp.normalize(movies_watched.tocsc(), norm='l2', axis=0)
    cosine_sims = normalized_matrix_by_column.T * normalized_matrix_by_column
    assert cosine_sims.shape[0] == cosine_sims.shape[1] == len(user_ids_in_matrix)

# convert similarities computed to dict
    sims = dict()
    for i in user_ids_in_matrix:
        sims[i] = []
    cosine_sims = cosine_sims.todok().items()

    for ((row,col), sim) in cosine_sims:
        if row != col:
            sims[user_ids_in_matrix[row]].append((user_ids_in_matrix[col], sim))

    end = time.time()
    print("\nComputing similarity took %d seconds" % (end-start))

# convert ratings to a format as follows (film_id, user_id): rating]
    ratings_dict = dict()
    user_ids = train_ratings_dict["userID"]
    item_ids = train_ratings_dict["itemID"]
    train_ratings = train_ratings_dict["rating"]
    assert len(user_ids) == len(item_ids) == len(train_ratings)
    return films, ratings_dict, compressed_test_ratings_dict, sims, movies_all_genres_matrix, movies_all_directors_matrix, movies_all_actors_matrix


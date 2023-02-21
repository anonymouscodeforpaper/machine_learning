''' script to get predictions for netflix data '''


from measures import predictions
from processing import preprocessing
import time
import pickle
import pandas as pd
import json


if __name__ == "__main__":
    trainset = pd.read_csv('train.csv')
    testset = pd.read_csv('test.csv')
    def traite_train_test(df):
        df['actors'] = df['actors'].apply(lambda x: json.loads(x))
        df['director'] = df['director'].apply(lambda x: json.loads(x))
        df['genre'] = df['genre'].apply(lambda x: json.loads(x))
        return df
    hehe_test = traite_train_test(trainset)
    df_empty = traite_train_test(testset)
    df_empty['user_id'] = df_empty['user_id'].astype('int')
    df_empty['user_rating'] = df_empty['user_rating'].astype('float')
    df_empty['movie'] = df_empty['movie'].astype('int')
    hehe_test.index = range(len(hehe_test))
    df_empty.index = range(len(df_empty))
    df_all = hehe_test.append(df_empty)
    films, ratings_dict, compressed_test_ratings_dict, sims, movies_all_genres_matrix, movies_all_directors_matrix, movies_all_actors_matrix = preprocessing(df_all,hehe_test, df_empty, 'netflix')
    start = time.time()
    
    MUR = 0.1
    MUG = 0.8
    MUA = 0.1
    MUD = 0.1
    
    nr_predictions, accuracy, rmse, mae, precision, recall, f1 = predictions(MUR, MUG, MUA, MUD, films, compressed_test_ratings_dict, ratings_dict, sims, movies_all_genres_matrix, movies_all_directors_matrix, movies_all_actors_matrix, 'netflix')

# print results
    print("Number of user-items pairs: %d" % nr_predictions)
    print("Accuracy: %.2f " % accuracy)
    print("RMSE: %.2f" % rmse)
    print("MAE: %.2f" % mae)
    print("Precision: %.2f" % precision)
    print("Recall: %.2f" % recall)
    print("F1: %.2f" % f1)
    end = time.time()
    print("\nComputing strengths took %d seconds" % (end-start))



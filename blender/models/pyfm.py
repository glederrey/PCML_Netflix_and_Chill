import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm
from rescaler import Rescaler


def pyfm_rescaling(df_train, df_test, **kwargs):
    """
    pyFM
    First do a rescaling of the user in a way that they all have the same mean of rating.
    This counter the effect of "mood" of users. Some of them given worst/better grade even if they have the same
    appreciation of a movie.

    :param df_train:
    :param df_test:
    :param kwargs:
        gamma (float): regularization parameter
        n_features (int): number of features for matrices
        n_iter (int): number of iterations
        init_method ('global_mean' or 'movie_mean'): kind of initial matrices (better result with 'global_mean')
    :return:
    """
    rescaler = Rescaler(df_train)
    df_train_normalized = rescaler.normalize_deviation()

    prediction_normalized = pyfm(df_train_normalized, df_test, **kwargs)
    prediction = rescaler.recover_deviation(prediction_normalized)
    return prediction


def pyfm(train, test, **arg):
    print('[PYFM] applying')

    # Get the args
    num_factors = arg['num_factors']
    num_iter = arg['num_iter']    
    task = arg['task']
    initial_learning_rate = arg['initial_learning_rate']
    learning_rate_schedule = arg['learning_rate_schedule']
    
    (train_data, y_train, train_users, train_items) = prepare_data(train)
    (test_data, y_test, test_users, test_items) = prepare_data(test)
    
    v = DictVectorizer()
    X_train = v.fit_transform(train_data)
    X_test = v.transform(test_data)
    
    fm = pylibfm.FM(num_factors=num_factors, num_iter=num_iter, task=task, initial_learning_rate=initial_learning_rate, learning_rate_schedule=learning_rate_schedule)
    
    fm.fit(X_train,y_train)
    
    preds = fm.predict(X_test)
    
    for i in range(len(preds)):
        if preds[i] > 5:
            preds[i] = 5
        elif preds[i] < 1:
            preds[i] = 1
    
    df_return = test.copy()
    
    df_return.Rating = preds
    
    print('[PYFM] done')
    
    return df_return
    

def prepare_data(df):
    data = []
    y = list(df.Rating)
    users=set(df.User.unique())
    items=set(df.Movie.unique())
    usrs = list(df.User)
    movies = list(df.Movie)
    for i in range(len(df)):
        y[i] = float(y[i])
        data.append({ "user_id": str(usrs[i]), "movie_id": str(movies[i])})
    return (data, np.array(y), users, items)

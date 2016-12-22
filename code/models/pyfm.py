#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>, Gael Lederrey <gael.lederrey@epfl.ch>,
# Stefano Savare <stefano.savare@epfl.ch>
#
# Distributed under terms of the MIT license.

"""
Factorization Machine with the library LibFM (Wrapper Python PyFM)
"""
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm
from rescaler import Rescaler
import pandas as pd


def pyfm_rescaled(train, test, **kwargs):
    """
    Factorization Machines with PyFM rescaled

    First, a rescaling of the user such that they all have the same average of rating is done.
    Then, the predictions are done using the function pyfm().
    Finally, the predictions are rescaled to recover the deviation of each user.

    Args:
        train (pandas.DataFrame): train set
        test (pandas.DataFrame): test set
        **kwargs: Arbitrary keyword arguments. Directly given to pyfm().

    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)
    """
    
    # We commented the algorithm because it takes a long time to train and
    # returns the predictions. If you want, you can uncomment the algorithm
    # and make it run with the run.py
    
    """
    # Load the class Rescaler
    rescaler = Rescaler(train)
    # Normalize the train data
    df_train_normalized = rescaler.normalize_deviation()

    # Predict using the normalized trained data
    prediction_normalized = pyfm(df_train_normalized, test, **kwargs)
    # Rescale the prediction to recover the deviations
    prediction = rescaler.recover_deviation(prediction_normalized)
    """
    
    # Comment this line if you uncomment the previous lines
    prediction = pd.read_csv('data/pyfm_rescaled_pred.csv')
    return prediction


def pyfm(train, test, **kwargs):
    """
    Factorization Machines with PyFM

    Args:
        train (pd.DataFrame): train set
        test (pd.DataFrame): test set
        **kwargs: Arbitrary keyword arguments.
            num_factors (int): Number of factors for the Factorization Machine
            num_iter (int): Number of iterations
            init_lr (float): Initial learning rate

    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)

    """

    # We commented the algorithm because it takes a long time to train and
    # returns the predictions. If you want, you can uncomment the algorithm
    # and make it run with the run.py

    """
    # Get the args
    num_factors = kwargs['num_factors']
    num_iter = kwargs['num_iter']
    task = 'regression'
    initial_learning_rate = kwargs['init_lr']
    learning_rate_schedule = 'optimal'

    # Prepare the data
    (train_data, y_train, train_users, train_items) = prepare_data(train)
    (test_data, y_test, test_users, test_items) = prepare_data(test)

    # Transform in dict
    v = DictVectorizer()
    X_train = v.fit_transform(train_data)
    X_test = v.transform(test_data)

    # Start PyFM
    fm = pylibfm.FM(num_factors=num_factors, num_iter=num_iter, task=task,
                    initial_learning_rate=initial_learning_rate,
                    learning_rate_schedule=learning_rate_schedule)

    # Fit
    fm.fit(X_train, y_train)

    # And predict
    preds = fm.predict(X_test)

    # Small postprocessing
    for i in range(len(preds)):
        if preds[i] > 5:
            preds[i] = 5
        elif preds[i] < 1:
            preds[i] = 1

    df_return = test.copy()

    df_return.Rating = preds
    """
    
    # Comment this line if you uncomment the previous lines
    df_return = pd.read_csv('data/pyfm_pred.csv')

    return df_return


def prepare_data(df):
    """
    Prepare the data for the specific format used by PyFM.

    Args:
        df (pd.DataFrame): Initial DataFrame to transform

    Returns:
        data (array[dict]): Array of dict with user and movie ids
        y (np.array): Ratings give in the initial pd.DataFrame
        users (set): Set of user ids
        movies (set): Set of movie ids

    """
    data = []
    y = list(df.Rating)
    users = set(df.User.unique())
    movies = set(df.Movie.unique())
    usrs = list(df.User)
    mvies = list(df.Movie)
    for i in range(len(df)):
        y[i] = float(y[i])
        data.append({"user_id": str(usrs[i]), "movie_id": str(mvies[i])})
    return (data, np.array(y), users, movies)

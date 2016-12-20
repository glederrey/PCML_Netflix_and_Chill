#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>, Gael Lederrey <gael.lederrey@epfl.ch>,
# Stefano Savare <stefano.savare@epfl.ch>
#
# Distributed under terms of the MIT license.

import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from rescaler import Rescaler


def collaborative_filtering_rescaling(df_train, df_test, **kwargs):
    """
    Collaborative Filtering
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

    prediction_normalized = collaborative_filtering(df_train_normalized, df_test, **kwargs)
    prediction = rescaler.recover_deviation(prediction_normalized)
    return prediction


def collaborative_filtering(train, test, **kwarg):
    """
    Function to return the prediction of the collaborative filtering implementation

    Args:
        train (pandas.DataFrame): train set
        test (pandas.DataFrame): test set
        **kwarg: Arbitrary keyword arguments.

    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)
    """
    # Rating of the train set
    rating = train.pivot(index='User', columns='Movie').Rating

    users, movies = rating.index, rating.columns
    Nu, Nm = len(users), len(movies)

    # Number of movie features
    nbr_movie_features = kwarg['movie_features']

    # Matrix of user preferences
    U = pd.DataFrame(np.random.rand(Nu, nbr_movie_features), index=users, columns=range(1, nbr_movie_features + 1))

    # Matrix of movie features
    M = pd.DataFrame(np.random.rand(Nm, nbr_movie_features), index=movies, columns=range(1, nbr_movie_features + 1))

    U0, M0 = U.copy(), M.copy()

    alpha = kwarg['alpha']

    bestFit = {'U': U0, 'M': M0, 'error': np.inf}
    fitUM(bestFit, alpha, U0, M0, train, rating, users, movies)

    U_opt, M_opt, error = bestFit['U'], bestFit['M'], bestFit['error']
    pred_mat = predict(U_opt, M_opt)

    pred = []

    df_return = test.copy()

    for i in range(len(test)):
        dd = test.iloc[i]
        val = pred_mat[dd.Movie][dd.User]
        if val > 5:
            val = 5
        elif val < 1:
            val = 1
        pred.append(val)

    df_return.Rating = pred

    return df_return


def rmse_cf(y, yhat):
    """ Function that computes the RMSE between 2 vectors (np.array) """
    err = (y - yhat) ** 2
    return np.sqrt(np.mean(err.mean(skipna=True)))

    
def fitU(U, M, train, rating, error_history, bestFit, alpha, users):
    """
    Function to fit the user matrix U in ALS

    Args:
        U:  user matrix in ALS
        M: movie matrix in ALS
        train: train set
        rating: vector of known ratings, the algorithm will predict the corresponding ratings
        error_history: list containing the errors of all the iterations
        bestFit: dictionary with keys = {'U', 'M', 'error'}
        alpha: parameter in the Ridge Regression
        users: list of all the users

    Returns:
        U
        delta (int)
    """

    # Join ratings and movie features
    Udata = train.set_index('Movie').join(M).sort_values('User').set_index('User')

    # Function for fitting individual users
    model = Ridge(fit_intercept=False, alpha=alpha)

    def Ufit(i):
        df = Udata.ix[i]
        try:
            X = df.drop('Rating', axis=1)
            y = df.Rating
            model.fit(X, y)
            return model.coef_
        except:
            return df

    # Fit all users
    for i in users:
        U.ix[i] = Ufit(i)
    # Calculate the error
    pred = predict(U, M)
    error = rmse_cf(rating, pred)
    error_history.append(error)
    delta = bestFit['error'] - error
    if delta > 0:
        bestFit['U'] = U
        bestFit['M'] = M
        bestFit['error'] = error
    return U, delta


def fitM(U, M, train, rating, error_history, bestFit, alpha, movies):
    """
    Function to fit the user matrix M in ALS

    Args:
        U:  user matrix in ALS
        M: movie matrix in ALS
        train: train set
        rating: vector of known ratings, the algorithm will predict the corresponding ratings
        error_history: list containing the errors of all the iterations
        bestFit: dictionary with keys = {'U', 'M', 'error'}
        alpha: parameter in the Ridge Regression
        users: list of all the users

    Returns:
        M
        delta (int)
    """

    # Join ratings and user preferences
    Mdata = train.set_index('User').join(U).sort_values('Movie').set_index('Movie')

    # Function for fitting individual movies
    model = Ridge(fit_intercept=False, alpha=alpha)

    def Mfit(j):
        df = Mdata.ix[j]
        X = df.drop('Rating', axis=1)
        y = df.Rating
        model.fit(X, y)
        return model.coef_

    # Fit all movies
    for j in movies:
        M.ix[j] = Mfit(j)
    # Calculate the error
    pred = predict(U, M)
    error = rmse_cf(rating, pred)
    error_history.append(error)
    delta = bestFit['error'] - error
    if delta > 0:
        bestFit['U'] = U
        bestFit['M'] = M
        bestFit['error'] = error
    return M, delta


def predict(U, M):
    """ Function to predict the ratings given the matrices U and M """
    pred = U.dot(M.T)
    pred[pred > 5] = 5
    pred[pred < 1] = 1
    return pred


def fitUM(bestFit, alpha, U0, M0, train, rating, users, movies, tol=0.05):
    """
    Function that repeat the ALS optimization up to convergence

    Args:
        bestFit (dict): keys = {'U', 'M', 'error'}
        alpha (float):  Ridge Regression regularizer
        U0: starting matrix U
        M0: starting matrix M
        train: train set
        rating: vector of known ratings, the algorithm will predict the corresponding ratings
        users (list): list of all the users
        movies (list): list of all the items
        tol (float): convergence tolerance

    Returns:
        list: error_history, a list containing all the errors
    """

    U, M = U0.copy(), M0.copy()
    error_history = []
    U, delta = fitU(U, M, train, rating, error_history, bestFit, alpha, users)
    tolerance = tol
    while delta > tolerance:
        M, delta = fitM(U, M, train, rating, error_history, bestFit, alpha, movies)
        if delta > tolerance:
            U, delta = fitU(U, M, train, rating, error_history, bestFit, alpha, users)
    return error_history

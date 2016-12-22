#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>, Gael Lederrey <gael.lederrey@epfl.ch>,
# Stefano Savare <stefano.savare@epfl.ch>
#
# Distributed under terms of the MIT license.

"""
Matrix Factorization using Stochastic Gradient Descent (MF-SGD)
"""

from models.means import *
from rescaler import Rescaler
from models.helpers_scipy import *


def mf_SGD_rescaled(train, test, **kwargs):
    """
    Matrix factorization using SGD rescaled.

    First, a rescaling of the user such that they all have the same average of rating is done.
    Then, the predictions are done using the function mf_SGD().
    Finally, the predictions are rescaled to recover the deviation of each user.

    Args:
        train (pandas.DataFrame): train set
        test (pandas.DataFrame): test set
        **kwargs: Arbitrary keyword arguments. Directly given to mf_SGD().

    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)
    """
    # Load the class Rescaler
    rescaler = Rescaler(train)
    # Normalize the train data
    df_train_normalized = rescaler.normalize_deviation()

    # Predict using the normalized trained data
    prediction_normalized = mf_SGD(df_train_normalized, test, **kwargs)
    # Rescale the prediction to recover the deviations
    prediction = rescaler.recover_deviation(prediction_normalized)
    return prediction


def mf_SGD(train, test, **kwargs):
    """
    Matrix factorization using SGD.

    Args:
        train (pd.DataFrame): train set
        test (pd.DataFrame): test set
        **kwargs: Arbitrary keyword arguments.
            gamma (float): regularization parameter
            n_features (int): number of features for matrices
            n_iter (int): number of iterations
            init_method ('global_mean' or 'movie_mean'): kind of initial matrices (better result with 'global_mean')

    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)

    """

    # initial parameters
    gamma = kwargs['gamma']
    n_features = kwargs['n_features']
    n_iter = kwargs['n_iter']
    init_method = kwargs['init_method']

    # convert to scipy.sparse matrices
    train = df_to_sp(train)
    test = df_to_sp(test)

    # set seed
    np.random.seed(988)

    # initiate matrix
    user_features, item_features = init_MF(train, n_features, init_method)

    # find the non-zero indices
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    # do learning
    for it in range(n_iter):
        # shuffle the training rating indices
        np.random.shuffle(nz_train)

        # decrease step size
        gamma /= 1.2

        for d, n in nz_train:
            # matrix factorization.
            loss = train[d, n] - np.dot(item_features[d, :], user_features[n, :].T)
            grad_item = loss * user_features[n, :]
            grad_user = loss * item_features[d, :]

            item_features[d, :] += gamma * grad_item
            user_features[n, :] += gamma * grad_user

    # predict on test set
    prediction = predict(item_features, user_features, test)

    # convert do DataFrame
    prediction = sp_to_df(prediction)

    return prediction


def init_MF(train, num_features, method='global_mean'):
    """init the parameter for matrix factorization.
    :param train:
    :param num_features:
    :param method:
    :return:
            user_features: shape = num_user, num_features (matrix Z)
            item_features: shape = num_item, num_features (matrix W)
    """

    num_item, num_user = train.shape

    # fill matrices in a way that their first multiplication give the mean
    # (speed up the convergence)
    if method == 'global_mean':
        m = train.sum() / train.nnz
        x = np.sqrt(m / num_features)
        user_features = np.full((num_user, num_features), x, dtype=np.float64)
        item_features = np.full((num_item, num_features), x, dtype=np.float64)

    # fill the matrices with movie mean in first column (worst)
    # (slower but can give good results)
    if method == 'movie_mean':
        user_features = np.random.rand(num_user, num_features) / 1000.
        item_features = np.random.rand(num_item, num_features) / 1000.

        item_means = nonzero_row_mean(train)
        if np.isnan(item_means).any():
            print("item_means has nan")
        item_features[:, 0] = item_means

        user_ones = np.ones(train.shape[1])
        user_features[:, 0] = user_ones

    return user_features, item_features


def predict(item_features, user_features, test_set):
    """ Apply MF model. Multiply matrices W and Z and select the wished
    predictions according to test set
    """

    # copy test set
    filled_test = sp.lil_matrix.copy(test_set)

    # compute prediction
    pred_matrix = np.dot(item_features, user_features.T)

    # fill test set with predicted label
    users, items, ratings = sp.find(filled_test)
    for row, col in zip(users, items):
        filled_test[row, col] = pred_matrix[row, col]

    return filled_test

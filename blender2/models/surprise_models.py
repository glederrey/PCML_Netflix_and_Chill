#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>, Gael Lederrey <gael.lederrey@epfl.ch>,
# Stefano Savare <stefano.savare@epfl.ch>
#
# Distributed under terms of the MIT license.

"""
Models defined in library Surprise. (Models are KNN, SlopeOne, SVD and BaselineOnly)
"""

from surprise import *
import numpy as np
from rescaler import Rescaler


def knn_rescaled(train, test, **kwargs):
    """
    K Nearest Neighbors with Baseline from library Surprise rescaled

    First, a rescaling of the user such that they all have the same average of rating is done.
    Then, the predictions are done using the function knn().
    Finally, the predictions are rescaled to recover the deviation of each user.

    Args:
        train (pandas.DataFrame): train set
        test (pandas.DataFrame): test set
        **kwargs: Arbitrary keyword arguments. Directly given to knn().

    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)
    """
    # Load the class Rescaler
    rescaler = Rescaler(train)
    # Normalize the train data
    df_train_normalized = rescaler.normalize_deviation()

    # Predict using the normalized trained data
    prediction_normalized = knn(df_train_normalized, test, **kwargs)
    # Rescale the prediction to recover the deviations
    prediction = rescaler.recover_deviation(prediction_normalized)
    return prediction


def knn(train, test, **kwargs):
    """
    K Nearest Neighbors with Baseline from library Surprise

    Args:
        train (pandas.DataFrame): train set
        test (pandas.DataFrame): test set
        **kwargs: Arbitrary keyword arguments.
            k (int): Number of nearest neighbor for the algorithm
            sim_options (dict): Dictionary specific for the kNN algorithms in Surprise

    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)
    """

    # Get parameters
    k = kwargs['k']
    sim_options = kwargs['sim_options']

    # First, we need to dump the pandas DF into files
    train_file = 'tmp_train.csv'
    test_file = 'tmp_test.csv'
    train.to_csv(train_file, index=False, header=False)
    test.to_csv(test_file, index=False, header=False)

    # Create Reader
    reader = Reader(line_format='user item rating', sep=',')

    # Train and test set for Surprise
    fold = [(train_file, test_file)]

    # Load the data
    data = Dataset.load_from_folds(fold, reader=reader)

    # Algorithm
    algo = KNNBaseline(k=k, sim_options=sim_options)

    # Go through 1 fold
    for trainset, testset in data.folds():
        # Train
        algo.train(trainset)

        # Predict
        predictions = algo.test(testset)

    # Postprocess the predictions
    pred = np.zeros(len(predictions))
    for i in range(len(predictions)):
        val = predictions[i].est
        if val > 5:
            pred[i] = 5
        elif val < 1:
            pred[i] = 1
        else:
            pred[i] = val

    # Copy the test
    df_return = test.copy()
    df_return.Rating = pred

    return df_return


def svd_rescaled(train, test, **kwargs):
    """
    Singular Value Decomposition from library Surprise rescaled
    (Based on Matrix Factorization)

    First, a rescaling of the user such that they all have the same average of rating is done.
    Then, the predictions are done using the function svd().
    Finally, the predictions are rescaled to recover the deviation of each user.

    Args:
        train (pandas.DataFrame): train set
        test (pandas.DataFrame): test set
        **kwargs: Arbitrary keyword arguments. Directly given to svd().

    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)
    """
    # Load the class Rescaler
    rescaler = Rescaler(train)
    # Normalize the train data
    df_train_normalized = rescaler.normalize_deviation()

    # Predict using the normalized trained data
    prediction_normalized = svd(df_train_normalized, test, **kwargs)
    # Rescale the prediction to recover the deviations
    prediction = rescaler.recover_deviation(prediction_normalized)
    return prediction


def svd(train, test, **kwargs):
    """
    Singular Value Decomposition from library Surprise
    (Based on Matrix Factorization)

    Args:
        train (pandas.DataFrame): train set
        test (pandas.DataFrame): test set
        **kwargs: Arbitrary keyword arguments.
            n_epochs (int): Number of epochs to train the algorithm
            lr_all (float): Learning rate for all the parameters in the algorithm
            reg_all (float): Regularization parameter for all the parameters in the algorithm

    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)
    """
    # Get parameters
    n_epochs = kwargs['n_epochs']
    lr_all = kwargs['lr_all']
    reg_all = kwargs['reg_all']

    # First, we need to dump the pandas DF into files
    train_file = 'tmp_train.csv'
    test_file = 'tmp_test.csv'
    train.to_csv(train_file, index=False, header=False)
    test.to_csv(test_file, index=False, header=False)

    # Create Reader
    reader = Reader(line_format='user item rating', sep=',')

    # Train and test set for Surprise
    fold = [(train_file, test_file)]

    # Load the data
    data = Dataset.load_from_folds(fold, reader=reader)

    # Algorithm
    algo = SVD(n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)

    # Go through 1 fold
    for trainset, testset in data.folds():
        # Train
        algo.train(trainset)

        # Predict
        predictions = algo.test(testset)

    # Postprocess the predictions
    pred = np.zeros(len(predictions))
    for i in range(len(predictions)):
        val = predictions[i].est
        if val > 5:
            pred[i] = 5
        elif val < 1:
            pred[i] = 1
        else:
            pred[i] = val

    # Copy the test
    df_return = test.copy()
    df_return.Rating = pred

    return df_return


def slope_one_rescaled(train, test):
    """
    SlopeOne from library Surprise rescaled

    First, a rescaling of the user such that they all have the same average of rating is done.
    Then, the predictions are done using the function slope_one().
    Finally, the predictions are rescaled to recover the deviation of each user.

    Args:
        train (pandas.DataFrame): train set
        test (pandas.DataFrame): test set

    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)
    """
    # Load the class Rescaler
    rescaler = Rescaler(train)
    # Normalize the train data
    df_train_normalized = rescaler.normalize_deviation()

    # Predict using the normalized trained data
    prediction_normalized = slope_one(df_train_normalized, test)
    # Rescale the prediction to recover the deviations
    prediction = rescaler.recover_deviation(prediction_normalized)
    return prediction


def slope_one(train, test):
    """
    SlopeOne from library Surprise

    Args:
        train (pandas.DataFrame): train set
        test (pandas.DataFrame): test set

    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)
    """
    # First, we need to dump the pandas DF into files
    train_file = 'tmp_train.csv'
    test_file = 'tmp_test.csv'
    train.to_csv(train_file, index=False, header=False)
    test.to_csv(test_file, index=False, header=False)

    # Create Reader
    reader = Reader(line_format='user item rating', sep=',')

    # Train and test set for Surprise
    fold = [(train_file, test_file)]

    # Load the data
    data = Dataset.load_from_folds(fold, reader=reader)

    # Algorithm
    algo = SlopeOne()

    # Go through 1 fold
    for trainset, testset in data.folds():
        # Train
        algo.train(trainset)

        # Predict
        predictions = algo.test(testset)

    # Postprocess the predictions
    pred = np.zeros(len(predictions))
    for i in range(len(predictions)):
        val = predictions[i].est
        if val > 5:
            pred[i] = 5
        elif val < 1:
            pred[i] = 1
        else:
            pred[i] = val

    # Copy the test
    df_return = test.copy()
    df_return.Rating = pred

    return df_return


def baseline_rescaled(train, test):
    """
    BaselineOnly from library Surprise rescaled

    First, a rescaling of the user such that they all have the same average of rating is done.
    Then, the predictions are done using the function baseline().
    Finally, the predictions are rescaled to recover the deviation of each user.

    Args:
        train (pandas.DataFrame): train set
        test (pandas.DataFrame): test set

    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)
    """
    # Load the class Rescaler
    rescaler = Rescaler(train)
    # Normalize the train data
    df_train_normalized = rescaler.normalize_deviation()

    # Predict using the normalized trained data
    prediction_normalized = baseline(df_train_normalized, test)
    # Rescale the prediction to recover the deviations
    prediction = rescaler.recover_deviation(prediction_normalized)
    return prediction


def baseline(train, test):
    """
    BaselineOnly from library Surprise

    Args:
        train (pandas.DataFrame): train set
        test (pandas.DataFrame): test set

    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)
    """
    # First, we need to dump the pandas DF into files
    train_file = 'tmp_train.csv'
    test_file = 'tmp_test.csv'
    train.to_csv(train_file, index=False, header=False)
    test.to_csv(test_file, index=False, header=False)

    # Create Reader
    reader = Reader(line_format='user item rating', sep=',')

    # Train and test set for Surprise
    fold = [(train_file, test_file)]

    # Load the data
    data = Dataset.load_from_folds(fold, reader=reader)

    # Algorithm
    algo = BaselineOnly()

    # Go through 1 fold
    for trainset, testset in data.folds():
        # Train
        algo.train(trainset)

        # Predict
        predictions = algo.test(testset)

    # Postprocess the predictions
    pred = np.zeros(len(predictions))
    for i in range(len(predictions)):
        val = predictions[i].est
        if val > 5:
            pred[i] = 5
        elif val < 1:
            pred[i] = 1
        else:
            pred[i] = val

    # Copy the test
    df_return = test.copy()
    df_return.Rating = pred

    return df_return

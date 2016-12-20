#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>, Gael Lederrey <gael.lederrey@epfl.ch>,
# Stefano Savare <stefano.savare@epfl.ch>
#
# Distributed under terms of the MIT license.

"""
Mean prediction method, assigning movie/user/global mean to items.
"""

import pandas as pd
from rescaler import Rescaler


def global_mean(train, test):
    """
    Overall mean

    Args:
        train (pandas.DataFrame): train set
        test (pandas.DataFrame): test set

    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)
    """

    # Prepare
    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x))

    # Global mean
    mean = train['Rating'].mean()

    # Apply
    predictions.Rating = mean

    # Convert ID's to integer
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    return predictions


def user_mean(train, test):
    """
    User mean

    Args:
        train (pandas.DataFrame): train set
        test (pandas.DataFrame): test set

    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)
    """

    # Prepare
    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x))

    means = train.groupby('User').mean()['Rating']

    def line(df):
        df['Rating'] = means.loc[df['User'].iloc[0]]
        return df

    predictions = predictions.groupby('User').apply(line)

    # Convert ID's to integers
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    return predictions


def movie_mean_rescaled(train, test):
    """
    Movie mean rescaled

    First, a rescaling of the user such that they all have the same average of rating is done.
    Then, the predictions are done using the function movie_mean().
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
    prediction_normalized = movie_mean(df_train_normalized, test)
    # Rescale the prediction to recover the deviations
    prediction = rescaler.recover_deviation(prediction_normalized)
    return prediction


def movie_mean(train, test):
    """
    Movie mean

    Args:
        train (pandas.DataFrame): train set
        test (pandas.DataFrame): test set

    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)
    """

    # prepare
    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x))

    # pd.DataFrame ['Movie', 'Mean']
    means = train.groupby('Movie').mean()['Rating']

    def line(df):
        df['Rating'] = means.loc[df['Movie'].iloc[0]]
        return df

    predictions = predictions.groupby('Movie').apply(line)

    # convert ID's to integers
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    return predictions


def movie_mean_deviation_user_rescaled(train, test):
    """
    Movie mean rescaled with the 'deviation_per_user' file and rescaled again.

    First, a rescaling of the user such that they all have the same average of rating is done.
    Then, the predictions are done using the function movie_mean_deviation_user().
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
    prediction_normalized = movie_mean_deviation_user(df_train_normalized, test)
    # Rescale the prediction to recover the deviations
    prediction = rescaler.recover_deviation(prediction_normalized)
    return prediction


def movie_mean_deviation_user(train, test):
    """
    Movie mean rescaled with the 'deviation_per_user' file.

    Args:
        train (pandas.DataFrame): train set
        test (pandas.DataFrame): test set

    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)
    """

    # prepare
    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x))

    # pd.DataFrame ['Movie', 'Mean']
    means = train.groupby('Movie').mean()['Rating']

    # load file 'deviation_per_user.csv'
    deviation = pd.read_csv('data/deviations_per_users.csv')

    def line(df):
        df['Rating'] = means.loc[int(df['Movie'])] + deviation.loc[int(df['User']) - 1].dev
        return df

    predictions = predictions.apply(line, axis=1)

    # convert ID's to integer
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    return predictions

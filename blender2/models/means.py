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

Functions have the following signature:
    Args:
        train (pandas.DataFrame): train set
        test (pandas.DataFrame): test set

    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)
"""

from models.helpers_scipy import *
from helpers import *
import pandas as pd
from rescaler import Rescaler


def user_mean(train, test):
    """ user mean """

    # prepare
    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x))

    # pd.DataFrame ['User', 'Mean']
    means = train.groupby('User').mean()['Rating']

    def line(df):
        df['Rating'] = means.loc[df['User'].iloc[0]]
        return df

    predictions = predictions.groupby('User').apply(line)

    # convert ID's to integers
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    return predictions

def movie_mean_rescaling(df_train, df_test):
    rescaler = Rescaler(df_train)
    df_train_normalized = rescaler.normalize_deviation()

    prediction_normalized = movie_mean(df_train_normalized, df_test)
    prediction = rescaler.recover_deviation(prediction_normalized)
    return prediction

def movie_mean(train, test):
    """ movie mean """

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


def global_mean(train, test):
    """ overall mean """

    # prepare
    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x))

    # global mean
    mean = train['Rating'].mean()

    # apply
    predictions.Rating = mean

    # convert ID's to integer
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    return predictions

def movie_mean_deviation_user_rescaling(df_train, df_test):
    rescaler = Rescaler(df_train)
    df_train_normalized = rescaler.normalize_deviation()

    prediction_normalized = movie_mean_deviation_user(df_train_normalized, df_test)
    prediction = rescaler.recover_deviation(prediction_normalized)
    return prediction
    
def movie_mean_deviation_user(train, test):
    """ movie mean rescaled with the 'deviation_per_user' file """

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

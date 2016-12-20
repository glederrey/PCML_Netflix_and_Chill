#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>
#
# Distributed under terms of the MIT license.
"""
Median prediction method, assigning movie/user/global mean to items.

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


def user_median(train, test):
    """ user median """

    # prepare
    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x))

    # pd.DataFrame ['User', 'Median']
    means = train.groupby('User').median()['Rating']

    def line(df):
        df['Rating'] = means.loc[df['User'].iloc[0]]
        return df

    predictions = predictions.groupby('User').apply(line)

    # convert ID's to integers
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    return predictions
    
def movie_median_rescaling(df_train, df_test):
    rescaler = Rescaler(df_train)
    df_train_normalized = rescaler.normalize_deviation()

    prediction_normalized = movie_median(df_train_normalized, df_test)
    prediction = rescaler.recover_deviation(prediction_normalized)
    return prediction

def movie_median(train, test):
    """ movie median """

    # prepare
    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x))

    # pd.DataFrame ['Movie', 'Median']
    means = train.groupby('Movie').median()['Rating']

    def line(df):
        df['Rating'] = means.loc[df['Movie'].iloc[0]]
        return df

    predictions = predictions.groupby('Movie').apply(line)

    # convert ID's to integers
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    return predictions


def global_median(train, test):
    """ global median """
    # prepare
    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x))

    # global median
    mean = train['Rating'].median()

    # apply
    predictions.Rating = mean

    # convert ID's to integer
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    return predictions

def movie_median_deviation_user_rescaling(df_train, df_test):
    rescaler = Rescaler(df_train)
    df_train_normalized = rescaler.normalize_deviation()

    prediction_normalized = movie_median_deviation_user(df_train_normalized, df_test)
    prediction = rescaler.recover_deviation(prediction_normalized)
    return prediction
    
    
def movie_median_deviation_user(train, test):
    """ movie median rescaled with the 'deviation_per_user' file """

    # prepare
    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x))

    # pd.DataFrame ['Movie', 'Median']
    means = train.groupby('Movie').median()['Rating']

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

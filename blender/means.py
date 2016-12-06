#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>
#
# Distributed under terms of the MIT license.
"""
Mean prediction method, assigning movie/user/global mean to all items
"""

import numpy as np
from helpers_scipy import *
import pandas as pd


def user_mean(train, test):
    print("[USER_MEAN] applying")
    predictions = pd.DataFrame.copy(test)
    means = train.groupby('User').mean()['Rating']

    def line(df):
        df['Rating'] = means[df['User']]
        return df[['User', 'Movie', 'Rating']]

    predictions = predictions.apply(line, axis=1)

    # integer for id
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    print("[USER_MEAN] done")
    return predictions


def movie_mean(train, test):
    print("[MOVIE_MEAN] applying")

    predictions = pd.DataFrame.copy(test)
    means = train.groupby('Movie').mean()['Rating']

    def line(df):
        df['Rating'] = means[df['Movie']]
        return df[['User', 'Movie', 'Rating']]

    predictions = predictions.apply(line, axis=1)

    # integer for id
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    print("[MOVIE_MEAN] done")
    return predictions


def global_mean(train, test):
    print("[GLOBAL_MEAN] applying")

    predictions = pd.DataFrame.copy(test)
    mean = train['Rating'].mean()

    def line(df):
        df['Rating'] = mean
        return df[['User', 'Movie', 'Rating']]

    predictions = predictions.apply(line, axis=1)

    # integer for id
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    print("[GLOBAL_MEAN] done")
    return predictions

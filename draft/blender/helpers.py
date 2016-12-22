#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>
#
# Distributed under terms of the MIT license.


"""
Main helpers functions
HERE TO INSERT METHOD SUCH AS LOAD_DATA, WRITE_PREDICTION, ....
"""

import pandas as pd
import numpy as np
import os
import sys

#from pyspark.mllib.recommendation import ALS
#from pyspark import *
#from pyspark.sql import *


def load_csv(filename='../data/data_train.csv'):
    df = pd.read_csv(filename)
    df['User'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
    df['Movie'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
    df['Rating'] = df['Prediction']
    df = df.drop(['Id', 'Prediction'], axis=1)
    return df


def load_csv_kaggle():
    return load_csv('../data/sampleSubmission.csv')


def submission_table(original_df, col_userID, col_movie, col_rate):
    def id(row):
        return 'r' + str(int(row[col_userID])) + '_c' + str(int(row[col_movie]))

    def pred(row):
        return row[col_rate]

    df = pd.DataFrame.copy(original_df)
    df['Id'] = df.apply(id, axis=1)
    df['Prediction'] = df.apply(pred, axis=1)

    return df[['Id', 'Prediction']]


def extract_from_original_table(original_df):
    df = pd.DataFrame.copy(original_df)
    df['UserID'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
    df['MovieID'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
    df['Rating'] = df['Prediction']
    df = df.drop(['Id', 'Prediction'], axis=1)
    return df


def split(df, cut):
    size = df.shape[0]
    keys = list(range(size))
    np.random.shuffle(keys)

    key_cut = int(size * cut)
    test_key = keys[:key_cut]
    train_key = keys[key_cut:]

    test = df.loc[test_key]
    train = df.loc[train_key]

    return train, test


def evaluate(prediction, test_set):
    test2 = test_set.sort_values(['Movie', 'User']).reset_index(drop=True)

    test2['square_error'] = np.square(test2['Rating'] - prediction['Rating'])

    mse = test2['square_error'].mean()
    rmse = np.sqrt(mse)

    return rmse


def signed_evaluate(prediction, test_set):
    test2 = test_set.sort_values(['Movie', 'User']).reset_index(drop=True)

    test2['error'] = (test2['Rating'] - prediction['Rating'])

    mean_error = test2['error'].mean()

    return mean_error


def blender(array_df, weights):
    if len(array_df) != len(weights):
        print("[WARNING] Pélo... size(array_df) != size(weights)")

    blender = array_df[0].sort_values(['Movie', 'User']).reset_index(drop=True)

    blender['Rating'] = 0.

    for (df, w) in zip(array_df, weights):
        blender['Rating'] += w * (df.sort_values(['Movie', 'User']).reset_index(drop=True))['Rating']

    return blender

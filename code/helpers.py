#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>, Gael Lederrey <gael.lederrey@epfl.ch>,
# Stefano Savare <stefano.savare@epfl.ch>
#
# Distributed under terms of the MIT license.


"""
Main helpers functions
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
import time

def load_csv(filename='data/data_train.csv'):
    """
    Function to load as a pandas dataframe a csv dataset in the standard format

    Args:
        filename (str): the csv file to read. It should be a table with columns Id, Prediction,
            with Id in the form r44_c1 where 44 is the user and 1 is the item

    Returns:
        pandas.DataFrame: ['User', 'Movie', 'Rating']
    """

    df = pd.read_csv(filename)
    df['User'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
    df['Movie'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
    df['Rating'] = df['Prediction']
    df = df.drop(['Id', 'Prediction'], axis=1)
    return df


def load_csv_kaggle():
    """ Function to load the Kaggle CSV submission file sampleSubmission.csv """
    return load_csv('data/sampleSubmission.csv')


def submission_table(original_df, col_userID, col_movie, col_rate):
    """ return table according with Kaggle convention """

    def id(row):
        return 'r' + str(int(row[col_userID])) + '_c' + str(int(row[col_movie]))

    def pred(row):
        return row[col_rate]

    df = pd.DataFrame.copy(original_df)
    df['Id'] = df.apply(id, axis=1)
    df['Prediction'] = df.apply(pred, axis=1)

    return df[['Id', 'Prediction']]


def extract_from_original_table(original_df):
    """ extract User and Movie from kaggle's convention """
    df = pd.DataFrame.copy(original_df)
    df['User'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
    df['Movie'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
    df['Rating'] = df['Prediction']
    df = df.drop(['Id', 'Prediction'], axis=1)
    return df


def split(df, cut):
    """ split table into train and test set """
    size = df.shape[0]
    keys = list(range(size))
    np.random.shuffle(keys)

    key_cut = int(size * cut)
    test_key = keys[:key_cut]
    train_key = keys[key_cut:]

    test = df.loc[test_key]
    train = df.loc[train_key]

    return train, test


def evaluate(prediction, truth):
    """ compute RMSE for pandas.DataFrame prediction table """
    truth_sorted = truth.sort_values(['Movie', 'User']).reset_index(drop=True)
    prediction_sorted = prediction.sort_values(['Movie', 'User']).reset_index(drop=True)

    truth_sorted['square_error'] = np.square(truth_sorted['Rating'] - prediction_sorted['Rating'])

    mse = truth_sorted['square_error'].mean()
    rmse = np.sqrt(mse)

    return rmse


def blender(models, weights):
    """
    Blend models according with different weights

    Args:
        models (dict): keys: model name; values: pandas.DataFrame predicitons
        weights (dict): keys: model name; values: weights

    Returns:
        pandas.DataFrame: weighted blending
    """
    if len(models) != len(weights):
        print("[WARNING] size(predictions) != size(weights)")

    # initiate a DF with desired user/movie key and null prediction
    blend = unified_ordering(list(models.values())[0])
    blend['Rating'] = 0.

    # sum weighted predictions
    for model in models.keys():
        blend['Rating'] += \
            weights[model] * unified_ordering(models[model])['Rating']
            
    pred = list(blend['Rating'])
    
    for i in range(len(pred)):
        if pred[i] > 5:
            pred[i] = 5
        elif pred[i] < 1:
            pred[i] = 1
    
    blend['Rating'] = pred

    return blend

def unified_ordering(df):
    """ Order pandas.DataFrame by ('Movie', 'User') and reset index """
    return df.sort_values(['Movie', 'User']).reset_index(drop=True)


def create_folder(folder_name):
    """ check if folder exists to avoid error and create it if not """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
        
def time_str(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    
    if h > 0:
        str_ = "%d hour %d min. %d sec."%(h, m, s)
    elif m > 0:
        str_ = "%d min. %d sec."%(m, s)
    else:
        str_ = "%.3f sec."%s
    return str_  
    
def save(obj, name):
    pickle.dump(obj, open(name, 'wb'))      

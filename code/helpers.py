#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>
#
# Distributed under terms of the MIT license.


"""
Main helpers functions
"""

import pandas as pd
import numpy as np
import os
import sys

#from pyspark.mllib.recommendation import ALS
#from pyspark import *
#from pyspark.sql import *


def load_csv(filename='data/data_train.csv'):
    ''' Function to load as a pandas dataframe a csv dataset in the standard format 
    
    @ params
        - filename, the csv file to read. It should be a table with columns Id, Prediction, with Id in the form r44_c1
            where 44 is the user and 1 is the item
    @ returns 
        - df, a pandas dataframe with columns User, Movie, Rating
    '''
    df = pd.read_csv(filename)
    df['User'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
    df['Movie'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
    df['Rating'] = df['Prediction']
    df = df.drop(['Id', 'Prediction'], axis=1)
    return df


def load_csv_kaggle():
    ''' Function to load the Kaggle CSV submission file sampleSubmission.csv
    '''
    
    return load_csv('data/sampleSubmission.csv')


def submission_table(original_df, col_userID, col_movie, col_rate):
    ''' Function to compose the submission dataframe from the standard database with 3 columns specified as parameters
    
    @ params
        - original_df, the PD dataframe 
        - col_userID, the user column in the original_df dataframe
        - col_movie, the movie column in the original_df dataframe
        - col_rate, the rating column in the original_df dataframe
        
    '''
    def id(row):
        return 'r' + str(int(row[col_userID])) + '_c' + str(int(row[col_movie]))

    def pred(row):
        return row[col_rate]

    df = pd.DataFrame.copy(original_df)
    df['Id'] = df.apply(id, axis=1)
    df['Prediction'] = df.apply(pred, axis=1)

    return df[['Id', 'Prediction']]


def extract_from_original_table(original_df):
    ''' Function that extract a dataframe with columns User, Movie, Rating from the Kaggle dataframe
    '''
    df = pd.DataFrame.copy(original_df)
    df['User'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
    df['Movie'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
    df['Rating'] = df['Prediction']
    df = df.drop(['Id', 'Prediction'], axis=1)
    return df


def split(df, cut):
    ''' Function that randomly splits the dataframe 
    
    @ params
        - df
        - cut, percentage of pairs in the test dataframe
    @ returns
        - train,test
    '''
    
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
    ''' Function that computes the RMSE between the obtained prediction and the real ones
    
    @ params
        - prediction, vector of prediction obtained
        - test_set, PD dataframe with real ratings
    '''
    test2 = test_set.sort_values(['Movie', 'User']).reset_index(drop=True)

    test2['square_error'] = np.square(test2['Rating'] - prediction['Rating'])

    mse = test2['square_error'].mean()
    rmse = np.sqrt(mse)

    return rmse


def signed_evaluate(prediction, test_set):
    ''' Function that computes the mean difference (without abs) between the obtained predictions and the real ratings
    
    @ params
        - prediction, vector of prediction obtained
        - test_set, PD dataframe with real ratings
    '''
    test2 = test_set.sort_values(['Movie', 'User']).reset_index(drop=True)

    test2['error'] = (test2['Rating'] - prediction['Rating'])

    mean_error = test2['error'].mean()

    return mean_error


def blender(models, weights):
    ''' Function that returns a blended prediction of different models
    
    @ params
        - models, a dictionary of PD df with a Rating column
        - weights, a dictionary in the form model_name -> weight
    '''
    if len(models) != len(weights):
        print("[WARNING] size(predictions) != size(weights)")

    # initiate a DF with desired user/movie key and null prediction
    blend = unified_ordering(list(models.values())[0])
    blend['Rating'] = 0.

    # sum weighted predictions
    for model in models.keys():
        blend['Rating'] += \
            weights[model] * unified_ordering(models[model])['Rating']

    return blend

def unified_ordering(df):
    ''' Function that sorts the values of a PD dataframe, by Movie and then by User
    '''
    return df.sort_values(['Movie', 'User']).reset_index(drop=True)
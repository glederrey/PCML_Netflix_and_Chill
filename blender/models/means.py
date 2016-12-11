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
from helpers import *
import pandas as pd

def user_mean(train, test):
    print("[USER_MEAN] applying")
    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x)) 
    means = train.groupby('User').mean()['Rating']

    def line(df):
        df['Rating'] = means.loc[df['User'].iloc[0]]
        return df#[['User', 'Movie', 'Rating']]

    predictions = predictions.groupby('User').apply(line)
    #predictions = predictions.apply(line, axis=1)

    # integer for id
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    print("[USER_MEAN] done")
    return predictions


def movie_mean(train, test):
    print("[MOVIE_MEAN] applying")

    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x)) 
    means = train.groupby('Movie').mean()['Rating']

    def line(df):
        df['Rating'] = means.loc[df['Movie'].iloc[0]]
        return df#[['User', 'Movie', 'Rating']]

    predictions = predictions.groupby('Movie').apply(line)
    #predictions = predictions.apply(line, axis=1)

    # integer for id
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    print("[MOVIE_MEAN] done")
    return predictions


def global_mean(train, test):
    print("[GLOBAL_MEAN] applying")

    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x)) 

    mean = train['Rating'].mean()
       
    #def line(df):
    #    df['Rating'] = mean
    #    return df[['User', 'Movie', 'Rating']]

    #predictions = predictions.apply(line, axis=1)
    
    predictions.Rating=mean
    
    # integer for id
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    print("[GLOBAL_MEAN] done")
    return predictions
    
def movie_mean_deviation_user(train, test):
    print("[MOVIE_MEAN_DEVIATION_USER] applying")
    
    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x)) 
    means = train.groupby('Movie').mean()['Rating']
    deviation = pd.read_csv('../data/deviations_per_users.csv')
    
    def line(df):
        df['Rating'] = means.loc[int(df['Movie'])] + deviation.loc[int(df['User'])].dev 
        return df#[['User', 'Movie', 'Rating']]
        
    predictions = predictions.apply(line, axis=1)
    
    #predictions['Rating'] = np.where(predictions['Rating'] > 5, predictions['Rating'], 5)
    #predictions['Rating'] = np.where(predictions['Rating'] < 1, predictions['Rating'], 1)
    
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    print("[MOVIE_MEAN_DEVIATION_USER] done")
    return predictions    

    

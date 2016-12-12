#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>
#
# Distributed under terms of the MIT license.
"""
median prediction method, assigning movie/user/global median to all items
"""

import numpy as np
from helpers import *
import pandas as pd

def user_median(train, test):
    print("[USER_MEDIAN] applying")
    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x)) 
    medians = train.groupby('User').median()['Rating']

    def line(df):
        df['Rating'] = medians.loc[df['User'].iloc[0]]
        return df#[['User', 'Movie', 'Rating']]

    predictions = predictions.groupby('User').apply(line)
    #predictions = predictions.apply(line, axis=1)

    # integer for id
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    print("[USER_MEDIAN] done")
    return predictions


def movie_median(train, test):
    print("[MOVIE_MEDIAN] applying")

    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x)) 
    medians = train.groupby('Movie').median()['Rating']

    def line(df):
        df['Rating'] = medians.loc[df['Movie'].iloc[0]]
        return df#[['User', 'Movie', 'Rating']]

    predictions = predictions.groupby('Movie').apply(line)
    #predictions = predictions.apply(line, axis=1)

    # integer for id
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    print("[MOVIE_MEDIAN] done")
    return predictions


def global_median(train, test):
    print("[GLOBAL_MEDIAN] applying")

    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x)) 

    median = train['Rating'].median()
       
    #def line(df):
    #    df['Rating'] = median
    #    return df[['User', 'Movie', 'Rating']]

    #predictions = predictions.apply(line, axis=1)
    
    predictions.Rating=median
    
    # integer for id
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    print("[GLOBAL_MEDIAN] done")
    return predictions
    
def movie_median_deviation_user(train, test):
    print("[MOVIE_MEDIAN_DEVIATION_USER] applying")
    
    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x)) 
    medians = train.groupby('Movie').median()['Rating']
    deviation = pd.read_csv('../data/deviations_per_users.csv')
    
    def line(df):
        df['Rating'] = medians.loc[int(df['Movie'])] + deviation.loc[int(df['User']-1)].dev 
        return df#[['User', 'Movie', 'Rating']]
        
    predictions = predictions.apply(line, axis=1)
    
    #predictions['Rating'] = np.where(predictions['Rating'] > 5, predictions['Rating'], 5)
    #predictions['Rating'] = np.where(predictions['Rating'] < 1, predictions['Rating'], 1)
    
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    print("[MOVIE_MEDIAN_DEVIATION_USER] done")
    return predictions    

    

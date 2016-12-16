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
    ''' Function to assign the user mean to each non labelled element in the test
    
    @ params
        - train, pandas dataframe with columns User, Movie, Rating
        - test, pandas dataframe with columns User, Movie
    @ returns
        - predictions, pandas dataframe with columns User, Movie, Rating
    '''
    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x)) 
    means = train.groupby('User').mean()['Rating']

    def line(df):
        df['Rating'] = means.loc[df['User'].iloc[0]]
        return df

    predictions = predictions.groupby('User').apply(line)

    # integer for id
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    return predictions


def movie_mean(train, test):
    ''' Function to assign the item mean to each non labelled element in the test
    
    @ params
        - train, pandas dataframe with columns User, Movie, Rating
        - test, pandas dataframe with columns User, Movie
    @ returns
        - predictions, pandas dataframe with columns User, Movie, Rating
    '''
    
    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x)) 
    means = train.groupby('Movie').mean()['Rating']

    def line(df):
        df['Rating'] = means.loc[df['Movie'].iloc[0]]
        return df

    predictions = predictions.groupby('Movie').apply(line)

    # integer for id
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    return predictions


def global_mean(train, test):
    ''' Function to assign the global mean to each non labelled element in the test
    
    @ params
        - train, pandas dataframe with columns User, Movie, Rating
        - test, pandas dataframe with columns User, Movie
    @ returns
        - predictions, pandas dataframe with columns User, Movie, Rating
    '''
    
    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x)) 

    mean = train['Rating'].mean()
       
    
    predictions.Rating=mean
    
    # integer for id
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    return predictions
    
def movie_mean_deviation_user(train, test):
    ''' Function to assign to each non labelled element in the test the item mean with a deviation term for each user
    
    @ params
        - train, pandas dataframe with columns User, Movie, Rating
        - test, pandas dataframe with columns User, Movie
    @ returns
        - predictions, pandas dataframe with columns User, Movie, Rating
    '''
    predictions = pd.DataFrame.copy(test)
    predictions.Rating = predictions.Rating.apply(lambda x: float(x)) 
    means = train.groupby('Movie').mean()['Rating']
    deviation = pd.read_csv('../data/deviations_per_users.csv')
    
    def line(df):
    
        df['Rating'] = means.loc[int(df['Movie'])] + deviation.loc[int(df['User'])-1].dev 
        return df
        
    predictions = predictions.apply(line, axis=1)
    
    predictions['User'] = predictions['User'].astype(int)
    predictions['Movie'] = predictions['Movie'].astype(int)

    return predictions

    

#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>
#
# Distributed under terms of the MIT license.
"""
Allow a rescale of the data regarding the user mean, variation and deviation.

BEST: deviation

USE:
    rescaler = Rescaler(df)
    rescaled_train_set = rescaler.normalize_deviation()
    ...
    ... do the ML training which return `predicted_set`...
    ...
    rescaled_test_set = rescaler.recover_deviation(predicted_set)
"""

import numpy as np
import pandas as pd


def dict_mean_user(df):
    ''' Function that returns a dictionary in the form user_index->mean_rating_user
    
    @ params
        - df a PD dataframe with a User and Rating column
    @ returns
        - a dictionary with keys the index of the user and with values the mean of the ratings of that user
    '''
    return dict(df.groupby('User').mean().Rating)


def dict_var_user(df):
    ''' Function that returns a dictionary in the form user_index->variance_rating_user
    
    @ params
        - df a PD dataframe with a User and Rating column
    @ returns
        - a dictionary with keys the index of the user and with values the variance of the ratings of that user
    '''
    return dict(df.groupby('User').var().Rating)


def dict_dev_user(df):
    ''' Function that returns a dictionary in the form user_index->deviation_rating_user
    
    @ params
        - df a PD dataframe with a User and Rating column
    @ returns
        - a dictionary with keys the index of the user and with values the deviation of the ratings of that user with respect of the global mean
    '''
    global_mean = df.groupby('User').mean().Rating.mean()
    return dict(df.groupby('User').mean().Rating - global_mean)
    np.mean(list(dict_mean.values()))


class Rescaler:
    ''' Class that provide a normalized version of the dataframe 
    
    It provides all the method both to normalize the dataframe both to recover the right predictions from the predictions obtained from the normalized dataframe
    '''
    def __init__(self, df):
        ''' Class constructor
        
        It stores internally the dataframe and it computes the most relevant quantities about it.
        
        @ params
            - df, the PD dataframe to normalize
        '''
        self.df = df
        self.variances = dict_var_user(df)
        self.means = dict_mean_user(df)
        self.deviation = dict_dev_user(df)

    def normalize(self):
        ''' Function that returns a user fully normalized version of the dataframe
        
        @ returns 
            - norm_df, a database fully normalized over the users
        '''
        norm_df = pd.DataFrame.copy(self.df)
        norm_df['Rating'] = self.df.apply(
            lambda x: (x['Rating'] - self.means[x['User']]) / self.variances[x['User']],
            axis=1)

        return norm_df

    def recover(self, df):
        ''' Function that recovers the predictions of the non-normalized df from the fully normalized one
        
        @ params
            - df, PD dataframe with Rating and User colums
            
        @ returns
            - recovered_df, a dataframe of the same kind of df but with the correct predictions
        
        '''
        recovered_df = pd.DataFrame.copy(df)
        recovered_df['Rating'] = df.apply(
            lambda x: (x['Rating'] * self.variances[x['User']]) + self.means[x['User']],
            axis=1)

        return recovered_df

    def normalize_only_mean(self):
        ''' Function that returns a user normalized version of the dataframe. ONLY MEAN is considered, not variance
        
        @ returns 
            - norm_df, a database normalized over the users. ONLY MEAN is considered, not variance
        '''
        norm_df = pd.DataFrame.copy(self.df)
        norm_df['Rating'] = self.df.apply(
            lambda x: x['Rating'] - self.means[x['User']],
            axis=1)

        return norm_df

    def recover_only_mean(self, df):
        ''' Function that recovers the predictions of the non-normalized df from the ONLY MEAN normalized one
        
        @ params
            - df, PD dataframe with Rating and User colums
            
        @ returns
            - recovered_df, a dataframe of the same kind of df but with the correct predictions
        
        '''
        recovered_df = pd.DataFrame.copy(df)
        recovered_df['Rating'] = df.apply(
            lambda x: x['Rating'] + self.means[x['User']],
            axis=1)

        return recovered_df

    def normalize_deviation(self):
        ''' Function that returns a user normalized version of the dataframe. ONLY STD is considered, not mean
        
        @ returns 
            - norm_df, a database normalized over the users. ONLY STD is considered, not mean
        '''
        norm_df = pd.DataFrame.copy(self.df)
        norm_df['Rating'] = self.df.apply(
            lambda x: x['Rating'] - self.deviation[x['User']],
            axis=1)

        return norm_df

    def recover_deviation(self, df):
        ''' Function that recovers the predictions of the non-normalized df from the ONLY STD normalized one
        
        @ params
            - df, PD dataframe with Rating and User colums
            
        @ returns
            - recovered_df, a dataframe of the same kind of df but with the correct predictions
        
        '''
        recovered_df = pd.DataFrame.copy(df)
        recovered_df['Rating'] = df.apply(
            lambda x: x['Rating'] + self.deviation[x['User']],
            axis=1)

        return recovered_df
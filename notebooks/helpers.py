#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""
import pandas as pd
import numpy as np
import os
import sys

from pyspark.mllib.recommendation import ALS
from pyspark import *
from pyspark.sql import *


def load_csv(filename='../data/data_train.csv'):
    df = pd.read_csv(filename)
    df['UserID'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
    df['MovieID'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
    df['Rating'] = df['Prediction']
    df = df.drop(['Id', 'Prediction'], axis=1)
    return df


def to_rdd(sqlContext, df):
    df_sql = sqlContext.createDataFrame(df)
    df_rdd = df_sql.rdd
    return df_rdd

def split_train_val_test(proportions, rdd):
    training_RDD, validation_RDD, test_RDD = rdd.randomSplit([6, 2, 2], seed=0)
    validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
    test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))
    return training_RDD, validation_for_predict_RDD, test_for_predict_RDD


def recover_original_table(original_df, col_userID, col_movie, col_rate):
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

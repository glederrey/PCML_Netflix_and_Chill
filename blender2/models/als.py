#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>, Gael Lederrey <gael.lederrey@epfl.ch>,
# Stefano Savare <stefano.savare@epfl.ch>
#
# Distributed under terms of the MIT license.

from pyspark.sql import SQLContext
from pyspark.mllib.recommendation import ALS
from rescaler import Rescaler
import os

''' 
Matrix Factorization using Alternating Least Squares (ALS)
It works with Spark RDD dataframes, but it accepts as input only Pandas dataframes and converts them to keep consistency between methods
'''

def predictions_ALS_rescaling(df_train, df_test, **kwargs):
    """
    ALS with pyspark
    First do a rescaling of the user in a way that they all have the same mean of rating.
    This counter the effect of "mood" of users. Some of them given worst/better grade even if they have the same
    appreciation of a movie.

    :param df_train:
    :param df_test:
    :param kwargs:
        gamma (float): regularization parameter
        n_features (int): number of features for matrices
        n_iter (int): number of iterations
        init_method ('global_mean' or 'movie_mean'): kind of initial matrices (better result with 'global_mean')
    :return:
    """
    rescaler = Rescaler(df_train)
    df_train_normalized = rescaler.normalize_deviation()

    prediction_normalized = predictions_ALS(df_train_normalized, df_test, **kwargs)
    prediction = rescaler.recover_deviation(prediction_normalized)
    return prediction

def predictions_ALS(train_set, test_set, **kwargs):
    """
    Function to return the predictions of an ALS model.

    Args:
        train_set (pandas.DataFrame): train set
        test_set (pandas.DataFrame): test set
        spark_context (spark.SparkContext): spark context
        **kwarg: Arbitrary keyword arguments. Passed to ALS.train()

    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)
    """
    
    os.system('rm -rf metastore_db')
    os.system('rm -rf __pycache__')

    # take spark_context
    spark_context = kwargs.pop('spark_context')

    # Convert pd.DataFrame to Spark.rdd
    sqlContext = SQLContext(spark_context)

    train_sql = sqlContext.createDataFrame(train_set).rdd
    test_sql = sqlContext.createDataFrame(test_set).rdd

    # Train the model
    model = ALS.train(train_sql, **kwargs)

    # Get the predictions
    data_for_predictions = test_sql.map(lambda x: (x[0], x[1]))
    predictions = model.predictAll(data_for_predictions).map(lambda r: ((r[0], r[1]), r[2]))

    # Convert Spark.rdd to pd.DataFrame
    df = predictions.toDF().toPandas()

    # Post processing database
    df['User'] = df['_1'].apply(lambda x: x['_1'])
    df['Movie'] = df['_1'].apply(lambda x: x['_2'])
    df['Rating'] = df['_2']
    df = df.drop(['_1', '_2'], axis=1)
    df = df.sort_values(by=['Movie', 'User'])
    df.index = range(len(df))

    return df

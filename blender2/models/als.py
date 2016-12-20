#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>, Gael Lederrey <gael.lederrey@epfl.ch>,
# Stefano Savare <stefano.savare@epfl.ch>
#
# Distributed under terms of the MIT license.

"""
Matrix Factorization using Alternating Least Squares (ALS) from PySpark
"""

from pyspark.sql import SQLContext
from pyspark.mllib.recommendation import ALS
from rescaler import Rescaler
import os


def predictions_ALS_rescaled(train, test, **kwargs):
    """
    ALS with PySpark rescaled.

    First, a rescaling of the user such that they all have the same average of rating is done.
    Then, the predictions are done using the function prediction_ALS().
    Finally, the predictions are rescaled to recover the deviation of each user.

    Args:
        train (pandas.DataFrame): train set
        test (pandas.DataFrame): test set
        **kwargs: Arbitrary keyword arguments. Directly given to predictions_ALS().

    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)
    """
    # Load the class Rescaler
    rescaler = Rescaler(train)
    # Normalize the train data
    df_train_normalized = rescaler.normalize_deviation()

    # Predict using the normalized trained data
    prediction_normalized = predictions_ALS(df_train_normalized, test, **kwargs)
    # Rescale the prediction to recover the deviations
    prediction = rescaler.recover_deviation(prediction_normalized)
    return prediction


def predictions_ALS(train, test, **kwargs):
    """
    ALS with PySpark.

    Compute the predictions on a test_set after training on a train_set using the method ALS from PySpark.

    Args:
        train (pandas.DataFrame): train set
        test (pandas.DataFrame): test set
        **kwargs: Arbitrary keyword arguments. Passed to ALS.train() (Except for the spark_context)
            spark_context (SparkContext): SparkContext passed from the main program. (Useful when using Jupyter)
            rank (int): Rank of the matrix for the ALS
            lambda (float): Regularization parameter for the ALS
            iterations (int): Number of iterations for the ALS
            nonnegative (bool): Boolean to allow negative values or not.

    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)
    """

    # Delete folders that causes troubles
    os.system('rm -rf metastore_db')
    os.system('rm -rf __pycache__')

    # Extract Spark Context from the kwargs
    spark_context = kwargs.pop('spark_context')

    # Convert pd.DataFrame to Spark.rdd
    sqlContext = SQLContext(spark_context)

    train_sql = sqlContext.createDataFrame(train).rdd
    test_sql = sqlContext.createDataFrame(test).rdd

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

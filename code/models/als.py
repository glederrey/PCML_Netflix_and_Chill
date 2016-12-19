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


def predictions_ALS(train_set, test_set, spark_context, **kwarg):
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

    # Convert pd.DataFrame to Spark.rdd
    sqlContext = SQLContext(spark_context)

    train_sql = sqlContext.createDataFrame(train_set).rdd
    test_sql = sqlContext.createDataFrame(test_set).rdd

    # Train the model
    model = ALS.train(train_sql, **kwarg)

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

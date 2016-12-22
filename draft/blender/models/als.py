import random
from pyspark.sql import SQLContext
from pyspark.mllib.recommendation import ALS
from rescaler import Rescaler

def predictions_ALS_rescaling(df_train, df_test, spark_context, **kwargs):
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

    prediction_normalized = predictions_ALS(df_train_normalized, df_test, spark_context, **kwargs)
    prediction = rescaler.recover_deviation(prediction_normalized)
    return prediction


def predictions_ALS(train_set,test_set,spark_context, **arg):
    ''' Function to return the predictions of an ALS model.

    @ params:
        - train_set, test_set, input PANDAS dataframe
        - spark_context
        - **arg, the parameters passed to ALS.train()
    @ returns:
        - prediction PANDAS dataframe. The column 'Rating' is the column with the sorted predictions

    '''

    print('[ALS] applying')
    sqlContext=SQLContext(spark_context)

    train_sql=sqlContext.createDataFrame(train_set).rdd
    test_sql=sqlContext.createDataFrame(test_set).rdd
    
    # Train the model
    model = ALS.train(train_sql, **arg)
    
    # Get the predictions
    data_for_preditions=test_sql.map(lambda x: (x[0], x[1]))
    predictions = model.predictAll(data_for_preditions).map(lambda r: ((r[0], r[1]), r[2]))
    
    # Convert to Pandas
    pred_df = predictions.toDF().toPandas()
    
    # Post processing database
    pred_df['User'] = pred_df['_1'].apply(lambda x: x['_1'])
    pred_df['Movie'] = pred_df['_1'].apply(lambda x: x['_2'])
    pred_df['Rating'] = pred_df['_2']
    pred_df = pred_df.drop(['_1', '_2'], axis=1)
    pred_df = pred_df.sort_values(by=['Movie', 'User'])
    pred_df.index = range(len(pred_df))
    print('[ALS] done')
    return pred_df


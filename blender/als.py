import random
from pyspark.sql import SQLContext
from pyspark.mllib.recommendation import ALS


def predictions_ALS(train_set,test_set,spark_context,**arg):
    ''' Function to return the predictions of an ALS model.

    @ params:
        - train_set, test_set, input PANDAS dataframe
        - spark_context
        - **arg, the parameters passed to ALS.train()
    @ returns:
        - prediction PANDAS dataframe. The column 'Rating' is the column with the sorted predictions

    '''

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
    return pred_df


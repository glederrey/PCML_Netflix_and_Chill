import random
from pyspark.sql import SQLContext
from pyspark.mllib.recommendation import ALS

''' 
Matrix Factorization using Alternating Least Squares (ALS)
It works with Spark RDD dataframes, but it accepts as input only Pandas dataframes and converts them to keep consistency between methods
'''

def predictions_ALS(train_set,test_set,spark_context, **arg):
    ''' Function to return the predictions of an ALS model.

    @ params:
        - train_set, test_set, input PANDAS dataframe
        - spark_context, necessary to run Spark algorithms
        - **arg, the parameters passed to ALS.train()
    @ returns:
        - prediction PANDAS dataframe. The column 'Rating' is the column with the sorted predictions

    '''

    # sql context initialized from spark_context
    sqlContext=SQLContext(spark_context)

    # Conversion of Pandas df to RDD dataframes
    train_sql=sqlContext.createDataFrame(train_set).rdd
    test_sql=sqlContext.createDataFrame(test_set).rdd
    
    # Train the model
    model = ALS.train(train_sql, **arg)
    
    # Get the predictions
    data_for_preditions=test_sql.map(lambda x: (x[0], x[1]))
    predictions = model.predictAll(data_for_preditions).map(lambda r: ((r[0], r[1]), r[2]))
    
    # Convert to Pandas
    pred_df = predictions.toDF().toPandas()
    
    # Post processing database to get the predictions vector
    pred_df['User'] = pred_df['_1'].apply(lambda x: x['_1'])
    pred_df['Movie'] = pred_df['_1'].apply(lambda x: x['_2'])
    pred_df['Rating'] = pred_df['_2']
    pred_df = pred_df.drop(['_1', '_2'], axis=1)
    pred_df = pred_df.sort_values(by=['Movie', 'User'])
    pred_df.index = range(len(pred_df))
    return pred_df


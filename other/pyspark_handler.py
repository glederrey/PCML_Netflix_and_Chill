import numpy as np
import pandas as pd
from pyspark.mllib.recommendation import ALS
import pyspark
import itertools



class Recommander:

    def __init__(self):
        coucou = 'coucou'

    def load_csv(self, filename):
        train = pd.read_csv('../data/data_train.csv')
        train['UserID'] = train['Id'].apply(lambda x: int(x.split('_')[0][1:]))
        train['MovieID'] = train['Id'].apply(lambda x: int(x.split('_')[1][1:]))
        train['Rating'] = train['Prediction']
        train = train.drop(['Id', 'Prediction'], axis=1)
        return train

    def to_rdd(self, df):
        train_sql = sqlContext.createDataFrame(df)
        train_rdd = train_sql.rdd
        return train_rdd




# # Collaborative Filtering
# 
# For this Recommender System, we will use a collaborative filtering recommender system. Indeed, we don't have any information about the movies. Therefore, we will use the ratings of other users to guess the rating of a user. 
# 
# The Spark MLib provides a Collaborative Filtering implementation by using Alternating Least Squares. We will need the following parameters:
# 
# - `numBlocks`: Number of blocks used to parallelize computation (-1 for auto-configure)
# - `rank`: Number of latent factors in the model
# - `iterations`: Number of iterations
# - `lambda`: Regularization parameter in ALS
# - `implicitPrefs`: Specify whether to use the explicit feedback ALS variant or one adapted for implicit feedback data
# - `alpha`: Parameter applicable to the implicit feedback variant of ALS that governs the baseline confidence in preference observations
# 

# In[8]:

# Split into train, validation and test datasets
training_RDD, validation_RDD, test_RDD = train_rdd.randomSplit([6, 2, 2], seed=0)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))


# ## Training phase!

# In[9]:

def computeRMSE(model, data, prediction):
    predictions = model.predictAll(data).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = prediction.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    return error


# In[10]:

ranks = [2, 4, 6, 8, 10, 12]
lambdas = [0.1, 0.5, 1.0, 5.0, 10.0]
numIters = [5, 10, 20]
nbr_models = len(ranks)*len(lambdas)*len(numIters)

bestModel = None
bestValidationRmse = float("inf")
bestRank = 0
bestLambda = -1.0
bestNumIter = -1


# In[11]:

i = 0
for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
    try:
        model = ALS.train(training_RDD, rank, numIter, lmbda)
        validationRmse = computeRMSE(model, validation_for_predict_RDD, validation_RDD)
        print("Model %i/%i: RMSE (validation) = %f" %(i+1, nbr_models, validationRmse))
        print("  Trained with rank = %d, lambda = %.1f, and numIter = %d." % (rank, lmbda, numIter))
        print("")
        if (validationRmse < bestValidationRmse):
            bestModel = model
            bestValidationRmse = validationRmse
            bestRank = rank
            bestLambda = lmbda
            bestNumIter = numIter
    except:
        print("Model %i/%i failed!" %(i+1, nbr_models))
        print("  Parameters: rank = %d, lambda = %.1f, and numIter = %d." % (rank, lmbda, numIter))

    i += 1
    
# Evaluate the best model on the training set
print("The best model was trained with rank = %d and lambda = %.1f, " % (bestRank, bestLambda)   + "and numIter = %d, and its RMSE on the training set is %f" % (bestNumIter, bestValidationRmse))

# Evaluate the best model on the test set
testRmse = computeRMSE(bestModel, test_for_predict_RDD, test_RDD)
print("RMSE on the test set: %f"%(testRmse))


# Now, that we have the best rank, best lambda and best number of iterations, we can train on the whole train data.

# In[12]:

perfect_model = ALS.train(train_rdd, bestRank, bestNumIter, bestLambda)


# # Load and prepare the test data

# In[13]:

test = pd.read_csv('../data/sampleSubmission.csv')
test.head()


# In[14]:

# Prepare test for RDD
test_prep = test
test_prep['UserID'] = test_prep['Id'].apply(lambda x: int(x.split('_')[0][1:]))
test_prep['MovieID'] = test_prep['Id'].apply(lambda x: int(x.split('_')[1][1:]))
test_prep['Rating'] = test_prep['Prediction']
test_prep = test_prep.drop(['Prediction', 'Id'], axis=1)
test_prep.head()


# In[15]:

# First, we transform it using sqlContect
test_sql = sqlContext.createDataFrame(test_prep)
test_rdd = test_sql.rdd
test_rdd.take(3)


# In[16]:

test_RDD_Kaggle = test_rdd.map(lambda x: (x[0], x[1]))
predictions = perfect_model.predictAll(test_RDD_Kaggle).map(lambda r: ((r[0], r[1]), r[2]))


# In[17]:

predictions.take(3)


# In[18]:

pred_df = predictions.toDF().toPandas()


# In[19]:

pred_df.head()


# In[20]:

pred_df['UserID'] = pred_df['_1'].apply(lambda x: x['_1'])
pred_df['MovieID'] = pred_df['_1'].apply(lambda x: x['_2'])
pred_df['Rating'] = pred_df['_2']
pred_df = pred_df.drop(['_1', '_2'], axis=1)
pred_df.head()


# In[21]:

pred_df = pred_df.sort_values(by=['MovieID', 'UserID'])
pred_df.head()


# In[22]:

pred_df.index = range(len(pred_df))


# In[23]:

test['Prediction'] = pred_df['Rating']


# In[24]:

test.head()


# In[25]:

test = test.drop(['UserID', 'MovieID', 'Rating'], axis=1)


# In[26]:

test.to_csv('pred_pyspark_als.csv', index=False)


# In[ ]:




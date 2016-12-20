"""
Models defined in library Surprise
"""

from surprise import *
import numpy as np
import pandas as pd
from rescaler import Rescaler

def knn_rescaling(df_train, df_test, **kwargs):
    rescaler = Rescaler(df_train)
    df_train_normalized = rescaler.normalize_deviation()

    prediction_normalized = knn(df_train_normalized, df_test, **kwargs)
    prediction = rescaler.recover_deviation(prediction_normalized)
    return prediction

def knn(train, test, **kwargs): 
    # Get parameters
    k = kwargs['k']
    sim_options = kwargs['sim_options']

    # First, we need to dump the pandas DF into files
    train_file = 'tmp_train.csv'
    test_file = 'tmp_test.csv'
    train.to_csv(train_file, index=False, header=False)
    test.to_csv(test_file, index=False, header=False)
    
    # Create Reader
    reader = Reader(line_format='user item rating', sep=',')
    
    # Train and test set for Surprise
    fold = [(train_file, test_file)]
    
    # Load the data
    data = Dataset.load_from_folds(fold, reader=reader)
    
    # Algorithm
    algo = KNNBaseline(k=k, sim_options=sim_options)
    
    # Go through 1 fold
    for trainset, testset in data.folds():
        # Train
        algo.train(trainset)
        
        # Predict
        predictions = algo.test(testset)
        
    # Postprocess the predictions
    pred = np.zeros(len(predictions))
    for i in range(len(predictions)):
        val = predictions[i].est
        if val > 5:
            pred[i] = 5
        elif val < 1:
            pred[i] = 1
        else:
            pred[i] = val
            
    # Copy the test
    df_return = test.copy()
    df_return.Rating = pred
    
    return df_return
 
def svd_rescaling(df_train, df_test, **kwargs):
    rescaler = Rescaler(df_train)
    df_train_normalized = rescaler.normalize_deviation()

    prediction_normalized = svd(df_train_normalized, df_test, **kwargs)
    prediction = rescaler.recover_deviation(prediction_normalized)
    return prediction
       
def svd(train, test, **kwargs): 
    # Get parameters
    n_epochs = kwargs['n_epochs']
    lr_all = kwargs['lr_all']
    reg_all = kwargs['reg_all']

    # First, we need to dump the pandas DF into files
    train_file = 'tmp_train.csv'
    test_file = 'tmp_test.csv'
    train.to_csv(train_file, index=False, header=False)
    test.to_csv(test_file, index=False, header=False)
    
    # Create Reader
    reader = Reader(line_format='user item rating', sep=',')
    
    # Train and test set for Surprise
    fold = [(train_file, test_file)]
    
    # Load the data
    data = Dataset.load_from_folds(fold, reader=reader)
    
    # Algorithm
    algo = SVD(n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
    
    # Go through 1 fold
    for trainset, testset in data.folds():
        # Train
        algo.train(trainset)
        
        # Predict
        predictions = algo.test(testset)
        
    # Postprocess the predictions
    pred = np.zeros(len(predictions))
    for i in range(len(predictions)):
        val = predictions[i].est
        if val > 5:
            pred[i] = 5
        elif val < 1:
            pred[i] = 1
        else:
            pred[i] = val
            
    # Copy the test
    df_return = test.copy()
    df_return.Rating = pred
    
    return df_return

def slope_one_rescaling(df_train, df_test):
    rescaler = Rescaler(df_train)
    df_train_normalized = rescaler.normalize_deviation()

    prediction_normalized = slope_one(df_train_normalized, df_test)
    prediction = rescaler.recover_deviation(prediction_normalized)
    return prediction
    
def slope_one(train, test):
    # First, we need to dump the pandas DF into files
    train_file = 'tmp_train.csv'
    test_file = 'tmp_test.csv'
    train.to_csv(train_file, index=False, header=False)
    test.to_csv(test_file, index=False, header=False)
    
    # Create Reader
    reader = Reader(line_format='user item rating', sep=',')
    
    # Train and test set for Surprise
    fold = [(train_file, test_file)]
    
    # Load the data
    data = Dataset.load_from_folds(fold, reader=reader)
    
    # Algorithm
    algo = SlopeOne()
    
    # Go through 1 fold
    for trainset, testset in data.folds():
        # Train
        algo.train(trainset)
        
        # Predict
        predictions = algo.test(testset)
        
    # Postprocess the predictions
    pred = np.zeros(len(predictions))
    for i in range(len(predictions)):
        val = predictions[i].est
        if val > 5:
            pred[i] = 5
        elif val < 1:
            pred[i] = 1
        else:
            pred[i] = val
            
    # Copy the test
    df_return = test.copy()
    df_return.Rating = pred
    
    return df_return

def baseline_rescaling(df_train, df_test):
    rescaler = Rescaler(df_train)
    df_train_normalized = rescaler.normalize_deviation()

    prediction_normalized = baseline(df_train_normalized, df_test)
    prediction = rescaler.recover_deviation(prediction_normalized)
    return prediction
   
def baseline(train, test):
    # First, we need to dump the pandas DF into files
    train_file = 'tmp_train.csv'
    test_file = 'tmp_test.csv'
    train.to_csv(train_file, index=False, header=False)
    test.to_csv(test_file, index=False, header=False)
    
    # Create Reader
    reader = Reader(line_format='user item rating', sep=',')
    
    # Train and test set for Surprise
    fold = [(train_file, test_file)]
    
    # Load the data
    data = Dataset.load_from_folds(fold, reader=reader)
    
    # Algorithm
    algo = SlopeOne()
    
    # Go through 1 fold
    for trainset, testset in data.folds():
        # Train
        algo.train(trainset)
        
        # Predict
        predictions = algo.test(testset)
        
    # Postprocess the predictions
    pred = np.zeros(len(predictions))
    for i in range(len(predictions)):
        val = predictions[i].est
        if val > 5:
            pred[i] = 5
        elif val < 1:
            pred[i] = 1
        else:
            pred[i] = val
            
    # Copy the test
    df_return = test.copy()
    df_return.Rating = pred
    
    return df_return

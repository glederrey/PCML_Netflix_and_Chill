#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Helpers function to work with scipy.sparse matrices
Used for matrix factorization using SGD (MF-SGD)
"""

import scipy.sparse as sp
import pandas as pd
import numpy as np
from collections import defaultdict

def load_csv(filename='../data/data_train.csv'):
    ''' Function to load as a pandas dataframe a csv dataset in the standard format 
    
    @ params
        - filename, the csv file to read. It should be a table with columns Id, Prediction, with Id in the form r44_c1
            where 44 is the user and 1 is the item
    @ returns 
        - df, a pandas dataframe with columns User, Movie, Rating
    '''
    df = pd.read_csv(filename)
    df['User'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
    df['Movie'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
    df['Rating'] = df['Prediction']
    df = df.drop(['Id', 'Prediction'], axis=1)
    return df


"""

Specific for Scipy

"""


def sp_to_df(sparse):
    ''' Function to covert a sparse Scipy matrix to a Pandas dataframe
    
    @ params
        - sparse, a Scipy sparse matrix
    @ returns
        - df, a pandas dataframe with columns User, Movie, Rating
    '''
    row, col, rat = sp.find(sparse)
    row += 1
    col += 1

    df = pd.DataFrame({'User': row, 'Movie': col, 'Rating': rat})
    df = df[['User', 'Movie', 'Rating']].sort_values(['Movie', 'User'])
    return df


def df_to_sp(df):
    ''' Function to convert a pandas dataframe to a Scipy sparse matrix
    
    @ params
        - df, a pandas dataframe with columns User, Movie, Rating
    @ returns
        - sp_matrix, a Scipy sparse matrix
    '''
    n_user = df['User'].max()
    n_movie = df['Movie'].max()

    sp_matrix = sp.lil_matrix((n_user, n_movie))

    users = df['User']
    movies = df['Movie']
    ratings = df['Rating']

    for u, m, r in zip(users, movies, ratings):
        sp_matrix[u - 1, m - 1] = r

    return sp_matrix


def read_txt(path):
    """ Function to read text file from path.
    
    @ params
        - path, a string
    @ returns
        - a list of string, each is a line of the file in path
    
    """
    with open(path, "r") as f:
        return f.read().splitlines()


def load_data(path):
    """ Load data in text format as a scipy sparse matrix, one rating per line, as in the kaggle competition.
    
    @ params
        - path, a string containing the file to read
    @ returns
        - a Scipy sparse matrix corresponding to the file read
    """
    data = read_txt(path)[1:]
    return preprocess_data(data)


def preprocess_data(data):
    """ Function that convert a list of strings to a Scipy sparse matrix
    
    @ params
        - data, list of string. Each string is in the form r44_c1 where 44 is the user and 1 is the item
    @ returns
        - ratings, a Scipy sparse matrix containing the data read
    """

    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of users: {}, number of movies: {}".format(max_row, max_col))

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings


def percent_sparse(matrix):
    ''' Function to get the percentage of non-sparse item in a Scipy sparse matrix
    '''
    return matrix.nnz / (matrix.shape[0] * matrix.shape[1] )


def nonzero_mean(matrix):
    ''' Return the mean of the non-zero items of a Scipy sparse matrix 
    '''
    return matrix.sum() / matrix.nnz


def rmse(predicted, testset):
    ''' Function that computes the RMSE of two input matrix
    '''
    if type(predicted) == list:
        loss = np.subtract(testset, predicted)
    else:
        loss = (testset - predicted).data
    loss_square = np.square(loss)
    mse = np.mean(loss_square)
    rmse = np.sqrt(mse)
    return rmse


def counter(matrix, axis):
    ''' Function that count the non-zero element for each row/column of a Scipy sparse matrix
    
    @ params
        - matrix, Scipy sparse matrix
        - axis, if 0, count on the rows, if 1, count on the columns
    @ returns
        - a dictionary, with keys the index of the rows/columns and values the number of non-zero elements on it
    '''
    n_row = matrix.nonzero()[axis]

    d = defaultdict(int)
    for i in n_row:
        d[i] += 1
    return d


def nonzero_row_mean(matrix):
    ''' Function that returns the mean for each row in a Scipy sparse matrix
        
        @ returns
            - res, a dictionary containing the mean for each row
    '''
    size = matrix.shape[0]
    res = np.zeros(size)
    sum_ = matrix.sum(axis=1)
    n_ = counter(matrix, axis=0)

    for i in range(size):
        if n_[i] == 0:
            # if movie never rated, take global mean
            res[i] = nonzero_mean(matrix)
        else:
            # else take mean
            res[i] = sum_[i] / n_[i]
    return res


def nonzero_column_mean(matrix):
    ''' Function that 
    '''
    size = matrix.shape[1]
    res = np.zeros(size)
    sum_ = matrix.sum(axis=0)
    n_ = counter(matrix, axis=1)

    for i in range(size):
        a = sum_[:, i] / n_[i]
        res[i] = a

    return res


def compute_error(data, user_features, item_features, nz):
    """compute the loss (RMSE) of the prediction of nonzero elements."""

    square_errors = []
    for i, j in nz:
        square_errors.append(np.square(data[i, j] - np.dot(item_features[i, :], user_features[j, :].T)))

    mse = np.mean(square_errors)
    rmse = np.sqrt(mse)
    return rmse

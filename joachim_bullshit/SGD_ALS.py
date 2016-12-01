#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>
#
# Distributed under terms of the MIT license.

import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


# http://people.mpi-inf.mpg.de/~rgemulla/publications/gemulla11dsgd-short.pdf

def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()

def load_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)


def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""
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
    print("number of items: {}, number of users: {}".format(max_row, max_col))

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings

def nonzero_mean(matrix):
    return matrix.sum() / matrix.nnz


def rmse(predicted, testset):
    loss = (testset - predicted).data
    loss_square = np.square(loss)
    mse = np.mean(loss_square)
    rmse = np.sqrt(mse)
    return rmse


def counter(matrix, axis):
    n_row = matrix.nonzero()[axis]

    d = defaultdict(int)
    for i in n_row:
        d[i] += 1
    return d


def nonzero_row_mean(matrix):
    size = matrix.shape[0]
    res = np.zeros(size)
    sum_ = matrix.sum(axis=1)
    n_ = counter(matrix, axis=0)

    for i in range(size):
        a = sum_[i] / n_[i]
        res[i] = a

    return res


def nonzero_column_mean(matrix):
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
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # calculate rmse (we only consider nonzero entries.)
    # ***************************************************

    square_errors = []
    for i, j in nz:
        square_errors.append(np.square(data[i, j] - np.dot(item_features[i, :], user_features[j, :].T)))

    mse = np.mean(square_errors)
    rmse = np.sqrt(mse)
    return rmse

def split_data(ratings, num_items_per_user, num_users_per_item,
               min_num_ratings, p_test=0.1):
    """split the ratings to training data and test data.
    Args:
        min_num_ratings:
            all users and items we keep must have at least min_num_ratings per user and per item.
    """
    # set seed
    np.random.seed(22)

    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    valid_ratings = ratings[valid_items, :][:, valid_users]

    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data and return train and test data. TODO
    # NOTE: we only consider users and movies that have more
    # than 10 ratings
    # ***************************************************

    # initiate empty test matrix and full train matrix
    test = sp.lil_matrix(valid_ratings.shape, dtype=np.float64)
    train = sp.lil_matrix.copy(valid_ratings)

    # number of test elements
    n_test = int(train.nnz * p_test)
    ind = np.arange(train.nnz)
    np.random.shuffle(ind)
    ind = ind[:n_test]  # list of 10% of indicies between 0 and nnz

    rows, cols = train.nonzero()

    for i in ind:
        # get right coordonate
        row = rows[i]
        col = cols[i]

        # put this into test matrix
        test[row, col] = valid_ratings[row, col]

        # clear it from train matrix
        train[row, col] = 0.

    print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
    print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return valid_ratings, train, test


def plot_train_test_data(train, test):
    """visualize the train and test data."""
    ylim, xlim = train.shape

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.spy(train, precision=0.01, markersize=0.5)
    ax1.set_xlabel("Users")
    ax1.set_ylabel("Items")
    ax1.set_title("Training data")
    ax1.set_xlim(0, xlim)
    ax1.set_ylim(ylim, 0)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.spy(test, precision=0.01, markersize=0.5)
    ax2.set_xlabel("Users")
    ax2.set_ylabel("Items")
    ax2.set_title("Test data")
    ax2.set_xlim(0, xlim)
    ax2.set_ylim(ylim, 0)
    plt.tight_layout()
    plt.savefig("train_test")
    plt.show()





def baseline_global_mean(train, test):
    """baseline method: use the global mean."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************

    # Global mean
    mean = nonzero_mean(train)
    prediction = sp.lil_matrix(train.shape, dtype=np.float64)
    row, col = test.nonzero()

    for i, j in zip(row, col):
        prediction[i, j] = mean

    return rmse(prediction, test)


def baseline_user_mean(train, test):
    """baseline method: use the user means as the prediction."""
    mse = 0
    num_items, num_users = train.shape

    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    user_mean = nonzero_row_mean(train)

    prediction = sp.lil_matrix(train.shape, dtype=np.float64)
    row, col = test.nonzero()

    for i, j in zip(row, col):
        prediction[i, j] = user_mean[i]

    return rmse(prediction, test)


def baseline_item_mean(train, test):
    """baseline method: use item means as the prediction."""
    mse = 0
    num_items, num_users = train.shape

    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    item_mean = nonzero_column_mean(train)

    prediction = sp.lil_matrix(train.shape, dtype=np.float64)
    row, col = test.nonzero()

    for i, j in zip(row, col):
        prediction[i, j] = item_mean[j]

    return rmse(prediction, test)


def init_MF(train, num_features, method='global_mean'):
    """init the parameter for matrix factorization."""

    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # you should return:
    #     user_features: shape = num_user, num_features (matrix Z)
    #     item_features: shape = num_item, num_features (matrix W)
    # ***************************************************

    num_item, num_user = train.shape

    #     user_features = sp.lil_matrix((num_user, num_features), dtype=np.float64)
    #     item_features = sp.lil_matrix((num_item, num_features), dtype=np.float64)

    # fill matrices in a way that their first multiplication give the mean
    if method == 'global_mean':
        m = train.sum() / train.nnz
        x = np.sqrt(m / num_features)
        user_features = np.full((num_user, num_features), x, dtype=np.float64)
        item_features = np.full((num_item, num_features), x, dtype=np.float64)

    if method == 'movie_mean':
        user_features = np.random.rand(num_user, num_features) / 1000.
        item_features = np.random.rand(num_item, num_features) / 1000.

        item_means = nonzero_row_mean(train)
        item_features[:, 0] = item_means

        user_ones = np.ones(train.shape[1])
        user_features[:, 0] = user_ones

    return user_features, item_features


def matrix_factorization_SGD(train, test):
    """matrix factorization by SGD."""
    # define parameters
    gamma = 0.01
    num_features = 20  # K in the lecture notes
    lambda_user = 0.1
    lambda_item = 0.7
    num_epochs = 20  # number of full passes through the train set
    #     num_epochs = 1
    errors = [0]

    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, num_features, 'movie_mean')

    # find the non-zero ratings indices
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):
        # shuffle the training rating indices
        np.random.shuffle(nz_train)

        # decrease step size
        gamma /= 1.2

        for i, (d, n) in enumerate(nz_train):
            size = len(nz_train)
            if i % 10000 == 0 and False:
                print("step", i, "over", size)
                # d: train_rows (items), n: train_columns (users)

                # ***************************************************
                # INSERT YOUR CODE HERE
                # TODO
                # do matrix factorization.
                # ***************************************************
            loss = train[d, n] - np.dot(item_features[d, :], user_features[n, :].T)
            grad_item = loss * user_features[n, :]
            grad_user = loss * item_features[d, :]

            item_features[d, :] += gamma * grad_item
            user_features[n, :] += gamma * grad_item
        # add_scalar_row(item_features, d, descent)
        #             add_scalar_row(user_features, n, descent)

        #             print(compute_error(train, user_features, item_features, nz_train))

        rmse = compute_error(train, user_features, item_features, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))

        errors.append(rmse)
    # ***************************************************
    # TODO
    # evaluate the test error.
    # ***************************************************
    rmse = compute_error(test, user_features, item_features, nz_test)
    print("RMSE on test data: {}.".format(rmse))

    return item_features, user_features, errors


def update_user_feature(
        train, item_features, lambda_user,
        nnz_items_per_user, nz_user_itemindices):
    """update user feature matrix."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # update and return user feature.
    # ***************************************************
    n_features = item_features.shape[1]
    lambdas_diag = sp.identity(n_features, dtype=np.float64) * lambda_user

    return np.linalg.inv(item_features.T @ item_features + lambdas_diag) @ item_features.T @ train


def update_item_feature(
        train, user_features, lambda_item,
        nnz_users_per_item, nz_item_userindices):
    """update item feature matrix."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # update and return item feature.
    # ***************************************************
    n_features = user_features.shape[1]
    lambdas_diag = sp.identity(n_features, dtype=np.float64) * lambda_item

    return np.linalg.inv(user_features.T @ user_features + lambdas_diag) @ user_features.T @ train.T


def ALS(train, test):
    """Alternating Least Squares (ALS) algorithm."""
    # define parameters
    num_features = 20  # K in the lecture notes
    lambda_user = 0.1
    lambda_item = 0.7
    stop_criterion = 1e-4
    change = 1
    error_list = [0]

    # set seed
    np.random.seed(988)

    # init ALS
    user_features, item_features = init_MF(train, num_features, 'movie_mean')

    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # start you ALS-WR algorithm.
    # ***************************************************
    for it in range(100):
        user_features = update_user_feature(train, item_features, lambda_user, 0, 0).T
        item_features = update_item_feature(train, user_features, lambda_item, 0, 0).T

        rmse = compute_error(train, user_features, item_features, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))

        error_list.append(rmse)
        if np.abs(rmse - error_list[-2]) < stop_criterion:
            break

    rmse = compute_error(test, user_features, item_features, nz_test)
    print("RMSE on test data: {}.".format(rmse))

    return item_features, user_features, error_list
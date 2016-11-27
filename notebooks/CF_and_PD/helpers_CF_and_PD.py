#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"                                                                                                                     "
"        In this file, we will provide different functions that are used by the program we wrote for the project      "
"        "Recommender System" in the course PCML at EPFL, Fall 2016.                                                  "
"                                                                                                                     "
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import pandas as pd
import numpy as np
import multiprocessing
import pickle

import matplotlib.pyplot as plt
import itertools
from joblib import Parallel, delayed



def prepare_data(dataset):
    """

        This function reads the csv file for the dataset and returns the data. We also add two columns to the
        pandas DataFrame.

    :param      dataset: CSV file containing the data set

    :return:    data: pandas DataFrame containing the original columns of the CSV file plus the UserID
                      and MovieID
    """
    print("Load the dataset")

    data = pd.read_csv(dataset)
    data['UserID'] = data['Id'].apply(lambda x: int(x.split('_')[0][1:]) - 1)
    data['MovieID'] = data['Id'].apply(lambda x: int(x.split('_')[1][1:]) - 1)

    return data

def variables(data):
    """
        Using the data extracted in the function prepare_data, we extract more specific data.

    :param      data: pandas DataFrame extracted using the function prepare_data

    :return:    U: Array of users
                I: Array of items
                R: Ratings of items by user
                I_u: List of array of items rated by each user

    """

    print("Prepare the variables")

    # Users
    U = np.sort(data['UserID'].unique())

    # Items
    I = np.sort(data['MovieID'].unique())

    # Ratings
    # To create R, we create a matrix with User as first entry and item as second entry
    R = -1 * np.ones((len(U), len(I)))

    for lines in range(len(data)):
        R[data['UserID'][lines], data['MovieID'][lines]] = data['Prediction'][lines]

    # Replace -1 with NAN
    R[R == -1] = np.inf

    # Items rated by users
    # To create I_u, we create a matrix with User as first entry and item as second entry
    I_u = []
    for line in range(R.shape[0]):
        I_u.append(I[~np.isinf(R[line])])

    return U, I, R, I_u


def abs_diff(x, y, i, R):
    """
        Compute Absolute difference between the ratings given by user x and user y to the item i

    :param      x: A user
    :param      y: A second user
    :param      i: An item (or an array of items)
    :param      R: Matrix of ratings

    :return:    d: Absolute difference between ratings
    """

    return np.abs(R[x, i] - R[y, i])


def get_all_abs_diff(U, u, I_u, R):
    """
        Compute all the absolute differences between an active user and all the others for all the items
        the active user has rated.

    :param      U: Array of users
    :param      u: An active user
    :param      I_u: List of arrays of items rated by each user
    :param      R: Matrix of ratings

    :return:    AD_u: Matrix of all the absolute differences for an active user.
                   (Rows: users, Columns: items rated by active user u)
    """

    AD_u = np.zeros((len(U), len(I_u[u])))

    for usr in U:
        if usr != u:
            AD_u[usr, :] = abs_diff(u, usr, I_u[u], R)

    return AD_u


def get_matrix_dominance(U, AD_u):
    """
        Compute the dominance matrix according to the absolute difference for an active user.
        An element (x,y) of the matrix is either True or False:
        - True if user x dominates user y w.r.t. the active user u
        - False otherwise

        We say that a user x dominates a user y w.r.t. to another user u if the two conditions are satisfied:

            1. For all items that have been rated by user u, the absolute difference between u and x is
               always smaller or equal to the absolute difference between u and y
            2. There is at least on item such that the absolute difference between u and x is strictly
               smaller than the absolute difference between u and y

    :param      U: Array of users
    :param      D: Matrix of all the absolute differences between an active user and all the other users

    :return:    M: Matrix of dominance between all the users w.r.t. to active user u
    """

    M_u = np.zeros((len(U), len(U)))
    for x in range(len(U)):
        M_u[x, :] = ((AD_u[x] <= AD_u).all(axis=1) * (AD_u[x] < AD_u).any(axis=1))

    return M_u


def sets_dominated_dominator(U, u, M_u):
    """
        Compute the sets of dominated user for an active user and the set of non-dominated users.

    :param      U: Array of users
    :param      u: An active user
    :param      M_u: Matrix of dominance between all the users w.r.t. to active user u

    :return:    C_u: Set of all the non-dominated users w.r.t. to active user u (candidates for K-neighbors)
                D_u: Set of all users that are dominated by at least one user in C_u
    """

    D_u = []
    C_u = []

    for idx, usr in enumerate(np.delete(U, u)):
        if (M_u[:u:, usr] == 0).all():
            C_u.append(usr)
        else:
            D_u.append(usr)

    return C_u, D_u


def create_Cu(U, u, I_u, R, folder):
    """
        Create the set of non-dominated user for an active user. It will save it as a pickle file

    :param      U: Array of users
    :param      u: An active user
    :param      I_u: List of arrays of items rated by each user
    :param      R: Matrix of Ratings
    :param      folder: Path to a folder to save the pickle file

    :return:    Nothing. Just a pickle file saved.
    """

    # Create Matrix of absolute differences
    AD_u = get_all_abs_diff(U, u, I_u, R)

    # Create Matrix of Dominance
    M_u = get_matrix_dominance(U, AD_u)

    # Create C_u and D_u
    C_u, D_u = sets_dominated_dominator(U, u, M_u)

    # Save the file
    filename = folder + "C_" + str(u) + ".pickle"
    pickle.dump(C_u, open(filename, "wb"))
    print("  C_%i created"%(u))

def preprocessing(folder, U, I_u, R, nbr_jobs=None, subset=None):
    """
        Creates all the files with the sets of dominated users. At the end, it will compile everything into
        a single txt file called "non-dominated-users.dat".

    :param      folder: Folder where to save the pickle files and the final "non-dominated-users.dat" file
    :param      U: Array of users
    :param      I_u: List of arrays of items rated by each user
    :param      R: Matrix of Ratings
    :param      nbr_jobs: Number of threads you will use for the parallelization. If not given, we will use
                          default value using the function ????
    :param      subset: Array of size 2x1 with starting index and ending index. If not given, we will create
                        the sets of non-dominated users for all the users

    :return:    Nothing. Just some files saved on the disk
    """

    if nbr_jobs == None:
        nbr_jobs = multiprocessing.cpu_count()
    if subset == None:
        subset = [0, len(U)]

    print("Start the preprocessing on %i threads for U[%i:%i]."%(nbr_jobs, subset[0], subset[1]))

    Parallel(n_jobs=nbr_jobs)(delayed(create_Cu)(U, u, I_u, R, folder) for u in U[subset[0]:subset[1]])

    print("Preprocessing finished. Preparing the final file.")

    C = []
    for usr in U[subset[0]:subset[1]]:
        filename = folder + "C_" + str(usr) + ".pickle"
        C.append(pickle.load(open(filename, "rb")))

    with open(folder + "non-dominated-users.dat", "w") as file:
        for item in C:
            file.write("{}\n".format(item))

    return folder + "non-dominated-users.dat"

def get_preprocessing(non_dom_users_file):

    print("Loading the preprocessing file")

    with open(non_dom_users_file, "r") as file:
        C = file.readlines()
        
    for i in range(len(C)):
        C[i] = C[i][1:-2]
        tmp = C[i].split(', ')
        C[i] = list(map(int, tmp))    

    return C





#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

from helpers_CF_and_PD import *

def main():

    # Dataset path
    dataset = '../../data/data_train.csv'

    data = prepare_data(dataset)

    # Prepare the data
    U, I, R, I_u = variables(data)

    # Folder for the pickle file
    folder = "./pickles/"

    """ Use these line if you start from scratch
    # Do the preprocessing
    non_dom_users_file = non_dominated_users(folder, U, I_u, R)

    """

    non_dom_users_file = folder + "non-dominated-users.dat"

    # Get the file
    C = get_pickle(non_dom_users_file)

    """ Use these line if you start from scratch
    # Get the Nearest neighbors for the 3 different methods
    NN_pearson_file = nearest_neighbors(folder, R, U, I_u, "pearson", subset=C)
    NN_cosine_file = nearest_neighbors(folder, R, U, I_u, "cosine", subset=C)
    NN_msd_file = nearest_neighbors(folder, R, U, I_u, "msd", subset=C)
    """

    NN_pearson_file = folder + "NN_pearson.pickle"
    NN_cosine_file = folder + "NN_copsine.pickle"
    NN_msd_file = folder + "NN_msd.pickle"



if __name__ == "__main__":
    main()
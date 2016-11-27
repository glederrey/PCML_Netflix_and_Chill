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

    """ Use these line if you start from scratch

    # Folder for the pickles files
    folder = "./pickles/"

    # Do the preprocessing
    non_dom_users_file = preprocessing(folder, U, I_u, R)

    """

    # Folder for the preprocessing file
    folder = "./"

    non_dom_users_file = folder + "non-dominated-users.dat"

    # Get the file
    C = get_preprocessing(non_dom_users_file)

    print(C[0])

if __name__ == "__main__":
    main()
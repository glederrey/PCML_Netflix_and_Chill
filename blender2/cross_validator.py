#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>, Gael Lederrey <gael.lederrey@epfl.ch>,
# Stefano Savare <stefano.savare@epfl.ch>
#
# Distributed under terms of the MIT license.

"""

"""

import numpy as np
import pandas as pd
import sys
from helpers import create_folder, evaluate
import os


def elements_in_folder(folder):
    return len([name for name in os.listdir(folder)])


class CrossValidator:
    """
    Class that provide a normalized version of the dataframe

    It provides all the method both to normalize the dataframe both to recover the right predictions
    from the predictions obtained from the normalized dataframe
    """

    def __init__(self):
        """
        It stores internally the dataframe and it computes the most relevant quantities about it.

        @ params
            - df, the PD dataframe to normalize
        """
        self.indices_dictionary = None
        self.predictions_dictionary = {}

    """"""""""""""""""""""""""""""""""""

    """ CROSS VALIDATION """

    """"""""""""""""""""""""""""""""""""


    def shuffle_indices(self, df, k):
        """ take a pandas.DataFrame and calculate k array of shuffled indices for cross-validation"""

        n = df.shape[0]
        cut = int(n / k)
        indices = list(range(n))

        np.random.seed(42)

        np.random.shuffle(indices)

        list_indices = []
        for i in range(k):
            list_indices.append(indices[cut * i: cut * (i + 1)])

        fold_dictionary = {}
        for i in range(k):
            train = np.array([x for j, x in enumerate(list_indices) if j != i]).flat
            train = list(train)
            train.sort()
            test = list_indices[i]
            test.sort()
            fold_dictionary[i] = {'train': train, 'test': test}

        self.indices_dictionary = fold_dictionary
        return fold_dictionary

    def shuffle_indices_and_store(self, df, k):
        """ create indices_dictionary AND store it in file """
        dic = self.shuffle_indices(df, k)
        self.store_indices()

        return dic

    def k_fold_predictions(self, df, model, model_name, **kwargs):
        """
        add a model to the predictions dictionary containing k pandas.DataFrame for each model

        Args:
            df (pandas.DataFrame): dataset, will be split according to indices_dictionary
            model (function): model to be used
            model_name (str): name given to the model in the dictionary
            **kwargs: arguments to be passed to the model function

        Returns:
            dict: key n_fold; value train/test dictionnary; sub_value: array of indices
                e.g. dict['name'][0]: prediction for first fold
        """
        if self.indices_dictionary is None:
            print("[ERROR] first define fold_indices dictionary")
            sys.exit()

        predictions_dict = {}
        for i in range(len(self.indices_dictionary)):
            train = df.loc[self.indices_dictionary[i]['train']].sort_index()
            test = df.loc[self.indices_dictionary[i]['train']].sort_index()

            predictions = model(train, test, **kwargs)
            predictions_dict[i] = predictions

        print(model_name)
        self.predictions_dictionary[model_name] = predictions_dict

        return predictions_dict

    def k_fold_predictions_and_store(self, df, model, model_name, **kwargs):
        """ produce k-fold predictions AND store the prediction in files """

        # check folder or create
        folder_name = './CV/' + model_name
        create_folder(folder_name)

        pred_dict = self.k_fold_predictions(df, model, model_name, **kwargs)

        for i in pred_dict.keys():
            file_name = folder_name + '/' + str(i) + '.csv'
            pred_dict[i].to_csv(file_name, index=False)

    def evaluate_model(self, model_name, test_df):
        """ cross validation """
        rmse_list = []
        for i in self.predictions_dictionary[model_name].keys():
            pred = self.predictions_dictionary[model_name][i]
            test = test_df.loc[self.indices_dictionary[i]['test']]

            rmse = evaluate(pred, test)
            rmse_list.append(rmse)
        return np.mean(rmse_list)

    def print_models(self):
        print(list(self.predictions_dictionary.keys()))

    """"""""""""""""""""""""""""""""""""

    """ STORE AND LOAD """

    """"""""""""""""""""""""""""""""""""

    def store_prediction(self):
        """ dump predictions_dictionary in file """
        for model_name in self.predictions_dictionary.keys():
            folder_name = './CV/' + model_name
            create_folder(folder_name)

            for j in self.predictions_dictionary[model_name].keys():
                file_name = folder_name + '/' + str(j) + '.csv'
                self.predictions_dictionary[model_name][j].to_csv(file_name, index=False)

    def store_indices(self):
        """ dump indices_dictionary in file """
        folder_name = './CV/' + 'indices'
        create_folder(folder_name)

        for i in self.indices_dictionary.keys():
            file_name = folder_name + '/' + str(i)
            with open(file_name + '_train.csv', 'w') as file:
                for item in self.indices_dictionary[i]['train']:
                    file.write("%s\n" % item)

            with open(file_name + '_test.csv', 'w') as file:
                for item in self.indices_dictionary[i]['test']:
                    file.write("%s\n" % item)

    def load_model(self, model_names):
        """ load models from list of name and add it (replace if already existing) to predictions_dictionary """
        list_files = os.listdir('./CV/')
        for model_name in model_names:
            if model_name not in list_files:
                print("[ERROR] " + model_name + " does not exist")
                sys.exit()

            n = elements_in_folder('./CV/' + model_name)
            pred_dict = {}
            for i in range(n):
                pred_dict[i] = pd.read_csv('./CV/' + model_name + "/" + str(i) + '.csv')

            self.predictions_dictionary[model_name] = pred_dict

        return self.predictions_dictionary

    def load_indices(self):
        """ load indices and replace indices_dictionary by it """
        # clear indices dictionary
        self.indices_dictionary = {}

        # simple version for working with CWD
        n = elements_in_folder('./CV/indices/')
        n = int(n / 2)

        for i in range(n):
            f = open('./CV/indices/' + str(i) + '_train.csv', 'r')
            lines = f.readlines()
            train = [int(i) for i in lines]

            f = open('./CV/indices/' + str(i) + '_test.csv', 'r')
            lines = f.readlines()
            test = [int(i) for i in lines]

            dic = {'train': train, 'test': test}
            self.indices_dictionary[i] = dic

        return self.indices_dictionary

    def clean_predictions_dictionary(self):
        """ clear all predictions """
        self.predictions_dictionary = {}

    """"""""""""""""""""""""""""""""""""

    """ BLENDING """

    """"""""""""""""""""""""""""""""""""

    def blend(self, weights):
        """ produce blended prediction with weights dictionary"""
        # initial predictions df
        random_name = list(weights.keys())[0]
        pred = {}

        for i in self.predictions_dictionary[random_name].keys():
            pred[i] = pd.DataFrame.copy(self.predictions_dictionary[random_name][i])
            pred[i]['Rating'] = 0.0

        for model_name in self.predictions_dictionary.keys():
            for i in self.predictions_dictionary[model_name].keys():
                pred[i]['Rating'] += \
                    weights[model_name] * self.predictions_dictionary[model_name][i]['Rating']

        return pred

    def evaluate_blending(self, weights, df):
        """ cross-validate blended prediction """
        blend_dict = self.blend(weights)

        list_rmse = []
        for i in blend_dict.keys():
            pred = blend_dict[i]
            test = df.loc[self.indices_dictionary[i]['test']]
            rmse = evaluate(pred, test)
            list_rmse.append(rmse)

        return np.mean(list_rmse)

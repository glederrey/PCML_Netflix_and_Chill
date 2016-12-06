import pandas as pd
import numpy as np
from operator import itemgetter
import pickle

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

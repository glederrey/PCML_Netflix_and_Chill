#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>
#
# Distributed under terms of the MIT license.


"""
Main helpers functions
HERE TO INSERT METHOD SUCH AS LOAD_DATA, WRITE_PREDICTION, ....
"""


import pandas as pd
import numpy as np
import os
import sys

from pyspark.mllib.recommendation import ALS
from pyspark import *
from pyspark.sql import *


def load_csv(filename='../data/data_train.csv'):
    df = pd.read_csv(filename)
    df['UserID'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
    df['MovieID'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
    df['Rating'] = df['Prediction']
    df = df.drop(['Id', 'Prediction'], axis=1)
    return df

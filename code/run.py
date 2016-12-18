#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

# import models
from models.means import *
from models.medians import *
from models.collaborative_filtering import *
from models.MF_SGD import *
from models.als import predictions_ALS

from helpers import load_csv, blender
from pyspark import SparkContext, SparkConf

import threading
import time


def main():
    print("============")
    print("[INFO] START")
    print("============")

    print("[INFO] Start Spark")
    # configure and start spark
    conf = (SparkConf()
            .setMaster("local")
            .setAppName("My app")
            .set("spark.executor.memory", "1g")
            )
    sc = SparkContext(conf=conf)

    # test if spark works
    if sc is not None:
        print("[INFO] Spark successfully initiated")
    else:
        print("[ERROR] Problem with spark, check your configuration")
        exit()

    # hide spark log information
    sc.setLogLevel("ERROR")

    print("[INFO] Load data set")
    train = load_csv('data/data_train.csv')
    test = load_csv('data/sampleSubmission.csv')

    print("========================================")
    print("[INFO] Start recommender system modeling")
    print("========================================")

    # dictionary containing predictions
    models = {}

    print("[INFO] Modeling: Global mean")
    models['global_mean'] = global_mean(train, test)

    print("[INFO] Modeling: Global median")
    models['global_median'] = global_median(train, test)

    print("[INFO] Modeling: User mean")
    models['user_mean'] = user_mean(train, test)

    print("[INFO] Modeling: User median")
    models['user_median'] = user_median(train, test)

    print("[INFO] Modeling: Movie mean")
    models['movie_mean'] = movie_mean(train, test)

    print("[INFO] Modeling: Movie median")
    models['movie_median'] = movie_median(train, test)

    print("[INFO] Modeling: Movie mean (deviation normalized)")
    models['movie_mean_deviation_user'] = movie_mean_deviation_user(train, test)

    print("[INFO] Modeling: Movie median (deviation normalized)")
    models['movie_median_deviation_user'] = movie_median_deviation_user(train, test)

    print("[INFO] Modeling: Collaborative filtering")
    models['collab_filt'] = collaborative_filtering(train, test, movie_features=20, alpha=19)

    print("[INFO] Modeling: Matrix Factorization using SGD")
    models['mf_sgd'] = matrix_factorization_SGD(train, test, gamma=0.004,
                                                n_features=20, n_iter=20, init_method='global_mean')

    print("[INFO] Modeling: ALS")
    models['als'] = predictions_ALS(train, test, spark_context=sc, rank=8,
                                    lambda_=0.081, iterations=24)

    weights = {'user_mean': -3.6773325300577424,
               'mf_sgd_rescale': -0.067287844047187018,
               'mf_sgd': -0.038620299655287627,
               'als': 0.89273834619373038,
               'movie_mean_deviation_user': -0.43266880016189613,
               'movie_median': -4.2973300370745839,
               'movie_mean': 0.58294734803419068,
               'movie_median_deviation_user': 4.2987360771658558,
               'user_median': -0.0049855566007924942,
               'collab_filt': 0.071096854713773971,
               'global_median': 4.1657975331095347,
               'global_mean': -0.6450305175665445
               }

    print("[INFO] Blending")
    blend = blender(models, weights)

    print("[INFO] Prepare submission")
    submission = submission_table(blend, 'User', 'Movie', 'Rating')
    file_name = 'prediction.csv'
    submission.to_csv(file_name, index=False)

    print("[INFO] Predictions written to file: ", file_name)

    print("=============")
    print("[INFO] FINISH")
    print("=============")


if __name__ == '__main__':
    main()

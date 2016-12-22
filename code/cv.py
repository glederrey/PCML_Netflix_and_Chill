#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>, Gael Lederrey <gael.lederrey@epfl.ch>,
# Stefano Savare <stefano.savare@epfl.ch>
#
# Distributed under terms of the MIT license.

"""
cv.py

run command "spark-submit run.py" to launch it
Cross-validate the models and test the blending
WARNING! Takes a LOT of time!
"""

# import models
from models.medians import *
from models.means import *
from models.MF_RR import *
from models.MF_SGD import *
from models.als import *
from models.surprise_models import *
from models.pyfm import *

from helpers import *
from cross_validator import *
from pyspark import SparkContext, SparkConf
import scipy.optimize as sco
import time


def main():
    start = time.time()
    print("============")
    print("[INFO] START")
    print("============")

    # configure and start spark
    conf = (SparkConf()
            .setMaster("local")
            .setAppName("My app")
            .set("spark.executor.memory", "1g")
            )
    sc = SparkContext(conf=conf)

    print("============================================")
    print("[INFO] Start Blending for Recommender System")
    print("============================================")

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

    print("\n")

    # Get the models
    models = get_models()

    print("[INFO] Prepare the 5-folds")

    # Start the CrossValidator
    cv = CrossValidator()

    # Create the 5-folds
    cv.shuffle_indices_and_store(train, 5)

    # Predict and store the models
    for idx, mdl in enumerate(models.keys()):
        print("[INFO] Predicting and Storing Model %2d/30: %s" % (idx + 1, mdl))
        tt = time.time()
        cv.k_fold_predictions_and_store(train, models[mdl]['function'], mdl, override=False, **models[mdl]['params'])
        print("[INFO] Completed in %s\n" % (time_str(time.time() - tt)))

    # Delete the CrossValidator class to free the memory
    del cv

    print("[INFO] Reload class CrossValidator")
    # Reload the CrossValidator class
    cv = CrossValidator()

    # Load the indices
    cv.load_indices()

    # Load the ground truth
    cv.define_ground_truth(train)

    model_names = list(models.keys())

    # Load model predictions
    cv.load_predictions(model_names)

    # Define first vector for Blending
    x0 = 1 / len(model_names) * np.ones(len(model_names))

    # Do the Optimization for the blending
    print("[INFO] Optimize for the blending")
    res = sco.minimize(eval_, x0, method='SLSQP', args=(cv, models), options={'maxiter': 1000, 'disp': True})

    # Create best dictionnary
    best_dict = {}
    for idx, key in enumerate(models.keys()):
        best_dict[key] = res.x[idx]

    # Test the blending
    test_blending(cv, best_dict)

    print("[INFO] Total time: %s" % (time_str(time.time() - start)))

    print("==============")
    print("[INFO] FINISH!")
    print("==============")


def eval_(x, cv, models):
    """ Evaluate the RMSE of the blending """
    dict_try = {}
    for idx, key in enumerate(models.keys()):
        dict_try[key] = x[idx]

    return cv.evaluate_blending(dict_try)


def test_blending(cv, best_dict):
    """ Evaluate the RMSE of each model and a specific blending """
    cv.evaluation_all_models()

    print()
    rmse = cv.evaluate_blending(best_dict)
    print("Best blending: %s" % best_dict)
    print("RMSE best blending: %.5f" % rmse)


def get_models():
    models = {
        'global_mean': {
            'function': global_mean,
            'params': {}
        },
        'global_median': {
            'function': global_median,
            'params': {}
        },
        'user_mean': {
            'function': user_mean,
            'params': {}
        },
        'user_median': {
            'function': user_median,
            'params': {}
        },
        'movie_mean': {
            'function': movie_mean,
            'params': {}
        },
        'movie_mean_rescaled': {
            'function': movie_mean_rescaled,
            'params': {}
        },
        'movie_median': {
            'function': movie_median,
            'params': {}
        },
        'movie_median_rescaled': {
            'function': movie_median_rescaled,
            'params': {}
        },
        'movie_mean_deviation_user': {
            'function': movie_mean_deviation_user,
            'params': {}
        },
        'movie_mean_deviation_user_rescaled': {
            'function': movie_mean_deviation_user_rescaled,
            'params': {}
        },
        'movie_median_deviation_user': {
            'function': movie_median_deviation_user,
            'params': {}
        },
        'movie_median_deviation_user_rescaled': {
            'function': movie_median_deviation_user_rescaled,
            'params': {}
        },
        'als': {
            'function': predictions_ALS,
            'params': {
                'spark_context': sc,
                'rank': 8,
                'lambda_': 0.081,
                'iterations': 24,
                'nonnegative': True
            }
        },
        'als_rescaled': {
            'function': predictions_ALS_rescaled,
            'params': {
                'spark_context': sc,
                'rank': 8,
                'lambda_': 0.081,
                'iterations': 24,
                'nonnegative': True
            }
        },
        'mf_rr': {
            'function': mf_RR,
            'params': {
                'movie_features': 20,
                'alpha': 19
            }
        },
        'mf_rr_rescaled': {
            'function': mf_RR_rescaled,
            'params': {
                'movie_features': 20,
                'alpha': 19
            }
        },
        'mf_sgd': {
            'function': mf_SGD,
            'params': {
                'gamma': 0.004,
                'n_features': 20,
                'n_iter': 20,
                'init_method': 'global_mean'
            }
        },
        'mf_sgd_rescaled': {
            'function': mf_SGD_rescaled,
            'params': {
                'gamma': 0.004,
                'n_features': 20,
                'n_iter': 20,
                'init_method': 'global_mean'
            }
        },
        'pyfm': {
            'function': pyfm,
            'params': {
                'num_factors': 20,
                'num_iter': 200,
                'init_lr': 0.001,
            }
        },
        'pyfm_rescaled': {
            'function': pyfm_rescaled,
            'params': {
                'num_factors': 20,
                'num_iter': 200,
                'init_lr': 0.001,
            }
        },
        'knn_ib': {
            'function': knn,
            'params': {
                'k': 60,
                'sim_options': {
                    'name': 'pearson_baseline',
                    'user_based': False
                }
            }
        },
        'knn_ib_rescaled': {
            'function': knn_rescaled,
            'params': {
                'k': 60,
                'sim_options': {
                    'name': 'pearson_baseline',
                    'user_based': False
                }
            }
        },
        'knn_ub': {
            'function': knn,
            'params': {
                'k': 300,
                'sim_options': {
                    'name': 'pearson_baseline',
                    'user_based': True
                }
            }
        },
        'knn_ub_rescaled': {
            'function': knn_rescaled,
            'params': {
                'k': 300,
                'sim_options': {
                    'name': 'pearson_baseline',
                    'user_based': True
                }
            }
        },
        'svd': {
            'function': svd,
            'params': {
                'n_epochs': 30,
                'lr_all': 0.001,
                'reg_all': 0.001
            }
        },
        'svd_rescaled': {
            'function': svd_rescaled,
            'params': {
                'n_epochs': 30,
                'lr_all': 0.001,
                'reg_all': 0.001
            }
        },
        'slope_one': {
            'function': slope_one,
            'params': {}
        },
        'slope_one_rescaled': {
            'function': slope_one_rescaled,
            'params': {}
        },
        'baseline': {
            'function': baseline,
            'params': {}
        },
        'baseline_rescaled': {
            'function': baseline_rescaled,
            'params': {}
        }
    }

    return models


if __name__ == '__main__':
    main()

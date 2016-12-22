#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Joachim Muth <joachim.henri.muth@gmail.com>, Gael Lederrey <gael.lederrey@epfl.ch>,
# Stefano Savare <stefano.savare@epfl.ch>
#
# Distributed under terms of the MIT license.

"""
run.py

run command "spark-submit run.py" to launch it
reproduce Kaggle's challenge predictions
"""

import time

# import models
from models.medians import *
from models.means import *
from models.MF_RR import *
from models.MF_SGD import *
from models.als import *
from models.surprise_models import *
from models.pyfm import *
import pickle

from helpers import *
from pyspark import SparkContext, SparkConf


def main():
    start = time.time()
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

    print("========================================")
    print("[INFO] Start recommender system modeling")
    print("========================================")

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
    
    print("\n")

    # dictionary containing predictions
    models = {}

    print("[INFO] Preparing Model 01/30: Global mean")
    tt = time.time()
    models['global_mean'] = global_mean(train, test)
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))    

    print("[INFO] Preparing Model 02/30: Global median")
    tt = time.time()    
    models['global_median'] = global_median(train, test)
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))

    print("[INFO] Preparing Model 03/30: User mean")
    tt = time.time()    
    models['user_mean'] = user_mean(train, test)
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))

    print("[INFO] Preparing Model 04/30: User median")
    tt = time.time()    
    models['user_median'] = user_median(train, test)
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))

    print("[INFO] Preparing Model 05/30: Movie mean")
    tt = time.time()    
    models['movie_mean'] = movie_mean(train, test)
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))
    
    print("[INFO] Preparing Model 06/30: Movie mean rescaled with User mood")
    tt = time.time()    
    models['movie_mean_rescaled'] = movie_mean_rescaled(train, test)
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))

    print("[INFO] Preparing Model 07/30: Movie median")
    tt = time.time()    
    models['movie_median'] = movie_median(train, test)
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))
    
    print("[INFO] Preparing Model 08/30: Movie median rescaled with User mood")
    tt = time.time()    
    models['movie_median_rescaled'] = movie_median_rescaled(train, test)
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))

    print("[INFO] Preparing Model 09/30: Movie mean (deviation normalized)")
    tt = time.time()    
    models['movie_mean_deviation_user'] = movie_mean_deviation_user(train, test)
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))
    
    print("[INFO] Preparing Model 10/30: Movie mean (deviation normalized) rescaled with User mood")
    tt = time.time()    
    models['movie_mean_deviation_user_rescaled'] = movie_mean_deviation_user_rescaled(train, test)
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))

    print("[INFO] Preparing Model 11/30: Movie median (deviation normalized)")
    tt = time.time()    
    models['movie_median_deviation_user'] = movie_median_deviation_user(train, test)
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))
    
    print("[INFO] Preparing Model 12/30: Movie median (deviation normalized) rescaled with User mood")
    tt = time.time()    
    models['movie_median_deviation_user_rescaled'] = movie_median_deviation_user_rescaled(train, test)
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))

    print("[INFO] Preparing Model 13/30: Matrix Factorization using RR")
    tt = time.time()    
    models['mf_rr'] = mf_RR(train, test, movie_features=20, alpha=19)
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))
    
    print("[INFO] Preparing Model 14/30: Matrix Factorization using RR rescaled with User mood")
    tt = time.time()    
    models['mf_rr_rescaled'] = mf_RR_rescaled(train, test, movie_features=20, alpha=19)
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))

    print("[INFO] Preparing Model 15/30: Matrix Factorization using SGD")
    tt = time.time()    
    models['mf_sgd'] = mf_SGD(train, test, gamma=0.004, n_features=20, n_iter=20, init_method='global_mean')
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))
                                                
    print("[INFO] Preparing Model 16/30: Matrix Factorization using SGD rescaled with User mood")
    tt = time.time()    
    models['mf_sgd_rescaled'] = mf_SGD_rescaled(train, test, gamma=0.004, n_features=20, n_iter=20,
                                                init_method='global_mean')
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))

    print("[INFO] Preparing Model 17/30: ALS")
    tt = time.time()    
    models['als'] = predictions_ALS(train, test, spark_context=sc, rank=8, lambda_=0.081, iterations=24)
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))
                                    
    print("[INFO] Preparing Model 18/30: ALS rescaled with User mood")
    tt = time.time()  
    models['als_rescaled'] = predictions_ALS_rescaled(train, test, spark_context=sc, rank=8,
                                                      lambda_=0.081, iterations=24)
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))
                                                      
    print("[INFO] Preparing Model 19/30: Factorization Machine with PyFM")
    tt = time.time()  
    models['pyfm'] = pyfm(train, test, num_factors=20, num_iter=200, init_lr=0.001) 
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))
 
    print("[INFO] Preparing Model 20/30: Factorization Machine with PyFM rescaled with User mood")
    tt = time.time()  
    models['pyfm_rescaled'] = pyfm_rescaled(train, test, num_factors=20, num_iter=200, init_lr=0.001) 
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))
                                            
    print("[INFO] Preparing Model 21/30: Baseline Estimate") 
    tt = time.time()  
    models['baseline'] = baseline(train, test)
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))
    
    print("[INFO] Preparing Model 22/30: Baseline Estimate rescaled with User mood") 
    tt = time.time()  
    models['baseline_rescaled'] = baseline_rescaled(train, test) 
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))
        
    print("[INFO] Preparing Model 23/30: Slope One") 
    tt = time.time()  
    models['slope_one'] = slope_one(train, test)  
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))
    
    print("[INFO] Preparing Model 24/30: Slope One rescaled with User mood") 
    tt = time.time()  
    models['slope_one_rescaled'] = slope_one_rescaled(train, test)    
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))
    
    print("[INFO] Preparing Model 25/30: SVD") 
    tt = time.time()  
    models['svd'] = svd(train, test, n_epochs=30, lr_all=0.001, reg_all=0.001)
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))
    
    print("[INFO] Preparing Model 26/30: SVD rescaled with User mood")
    tt = time.time()   
    models['svd_rescaled'] = svd_rescaled(train, test, n_epochs=30, lr_all=0.001, reg_all=0.001)
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))
                                          
    print("[INFO] Preparing Model 27/30: kNN item-based") 
    tt = time.time()  
    models['knn_ib'] = knn(train, test, k=60, sim_options={'name': 'pearson_baseline','user_based': False})  
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))
                           
    print("[INFO] Preparing Model 28/30: kNN item-based rescaled with User mood") 
    tt = time.time()  
    models['knn_ib_rescaled'] = knn_rescaled(train, test, k=60, sim_options={'name': 'pearson_baseline',
                                             'user_based': False}) 
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))
                                             
    print("[INFO] Preparing Model 29/30: kNN user-based") 
    tt = time.time()  
    models['knn_ub'] = knn(train, test, k=300, sim_options={'name': 'pearson_baseline','user_based': True})
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))
                           
    print("[INFO] Preparing Model 30/30: kNN user-based rescaled with User mood") 
    tt = time.time()  
    models['knn_ub_rescaled'] = knn_rescaled(train, test, k=300, sim_options={'name': 'pearson_baseline',
                                             'user_based': True})     
    print("[INFO] Completed in %s\n"%(time_str(time.time() - tt)))

    weights = {
        'global_mean': 1.7756776068889906,
        'global_median': 1.8469393889491512, 
        'user_mean': -3.6424669808916055,
        'user_median': 0.0051375146192670111, 
        'movie_mean': -0.83307991660204828, 
        'movie_mean_rescaled': -0.95695560022481185, 
        'movie_median': -0.93869701618369406, 
        'movie_median_rescaled': -0.91347225736204185, 
        'movie_mean_deviation_user': 1.0442870681129861, 
        'movie_mean_deviation_user_rescaled': 0.92108939957699987, 
        'movie_median_deviation_user': 0.93849170091288214, 
        'movie_median_deviation_user_rescaled': 0.96461941548011165, 
        'mf_rr': 0.032225151029461573, 
        'mf_rr_rescaled': 0.035378890871598068, 
        'mf_sgd': -0.78708629851314926,
        'mf_sgd_rescaled': 0.27624842029358976, 
        'als': 0.30659162734621315, 
        'als_rescaled': 0.31745406600610854,
        'pyfm': 0.15296423817447555, 
        'pyfm_rescaled': -0.021626658245201873,  
        'baseline': -0.70720719475460081,   
        'baseline_rescaled': -0.56908887025195931,
        'slope_one': -0.023119356625828508,   
        'slope_one_rescaled': 0.43863736787065016,   
        'svd': 0.67558779271650848,  
        'svd_rescaled': -0.0049814548552847716,                                                  
        'knn_ib': -0.095005112653966148, 
        'knn_ib_rescaled': 0.34178799145510136, 
        'knn_ub': 0.21758562399412981, 
        'knn_ub_rescaled': 0.12803210410741006
    }

    print("[INFO] Blending")
    blend = blender(models, weights)

    print("[INFO] Prepare submission")
    submission = submission_table(blend, 'User', 'Movie', 'Rating')
    file_name = 'prediction2.csv'
    submission.to_csv(file_name, index=False)

    print("[INFO] Predictions written to file: ", file_name)
    
    print("[INFO] Total time: %s"%(time_str(time.time()-start)))

    print("==============")
    print("[INFO] FINISH!")
    print("==============")
    
if __name__ == '__main__':
    main()

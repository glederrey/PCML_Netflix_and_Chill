# Blender 

Corpus containing all methods

## Convention

### Method signature

* arguments: 
    * train_set: pandas.DataFrame 
    * test_set: pandas.DataFrame
    * ... (all others arguments must have a default value)

* return:
   * predicted_test_set: pandas.DataFrame
   
### pandas.DataFrame

`['User', 'Movie', 'Rating']` sorted by `Movie` then `User`


## List of methods

* MF_SGD.matrix_factorization_SGD
* means.user_mean
* means.movie_mean
* means.global_mean
* als.predictions_ALS

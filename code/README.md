# EPFL ML Recommender System

## Team members

Gael Lederrey, SCIPER 204874, gael.lederrey@epfl.ch

Stefano Savarè, SCIPER 260960, stefano.savare@epfl.ch

Joachim Muth, SCIPER 214757, joachim.muth@epfl.ch

## Required Libraries

* NumPy
* SciPy
* Pandas
* PySpark
* scikit-learn
* PyFM

All these libraries are available through pip3 installer, except PyFM coming from 
[this GitHub repo](https://github.com/coreylynch/pyFM) and installable with the following
command:
```
pip install git+https://github.com/coreylynch/pyFM
```

## Run the project

The predictions can be reproduce by running the provided run.py file.

```
spark-submit run.py
```

The predictions will be written in a file `predictions.py`

## File organisation

- *data* : contains train and test sets as well as a calculated user deviation file
    - *data_train.csv* : train set
    - *sampleSubmissionn.csv* : test set provided by Kaggle
    - *deviation_per_user.csv* : pre-computed user deviation dictionary
- *models* : contains all models (one by file) used in the project
    - *als.py*
    - *collaborative_filtering.py*
    - *means.py*
    - *median.py*
    - *MF_SGD.py*
- *cross_validation* : classes used to test models and select their parameters
    - *cross_validation.py* 
    - *cross_validation_blender.py*
- *rescaler.py* : is a normalizer class called by the models
- *run.py* : run whole project and reproduce Kaggle's challenge predictions 


## Possible issues

### Spark conflict with Jupyter iPython Notebok
```
jupyter: '.../run.py' is not a Jupyter command
```

Such an error means that your Spark configuration was previously set to be run on Jupyter iPython Notebook.
Unset your local variable `PYSPARK_DRIVER_PYTHON` in order to fix it:

```
unset PYSPARK_DRIVER_PYTHON
```

### Spark conflict with cache
```
Caused by: org.apache.spark.SparkException: File .../run.py exists and does not match contents of file: .../run.py
```

Spark encounters some error while trying to reuse file previously stored in its own cache.
In order to avoid this clear the folders `./__pycache__` and `./models/__pycache__`if you want to 
run the project twice. (Or run the provided shell script `clean_pycache.sh`)
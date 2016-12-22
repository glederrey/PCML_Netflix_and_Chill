# EPFL ML Recommender System

## Team members

Gael Lederrey, SCIPER 204874, gael.lederrey@epfl.ch

Stefano Savarè, SCIPER 260960, stefano.savare@epfl.ch

Joachim Muth, SCIPER 214757, joachim.muth@epfl.ch

## Required Libraries

Default libraries
* os
* collections
* cython

Libraries installed with Pip
* NumPy
* SciPy
* Pandas
* PySpark
* scikit-learn

Libraries with their own specific installation
* PyFM
  - It comes from [this GitHub repo](https://github.com/coreylynch/pyFM) and installable with the following
command:
```
pip install git+https://github.com/coreylynch/pyFM
```
* SurPRISE
  - It comes from [this GitHub repo](https://github.com/NicolasHug/Surprise). Please install it using the following commands:
```
git clone https://github.com/NicolasHug/surprise.git
python setup.py install
```

## Run the project

The predictions can be reproduce by running the provided run.py file.

```
spark-submit run.py
```

The predictions will be written in a file `predictions.py`. On a computer with an Intel i7-6820HK, it took around 2 hours to produce the predictions.

## File organisation

- *data* : contains train and test sets as well as a calculated user deviation file
    - *data_train.csv* : train set
    - *sampleSubmissionn.csv* : test set provided by Kaggle
    - *deviation_per_user.csv* : pre-computed user deviation dictionary
    - *pyfm_pred.csv*: Predictions for the PyFM model. (To make the predictions faster)
    - *pyfm_rescaled_pred.csv*: Prediction for the PyFM rescaled model. (To make the predictions faster)
- *models* : contains all models (one by file) used in the project
    - *als.py*
    - *means.py*
    - *median.py*
    - *MF_SGD.py*
    - *MF_RR.py*
    - *pyfm.py*
    - *surprise_models.py*
    - *helpers_scipy.py*
- *cross_validator.py*: is the class used to test models and blend them together
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

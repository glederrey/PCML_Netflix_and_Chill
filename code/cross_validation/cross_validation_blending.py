import pandas as pd
import numpy as np
from cross_validation.cross_validation import KFoldIndexes
import itertools
import os

class CrossValidationBlending:
    ''' Class to cross validate a blended model. The different submodels should have a predefined set of parameters.
    No cross validation is done on the submodel parameters.
    
    Usage (blend of 2 models ALS and movie mean):
    blending_test=CrossValidationBlending(full_dataset,k_splits)
    blending_test.add_model(predictions_ALS,'als')
    blending_test.add_model(movie_mean,'movie_mean')
    blending_test.add_params_for_model('als',{'spark_context':sc,'rank':4})
    blending_test.add_params_for_model('movie_mean',{})
    blending_test.evaluate_blending({'movie_mean':0.5,'als':0.5})
    '''
    
    def __init__(self,data,k):
        ''' Initialization function. It creates self.tests_list, the list of all the test dataframe
        
        @ params
            - data, the input dataframe
            - k, the number of splits in the cross validation
        '''
        
        # Initialize static class variables
        self.models={} # Dict to store all the model functions
        self.params={} # Dict to store the params with which running each model
        self.predictions={} # Dict to store the predictions for each model with the given parameters
        self.real_values=[] # List to store the real values for each chunk
        self.blended_predictions=[]
        self.validation_predictions={}
        
        # Initialize the parameters
        self.k=k
        
        # Initialize the k_fold_indexes
        k_fold_indexes=KFoldIndexes(k,data.shape[0])
        
        if k>1:
            self.tests_list=self.get_tests_database(data,k_fold_indexes)
        else:
            pass
        
    def add_model(self,function, name):
        ''' Function to add a model to the evaluation
        
        Requirements for 'function'
        - First 2 parameters train and test database
        - Other optional parameters to be specified using the function add_params_for_model
        - It returns the prediction dataset, with columns named User, Movie, Rating, sorted by Movie and then User.
        
        @ params
            - function, function that, given train, test and possibly other parameters returns the predictions computed on the test dataset.
            - name, string representing the model name
        '''
        self.models[name]=function

    def add_params_for_model(self,model_name,params_dict,compute_predictions=True):
        ''' Function to add optional parameters to a model
        
        The function given as input in add_model() may require other parameters a part of train and test.
        
        WARNING - IT MAY BE SLOW - It computes the prediction for the model with these parameters
        
        @ params
            - model_name, string - the same passed in add_model
            - params_dict, dictionary containing the optional parameters to be passed to the model function
            - compute_predictions, if compute automatically the predictions for this set of params
        '''
        if model_name not in self.models:
            print('Warning: Adding parameters for a non-existing model')
        self.params[model_name]=params_dict
        if compute_predictions:
            self.compute_predictions(model_name)
        
    def compute_predictions(self,model):
        ''' Function that computes the predictions for a given model
        
        DO NOT CALL IT DIRECTLY - It is automatically called when adding the model parameters.
        
        The predictions are stored in predictions[model]. 
        It is a list with k_splits elements, each one is a Pandas dataframe containing the prediction for the corresponding test dataframe. 
        
        @ params
            - model, string that represents the model for which computing the predictions
        
        '''
        for model_name in self.models:
            if model_name!=model:
                continue
            self.real_values=[]
            
            function=self.models[model_name]
            try:
                arguments=self.params[model_name]
            except:
                print('Arguments not available for model',model_name)
                
            self.predictions[model_name]=[]
            for comb in itertools.combinations(range(self.k),self.k-1):
                trains=[self.tests_list[x] for x in comb]
                train=pd.concat(trains)                
                
                test_index=[x for x in range(self.k) if x not in comb][0]
                test=self.tests_list[test_index]
                
                self.real_values.append(test.Rating)
                self.predictions[model_name].append(function(train,test,**arguments))
                os.system('rm metastore_db/*.lck')
            
    def evaluate_blending(self,blending_dict):        
        ''' Evaluate a particular blended model
        
        @ params
            - blending_dict, dictionary in the form {'movie_mean':0,'user_mean':0,'global_mean':1}
        '''
        for model_name in blending_dict:
            if model_name not in self.predictions:
                print('Predictions not available for model',model_name)
                raise()
        
        self.blended_predictions=[]
        for i in range(self.k):            
            cont=0
            for model_name in blending_dict:
                if cont==0:
                    prediction=np.array(blending_dict[model_name]*self.predictions[model_name][i].Rating)
                    cont+=1
                else:
                    prediction+=np.array(blending_dict[model_name]*self.predictions[model_name][i].Rating)
            for i in range(len(prediction)):
                if prediction[i] > 5:
                    prediction[i] = 5
                elif prediction[i] < 1:
                    prediction[i] = 1
                
            self.blended_predictions.append(prediction)
        
        predictions_conc=np.concatenate(self.blended_predictions)
        real_values_conc=np.concatenate(self.real_values)
        rmse=np.sqrt(sum((predictions_conc-real_values_conc)**2)/predictions_conc.shape[0])
        return rmse
    
    def evaluate_blending_for_validation(self,blending_dict,train,validation):
        ''' Function to compute predictions for a particular blending with a particular train set
        
        @ params
            - blending_dict, dictionary in the form {'movie_mean':0,'user_mean':0,'global_mean':1}
            - train, the training df
            - validation, the df on which computing the predictions
        '''
        for model_name in blending_dict:
            if model_name not in self.params:
                print('Params not available for model',model_name)
                raise()
       
        cont=0
        for model_name in blending_dict:
            if blending_dict[model_name] != 0:
                function=self.models[model_name]
                arguments=self.params[model_name]
                predictions_model=function(train,validation,**arguments)
                if cont==0:
                    predictions=np.array(blending_dict[model_name]*predictions_model.Rating)
                    cont+=1
                else:
                    predictions+=np.array(blending_dict[model_name]*predictions_model.Rating)
                os.system('rm metastore_db/*.lck')
        for i in range(len(predictions)):
            if predictions[i] > 5:
                predictions[i] = 5
            elif predictions[i] < 1:
                predictions[i] = 1
        return predictions
    
        
    def delete_model(self,model_name):
        ''' Function to permanently delete a model and all its data'''
        try:
            del self.models[model_name]
        except:
            print('Model not saved')
        try:
            del self.params[model_name]
        except:
            print('Model params not presents')
        try:
            del self.predictions[model_name]
        except:
            print('Model predictions not presents')
        
    def get_tests_database(self,data,k_fold_indexes):
        '''Internal function to get the list of test pandas dataframe'''
        tests=[]
        for i in k_fold_indexes.indexes:
            tests.append(data.loc[i[1]])
        return tests

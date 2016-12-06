import random
from pyspark.sql import SQLContext
import itertools

class KFoldIndexes:
    ''' Class to get indexes for cross validation
    
    Usage: 
    indexes=KFoldIndexes(4,10)
    indexes.indexes  # to access the indexes of the cross validation, it is a list of tuples (train,test)
    
    @ variables
        - indexes, a list of tuples (train,test), access to this variable to get the splits
          train-test. train and test are 2 list of the indexes of the elements of the train and of the test.
    '''
    
    def __init__(self,n_splits,rows):
        ''' Method to initialize the class
        @ params
            - n_splits, the number of folds in the cross validation
            - rows, the rows in the database to split
        '''
        test_elements=int(rows/n_splits) # How many elements in the tests partitions
        remaining_elements=rows%n_splits # If rows is not divisible by n_splits, some elements have to be reallocated
        elements_remaining_list=list(range(rows)) # List of all the elements that are not already been present in a test dataset
        
        self.indexes=[]
        
        # Compute all the (train,test) tuples
        # The lists train and test are correct but not sorted
        for i in range(n_splits):
            random.shuffle(elements_remaining_list)
            test=elements_remaining_list[:test_elements+int(remaining_elements>i)]
            train=elements_remaining_list[test_elements+int(remaining_elements>i):]
            elements_remaining_list=train.copy()

            for j in range(i):
                train+=self.indexes[j][1]            
            
            self.indexes.append((train,test))
            
        # Sort all the lists train and tests
        for i in self.indexes:
            i[0].sort()
            i[1].sort()
        
class CrossValidation:
    ''' Class to run the cross validation with different models
    
    Usage:
    cross_val=CrossValidation(data,k,use_spark)
    cross_val.evaluate(model)
    
    The model passed as input parameter should have the following interface:
    def fit(train,**arg) -> void function
    def predict(test)    -> void function
    def evaluate(test)   -> return the rmse with the parameter passed in fit()
    
    The split is ALWAYS done by rows
    
    Variables:
    self.k
    self.use_spark   -> if True the train and test database are rdd dataframe. If False they are pd dataframe
    self.tests_list  -> list of all the test dataframe used. When a test database is used, the train is computed as the union of the union of the other tests in this list. 
    self.sc          -> spark context
    '''
    
    
    def __init__(self,data,k,use_spark,spark_context):
        ''' Initialization function. It creates self.tests_list, the list of all the test dataframe
        
        @ params
            - data, the input dataframe
            - k, the number of splits in the cross validation
            - use_spark, if using rdd dataframe or pd dataframe
            - spark_context, typically sc
        '''
        
        # Initialize the parameters
        self.k=k
        self.use_spark=use_spark
        self.sc=spark_context
        
        # Initialize the k_fold_indexes
        k_fold_indexes=KFoldIndexes(k,data.shape[0])
        
        if use_spark:
            self.tests_list=self.get_sql_from_pd(self.get_tests_database(data,k_fold_indexes))
        else:
            self.tests_list=self.get_tests_database(data,k_fold_indexes)

    def evaluate(self,model,**arg):
        ''' Function that evaluates a model passed as input with some arguments and returns a list
of rmse errors
        
        @ params
            - model. See class description for the requirements of this model
            - a set of params **arg that will be passed to the fit function of the model.
        
        @ returns
            - a list of rmse error. One for each split in the cross validation
        '''
        error_list=[]
        for comb in itertools.combinations(range(self.k),self.k-1):
            trains=[self.tests_list[x] for x in comb]
            if self.use_spark:  
                train=self.sc.union(trains)
            else:
                train=pd.concat(trains)
            test_index=[x for x in range(self.k) if x not in comb][0]
            test=self.tests_list[test_index]
            
            model.fit(train,**arg)
            model.predict(test)
            error=model.evaluate(test)
            error_list.append(error)
        return error_list
    
    def get_tests_database(self,data,k_fold_indexes):
        '''Internal function to get the list of test pandas dataframe'''
        tests=[]
        for i in k_fold_indexes.indexes:
            tests.append(data.loc[i[1]])
        return tests
    
    def get_sql_from_pd(self,df_list):
        '''Internal function to convert the list of pandas dataframe to a list of rdd dataframe'''
        sqlContext=SQLContext(self.sc)
        sql_list=[]
        for i in df_list:
            sql_list.append(sqlContext.createDataFrame(i).rdd)
        return sql_list
    

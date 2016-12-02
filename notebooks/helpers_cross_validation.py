import random

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
        

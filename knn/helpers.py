import pandas as pd
import numpy as np
from operator import itemgetter
import pickle

def prepare_data(dataset):
    """
        This function reads the csv file for the dataset and returns the data. We also add two columns to the
        pandas DataFrame.
        
    :param      dataset: CSV file containing the data set
    :return:    data: pandas DataFrame containing the original columns of the CSV file plus the UserID
                      and MovieID
    """
    print("Load the dataset")

    data = pd.read_csv(dataset)
    data['UserID'] = data['Id'].apply(lambda x: int(x.split('_')[0][1:]) - 1)
    data['MovieID'] = data['Id'].apply(lambda x: int(x.split('_')[1][1:]) - 1)

    return data
    
def variables(data):
    """
        Using the data extracted in the function prepare_data, we extract more specific data.

    :param      data: pandas DataFrame extracted using the function prepare_data

    :return:    U: Array of users
                I: Array of items
                R: Ratings of items by user
                U_i: List of array of users that have rated item i

    """

    print("Prepare the variables")

    # Users
    U = np.sort(data['UserID'].unique())

    # Items
    I = np.sort(data['MovieID'].unique())

    # Ratings
    # To create R, we create a matrix with User as first entry and item as second entry
    R = -1 * np.ones((len(U), len(I)))

    for lines in range(len(data)):
        R[data['UserID'][lines], data['MovieID'][lines]] = data['Prediction'][lines]

    # Replace -1 with NAN
    R[R == -1] = np.inf

    # Items rated by users
    # To create U_i, we create a matrix with User as first entry and item as second entry
    U_i = []
    for column in range(R.shape[1]):
        U_i.append(U[~np.isinf(R[:,column])])

    return U, I, R, U_i
    
def get_pickle(non_dom_users_file):
    """
        Return an object loaded from a pickle file

    :param      non_dom_users_file: Name of the file to load (need to be a pickle file)

    :return:    Object loaded
    """

    return pickle.load(open(non_dom_users_file, "rb"))


def pearson_vector(R, i, U_i, I):
    """
       Create the vector of pearson's correlation for an active user against all the users in the set.

    :param      R: Matrix of Ratings
    :param      i: An active user
    :param      U_i: List of array of users that have rated item i
    :param      I: List of items

    :return:    The vector of pearson's correlation
    """
    
    # Users that have rated item i
    users_i = U_i[i]
    
    corr_vec = []
    
    for x in I:
        if x != i:
            A_ix = list(set(users_i).intersection(set(U_i[x])))
            corr_vec.append(pearson(R[A_ix, i], R[A_ix, x]))

    return corr_vec
    
def pearson(x, y):
    """
        Pearson value for two vectors X and Y
        
    :param      x: vector of float values
    :param      y: vector of float values
    
    :return:    Pearson value between the two vectors
    """
    
    if len(x) == 0:
        return 0
    else:
        mx = np.mean(x)
        my = np.mean(y)
        
        new_x = x-mx
        new_y = x-my
        
        return (np.dot(new_x, new_y))/(np.linalg.norm(new_x)*np.linalg.norm(new_y))
        
def cosine_vector(R, i, U_i, I):
    """
       Create the vector of cosine values for an active user against all the users in the set.

    :param      R: Matrix of Ratings
    :param      i: An active user
    :param      U_i: List of array of users that have rated item i
    :param      I: List of items

    :return:    The vector of cosine values
    """

    # Users that have rated item i
    users_i = U_i[i]
    
    cosine_vec = []
    
    for x in I:
        if x != i:
            A_ix = list(set(users_i).intersection(set(U_i[x])))
            cosine_vec.append(cosine(R[A_ix, i], R[A_ix, x]))

    return cosine_vec
    
def cosine(x, y):
    """
        Cosine value for two vectors X and Y
        
    :param      x: vector of float values
    :param      y: vector of float values
    
    :return:    Cosine value between the two vectors
    """
    
    if len(x) == 0:
        return 0
    else:
        return (np.dot(x, y))/(np.linalg.norm(x)*np.linalg.norm(y))

    
def msd_vector(R, i, U_i, I):
    """
       Create the vector of MSD (Mean Squared Difference) values for an active user against all the users in the set.

    :param      R: Matrix of Ratings
    :param      i: An active user
    :param      U_i: List of array of users that have rated item i
    :param      I: List of items
    
    :return:    The vector of MSD values
    """

    # Users that have rated item i
    users_i = U_i[i]
    
    msd_vec = []
    for x in I:
        if x != i:
            A_ix = list(set(users_i).intersection(set(U_i[x])))
            msd_vec.append(msd(R[A_ix, i], R[A_ix, x]))
            
    return msd_vec
    
def msd(x, y):
    """
        MSD value for two vectors X and Y
        
    :param      x: vector of float values
    :param      y: vector of float values
    
    :return:    MSD value between the two vectors
    """
    # WARNING: We hardcode the max and min value here
    max_ = 5
    min_ = 1

    if len(x) == 0:
        return -np.inf
    else:
        return 1 - (1 / len(x)) * np.sum(((x - y) / (max_ - min_)) ** 2)
        
def nearest_neighbors(folder, R, I, U_i, method="msd"):
    print("Start Calculating Nearest Neighbors for method %s"%method)
    for item in I:
        
        if method == 'msd':
            vec = msd_vector(R, item, U_i, I)
        elif method == 'pearson':
            vec = pearson_vector(R, item, U_i, I)
        elif method == 'cosine':
            vec = cosine_vector(R, item, U_i, I)
            
        sorted_vec = sorted(enumerate(vec), key=itemgetter(1), reverse=True)
        
        NN_u = [i[0] for i in sorted_vec]
        sim_u = [i[1] for i in sorted_vec]
        
        # Save the files
        filename_NN = folder + "NN_" + str(item) + "_" + method + ".pickle"
        pickle.dump(NN_u, open(filename_NN, "wb"))

        filename_sim = folder + "sim_" + str(item) + "_" + method + ".pickle"
        pickle.dump(sim_u, open(filename_sim, "wb"))


        if (item + 1) % 100 == 0:
            print("  %i/%i done!" % (item + 1, len(I)))
            
def rmse(a,b):
    return np.sqrt(np.mean((np.array(a)-np.array(b))**2))

def mean(i, R, R_G):
    if len(R_G) == 0:
        ratings = R[:,i]
        return np.mean(ratings[ratings!=np.inf])
    else:
        return np.mean(R_G)
    
def weighted_mean(u, i, neigh, sim, R, R_G):
    if len(R_G) == 0:
        ratings = R[:,i]
        return np.mean(ratings[ratings!=np.inf])
    else:
        return np.sum(sim*R_G)/sum(sim)
    
def deviation_from_mean(u, i, neigh, sim, R, R_G):
    if len(R_G) == 0:
        ratings = R[:,i]
        return np.mean(ratings[ratings!=np.inf])
    else:
        ratings = R[u,:]
        rating_u_mean = np.mean(ratings[ratings!=np.inf])
        deviation = 0
        for i in range(len(R_G)):
            rate_n = R[neigh[i],:]
            deviation += sim[i]*(R_G[i]-np.mean(rate_n[rate_n!=np.inf]))
        
        return rating_u_mean + deviation/np.sum(sim)

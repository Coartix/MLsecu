import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import utils.data_exploration_utils as deu
import utils.data_preparation_utils as dpu

def get_list_of_attack_types(dataframe):
    '''
    Retrieves the list of attack types of a pandas dataframe
    
    :param dataframe: input dataframe
    :return: list of attack types
    '''
    if dataframe is None:
        return None
    
    return dataframe['attack_type'].unique().tolist()

def get_nb_of_attack_types(dataframe):
    '''
    Retrieves the number of attack types of a pandas dataframe
    
    :param dataframe: input dataframe
    :return: the number of attack types
    '''
    if dataframe is None:
        return None
    
    return len(get_list_of_attack_types(dataframe))

def get_list_of_if_outliers(dataframe, outlier_fraction: float):
    '''
    Extract the list of outliers according to Isolation Forest algorithm
    
    :param dataframe: input dataframe
    :param outlier_fraction: rate of outliers to be extracted
    :return: list of outliers according to Isolation Forest algorithm
    '''
    if dataframe is None:
        return None
    
    train = dpu.get_one_hot_encoded_dataframe(dataframe)
    train = dpu.remove_nan_through_mean_imputation(train)

    clf = IsolationForest(contamination=outlier_fraction, random_state=42)
    clf.fit(train)
    y_pred = clf.predict(train)
    return train.index[y_pred == -1].tolist()

def get_list_of_lof_outliers(dataframe, outlier_fraction):
    '''
    Extract the list of outliers according to Local Outlier Factor algorithm
    
    :param dataframe: input dataframe
    :param outlier_fraction: rate of outliers to be extracted
    :return: list of outliers according to Local Outlier Factor algorithm
    '''
    if dataframe is None:
        return None
    
    train = dpu.get_one_hot_encoded_dataframe(dataframe)
    train = dpu.remove_nan_through_mean_imputation(train)

    clf = LocalOutlierFactor(n_neighbors=20, contamination=outlier_fraction)
    y_pred = clf.fit_predict(train)
    return train.index[y_pred == -1].tolist()

def get_list_of_parameters(dataframe):
    '''
    Retrieves the list of parameters of a pandas dataframe
    
    :param dataframe: input dataframe
    :return: list of parameters
    '''
    if dataframe is None:
        return None

    return dataframe.columns.tolist()

def get_nb_of_if_outliers(dataframe, outlier_fraction):
    '''
    Extract the number of outliers according to Isolation Forest algorithm
    
    :param dataframe: input dataframe
    :param outlier_fraction: rate of outliers to be extracted
    :return: number of outliers according to Isolation Forest algorithm
    '''
    if dataframe is None:
        return None
    return len(get_list_of_if_outliers(dataframe, outlier_fraction))

def get_nb_of_lof_outliers(dataframe, outlier_fraction):
    '''
    Extract the number of outliers according to Local Outlier Factor algorithm
    
    :param dataframe: input dataframe
    :param outlier_fraction: rate of outliers to be extracted
    :return: number of outliers according to Local Outlier Factor algorithm
    '''
    if dataframe is None:
        return None
    return len(get_list_of_lof_outliers(dataframe, outlier_fraction))

def get_nb_of_occurrences(dataframe):
    '''
    Retrieves the number of occurrences of a pandas dataframe
    
    :param dataframe: input dataframe
    :return: number of occurrences
    '''
    return deu.get_nb_of_rows(dataframe)


def get_nb_of_parameters(dataframe):
    '''
    Retrieves the number of parameters of a pandas dataframe
    
    :param dataframe: input dataframe
    :return: number of parameters
    '''
    if dataframe is None:
        return None
    return len(get_list_of_parameters(dataframe))
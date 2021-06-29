import os
import torch
import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression


def load_data(feature_list):
    '''
    # For Linux
    cd = os.getcwd()
    x_eicu = pd.read_csv(cd+'/data/x_eicu.csv')
    y_eicu = pd.read_csv(cd+'/data/y_eicu.csv')
    '''
    path = os.getcwd()+'/../'
    x_eicu = pd.read_csv(path+'/data/x_eicu.csv')
    y_eicu = pd.read_csv(path+'/data/y_eicu.csv')
    assert np.all(x_eicu['patientunitstayid'].to_numpy()
                  == y_eicu['patientunitstayid'].to_numpy())
    x_eicu = x_eicu.drop(columns=['patientunitstayid'])
    features = x_eicu.columns
    if feature_list == None:
        x_eicu = x_eicu.to_numpy()
    else:
        x_eicu = x_eicu[feature_list].to_numpy()
    y_eicu = y_eicu['actualicumortality'].to_numpy()
    x = x_eicu
    y = y_eicu
    shuffler = np.random.permutation(len(x))
    return x[shuffler], y[shuffler], features


def impute_scale(x_train, x_test, random_state):
    imputer = SimpleImputer()
    estimator = BayesianRidge(n_iter=1000)
    '''
    imputer = IterativeImputer(estimator=estimator,
                               verbose=1,
                               random_state=random_state,
                               max_iter=1000)
    '''
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(x_train))
    x_test = scaler.transform(imputer.transform(x_test))
    return x_train, x_test


def impute(x_train, x_test, random_state):
    imputer = IterativeImputer()
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)
    return x_train, x_test


def to_tensor(x_train_fs, y_train, device):
    x_train = torch.from_numpy(x_train_fs).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    return x_train, y_train


'''
def to_tensor(x_train_fs, y_train, x_test_fs, device):
    x_train = torch.from_numpy(x_train_fs).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    x_test = torch.from_numpy(x_test_fs).float().to(device)
    return x_train, y_train, x_test
'''

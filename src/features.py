from numpy.linalg import eig
from numpy import cov, mean, array
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, RFECV, VarianceThreshold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, mutual_info_classif
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import Lasso, Ridge, ElasticNet, ElasticNetCV


def variance_thresh(x_train, y_train, x_test, n):
    scaler = StandardScaler()
    varx = pd.DataFrame(x_train)
    varx = (varx.var())
    vary = pd.DataFrame(x_test)
    vary = (vary.var())

    thresholder = VarianceThreshold(threshold=n)
    x_train = thresholder.fit_transform(x_train)
    x_train = scaler.fit_transform(x_train)

    x_test = thresholder.transform(x_test)
    x_test = scaler.transform(x_test)

    return x_train, x_test


def rfecv(x_train, y_train, x_test):
    estimator = LogisticRegressionCV(max_iter=2000, n_jobs=-1, random_state=1)
    fs = RFECV(
        estimator=estimator,
        step=1,
        n_jobs=-1,
        scoring='roc_auc',
        verbose=2,
        min_features_to_select=1
    )
    x_train_fs = fs.fit_transform(x_train, y_train)
    x_test_fs = fs.transform(x_test)
    return x_train_fs, x_test_fs, fs


def rfe(x_train, y_train, x_test, n):
    estimator = LogisticRegressionCV(max_iter=2000, n_jobs=-1, random_state=1)
    fs = RFE(estimator=estimator,
             verbose=1,
             n_features_to_select=n
             )
    x_train_fs = fs.fit_transform(x_train, y_train)
    x_test_fs = fs.transform(x_test)
    return x_train_fs, x_test_fs, fs


def anova(x_train, y_train, x_test):
    # configure to select all features
    fs = SelectKBest(score_func=f_classif, k='all')
    # learn relationship from training data
    fs.fit(x_train, y_train)
    # transform train input data
    x_train_fs = fs.transform(x_train)
    # transform test input data
    x_test_fs = fs.transform(x_test)
    return x_train_fs, x_test_fs, fs


def elastic_nets(x_train, y_train, x_test):
    cv_model = ElasticNetCV(
        l1_ratio=[.1, .5, .7, .9, .95, .99, .995],
        # l1_ratio=0.001,
        # l1_ratio=0.9999999,
        eps=0.001,
        alphas=[0.0005, 0.001, 0.01, 0.03, 0.05, 0.1],
        n_alphas=100,
        fit_intercept=True,
        normalize=True,
        precompute='auto',
        max_iter=2000,
        tol=0.0001,
        cv=5,
        copy_X=True,
        verbose=0,
        n_jobs=-1,
        positive=False,
        random_state=10,
        selection='cyclic',
    )
    cv_model.fit(x_train, y_train)
    elastic = ElasticNet(l1_ratio=cv_model.l1_ratio_, alpha=cv_model.alpha_,
                         max_iter=cv_model.n_iter_, fit_intercept=True, normalize=True,
                         random_state=10)

    fs = SelectFromModel(elastic)
    x_train_fs = fs.fit_transform(x_train, y_train)
    x_test_fs = fs.transform(x_test)
    return x_train_fs, x_test_fs, fs


def sklearnPCA(x_train, y_train, x_test):
    fs = PCA(n_components=90)
    x_train_fs = fs.fit_transform(x_train, y_train)
    x_test_fs = fs.transform(x_test)
    return x_train_fs, x_test_fs, fs


def kpca(x_train, y_train, x_test):
    fs = KernelPCA(n_components=50, kernel='linear', n_jobs=-1)
    x_train_fs = fs.fit_transform(x_train, y_train)
    x_test_fs = fs.transform(x_test)
    return x_train_fs, x_test_fs, fs


def mutual_info(x, y, n_best):
    score = mutual_info_classif(x, y.to_numpy()[:, 1])[::-1]
    x = x[:, score[:n_best]]
    return x, y



class MyPCA:
    def __init__(self):
        self.all_components = np.empty(0)
        self.components = np.empty(0)
        self.n_components = 0
        self.variance_explained = np.empty(0)
        self.ratio_var_explained = np.empty(0)
        self.cumulative_ratio_var_explained = np.empty(0)
        self.S = 0

    def fit(self, X, n_components):

        # Find the eigendecomp of the covariance of X
        self.S = cov(X.T)
        eig_vals, eig_vecs = np.linalg.eig(self.S)

        # sort the eigenvalues in descending order
        sorted_indexes = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[sorted_indexes]

        # similarly sort the eigenvectors
        eig_vecs = eig_vecs[:, sorted_indexes]

        # pick n_components
        self.all_components = eig_vecs.T
        self.n_components = n_components
        self.components = self.all_components[0: self.n_components]

        # PCA variance info
        ratio_var_explained = (eig_vals / self.S.shape[0]).round(6)
        cumu_ratio_var_explained = ratio_var_explained.cumsum()
        cumu_ratio_var_explained[-1] = 1
        self.variance_explained = eig_vals
        self.ratio_var_explained = ratio_var_explained
        self.cumulative_ratio_var_explained = cumu_ratio_var_explained
        return

    def transform(self, X):
        return X @ self.components.T


def scratchPCA(x_train, y_train, x_test, n):
    fs = MyPCA()
    fs.fit(x_train, n_components=n)
    x_train_fs = fs.transform(x_train)
    x_test_fs = fs.transform(x_test)
    return x_train_fs, x_test_fs, fs

    '''
        # in the __init__ function
        # self.variance_explained = np.empty(0)
        # self.ratio_var_explained = np.empty(0)
        # self.cumulative_ratio_var_explained = np.empty(0)
        ratio_var_explained = (eig_vals / S.shape[0]).round(6)
        cumu_ratio_var_explained = ratio_var_explained.cumsum()
        cumu_ratio_var_explained[-1] = 1
        self.variance_explained = eig_vals
        self.ratio_var_explained = ratio_var_explained
        self.cumulative_ratio_var_explained = cumu_ratio_var_explained
        '''

import os
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif


def main():
    # Begin dataset Processing
    print('Processing Data...')
    id = 'patientunitstayid'    # I get tired of typing this
    labels = pd.read_csv(os.getcwd()+'/../data/apachePatientResult.csv')
    labels = labels[[id, 'actualicumortality']].replace(['ALIVE', 'EXPIRED'], [0, 1]).drop_duplicates()
    apachevar = pd.read_csv(os.getcwd()+'/../data/apacheApsVar.csv')
    feat_list = [id, 'intubated', 'wbc', 'temperature', 'respiratoryrate', 'sodium', 'heartrate', 'meanbp',
                 'ph', 'hematocrit', 'creatinine', 'albumin', 'pao2', 'pco2', 'bun', 'glucose', 'bilirubin', 'fio2']
    apachevar = apachevar[feat_list].replace(-1, np.nan)
    labs = pd.read_csv(os.getcwd()+'/../data/lab.csv')
    labs = labs.pivot_table(index=id, columns='labname', values='labresult').replace(-1, np.nan)
    indicators = pd.read_csv(os.getcwd()+'/../data/apachePredVar.csv')
    feat_list = [id, 'gender', 'age', 'verbal', 'motor', 'eyes', 'aids', 'hepaticfailure',
                 'lymphoma', 'metastaticcancer', 'leukemia', 'immunosuppression', 'cirrhosis', 'midur',
                 'oobventday1', 'oobintubday1', 'diabetes', 'pao2', 'fio2', 'creatinine']
    indicators = indicators[feat_list].replace(-1, np.nan)
    print('Merging input sets')
    data = labs.merge(apachevar, on=id).merge(indicators, on=id)
    x = data.loc[data[id].isin(labels[id])]
    y = labels.loc[labels[id].isin(x[id])]
    # Dataset is processed
    print('Missingness Eval')
    missing = x.isnull().sum() / len(x)
    print(x.isnull().sum() / len(x))
    tolerance = 0.75    # Do some research into what level of missingness is acceptable and set this accordingly
    too_much = np.hstack(np.argwhere(missing.to_numpy() > tolerance))
    x = x.drop(x.columns[too_much], axis=1)
    x = x.drop(columns=['patientunitstayid'])
    print('Dropped {} columns'.format(len(too_much)))

    # This is where you would need to start feature selection
    print('Imputing...')
    x = IterativeImputer(verbose=1).fit_transform(x.to_numpy()[:, 1:])
    x = StandardScaler().fit_transform(x)
    print('Imputing and Scaling Done')
    print('Mutual Info...')
    less = np.hstack(np.argwhere(missing.to_numpy() < tolerance))
    cols = data.columns[less[1:]]
    cols = data.columns

    # Mutual information is model independent so I can just do it all in one shot
    score = mutual_info_classif(x, y.to_numpy()[:, 1])
    print(cols[np.argsort(score)[::-1]])
    np.savetxt('../results/mi_ordered_feats.csv', cols[np.argsort(score)[::-1]].to_list(), delimiter=',')


if __name__ == '__main__':
    main()


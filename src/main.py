from train import train_model
from data_loader import load_data, impute_scale, to_tensor, impute
from evaluate import test_model, make_metrics
from features import rfecv, anova, sklearnPCA, kpca, scratchPCA, elastic_nets, rfe, variance_thresh
from sklearn.model_selection import StratifiedKFold, train_test_split as tts
from scipy import stats
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import os
import torch

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
warnings.filterwarnings("ignore")


def training_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("-------------")
        print("GPU available")
        print("-------------")
        print("Training on:", device)
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    return device


def main():
    FEATURE_LIST_RFECV = ["platelets x 1000", "phosphate", "paO2",
                          "paCO2", "pH", "myoglobin", "lactate", "haptoglobin",
                          "glucose_x", "direct bilirubin", "creatinine_x", "chloride",
                          "bicarbonate", "potassium", "bedside glucose", "ammonia",
                          "albumin_x", "WBC x 1000", "fio2_y", "Vent Rate",
                          "Vancomycin - trough", "Temperature", "TV", "T4", "T3",
                          "RDW", "RBC", "Pressure Support", "anion gap", "prealbumin",
                          "sodium_x", "total bilirubin", "pao2_y", "diabetes",
                          "oobintubday1", "oobventday1", "cirrhosis", "leukemia",
                          "metastaticcancer", "eyes", "motor", "age", "fio2_x", "bilirubin",
                          "glucose_y", "pao2_x", "albumin_y", "creatinine_y", "hematocrit",
                          "ph", "meanbp", "heartrate", "sodium_y", "respiratoryrate",
                          "temperature", "urinary sodium", "urinary osmolality",
                          "troponin - T", "transferrin", "total protein",
                          "total cholesterol", "PTT", "PT - INR", "Vitamin B12",
                          "creatinine", "-monos", "MCHC", "MCH", "Lithium", "LPM O2",
                          "LDL", "Hgb", "Hct", "MCV", "HSV 1&2 IgG AB titer", "HCO3",
                          "Fe/TIBC Ratio", "Cyclosporin", "Carboxyhemoglobin", "AST (SGOT)",
                          "Base Excess", "Base Deficit", "BUN", "PT", "MPV", "-polys",
                          "O2 Sat (%)", "-lymphs", "-basos", "O2 Content", "-eos"
                          ]
    elastic_features = [
        "lactate",
        "motor",
        "paCO2",
        "bicarbonate",
        "fio2_x",
        "BUN",
        "WBC x 1000",
        "Base Excess",
        "sodium_x",
        "age",
        "oobventday1",
        "albumin_x",
        "glucose_x",
        "temperature",
        "AST (SGOT)",
        "chloride",
        "pao2_y",
        "diabetes",
        "total bilirubin",
        "respiratoryrate",
        "eyes",
        "potassium",
        "glucose_y",
        "Temperature",
        "PT",
        "phosphate",
        "Base Deficit",
        "oobintubday1",
        "RDW",
        "O2 Sat (%)",
        "platelets x 1000",
        "HCO3",
        "-eos",
        "sodium_y",
        "anion gap",
        "heartrate",
        "bun",
        "Pressure Support",
        "pco2",
        "RBC",
        "bedside glucose",
        "-bands",
        "-monos",
        "Unnamed: 0",
        "PT - INR",
        "metastaticcancer",
        "PTT",
        "total protein",
        "bilirubin",
        "MCV",
        "creatinine",
        "wbc",
        "MPV",
        "troponin - T",
        "-polys",
        "Vent Rate",
        "MCHC",
        "ammonia",
        "Vancomycin - trough",
        "LPM O2",
        "fibrinogen",
        "hematocrit",
        "HSV 1&2 IgG AB titer",
        "meanbp",
        "Carboxyhemoglobin",
        "alkaline phos.",
        "paO2",
        "WBC's in body fluid",
        "WBC's in pericardial fluid",
        "immunosuppression",
        "creatinine_y",
        "ph",
        "CPK-MB",
        "Peak Airway/Pressure",
        "urinary osmolality",
        "intubated",
        "Vancomycin - random",
        "O2 Content",
        "haptoglobin",
        "WBC's in cerebrospinal fluid",
        "calcium",
        "albumin_y",
        "Digoxin",
        "LDH",
        "leukemia",
        "urinary sodium",
        "uric acid",
        "FiO2",
        "creatinine_x",
        "Cyclosporin",
        "ESR",
        "Theophylline",
        "Tobramycin - peak",
        "Total CO2",
        "lipase",
        "troponin - I",
        "Amikacin - random",
        "HDL",
        "ionized calcium",
        "Gentamicin - trough",
        "aids",
        "WBC's in peritoneal fluid",
        "BNP",
        "Lithium",
        "CPK",
        "cortisol",
        "CRP-hs",
        "urinary specific gravity",
        "LDL",
        "Tacrolimus-FK506",
        "prealbumin",
        "glucose - CSF",
        "myoglobin",
        "urinary creatinine",
        "triglycerides",
        "cirrhosis",
        "Device",
        "Oxyhemoglobin",
        "Vitamin B12",
        "Respiratory Rate",
        "verbal",
        "TIBC",
        "Tobramycin - random",
        "transferrin",
        "Fe/TIBC Ratio",
        "-basos",
        "Vent Other",
        "ethanol",
        "magnesium",
        "Phenobarbital",
        "WBC's in synovial fluid",
        "Pressure Control",
        "T3",
        "Vancomycin - peak",
        "protein - CSF",
        "serum ketones",
        "Ferritin",
        "pao2_x",
        "fio2_y"]
    elastic_31total = [
        'lactate',
        'motor',
        'paCO2',
        'bicarbonate',
        'fio2_x',
        'BUN',
        'WBC x 1000',
        'Base Excess',
        'sodium_x',
        'age',
        'oobventday1',
        'albumin_x',
        'glucose_x',
        'temperature',
        'AST (SGOT)',
        'chloride',
        'pao2_y',
        'diabetes',
        'total bilirubin',
        'respiratoryrate',
        'eyes',
        'potassium',
        'glucose_y',
        'Temperature',
        'PT',
        'phosphate',
        'Base Deficit',
        'oobintubday1',
        'RDW',
        'O2 Sat (%)',
        'platelets x 1000']
    RFE_25 = [
        "potassium",
        "paCO2",
        "pH",
        "myoglobin",
        "lactate",
        "glucose_x",
        "respiratoryrate",
        "chloride",
        "albumin_x",
        "Hct",
        "TV",
        "MCV",
        "RDW",
        "RBC",
        "bicarbonate",
        "BUN",
        "sodium_x",
        "diabetes",
        "-eos",
        "-lymphs",
        "oobventday1",
        "motor",
        "age",
        "fio2_x",
        "heartrate"]
    RFE_50 = [
        "platelets x 1000",
        "paO2",
        "paCO2",
        "pH",
        "myoglobin",
        "lactate",
        "glucose_x",
        "chloride",
        "bicarbonate",
        "bedside glucose",
        "anion gap",
        "MPV",
        "albumin_x",
        "WBC x 1000",
        "fio2_y",
        "Hct",
        "Hgb",
        "Temperature",
        "TV",
        "RDW",
        "RBC",
        "PT",
        "MCHC",
        "MCV",
        "WBC's in pleural fluid",
        "sodium_x",
        "potassium",
        "total bilirubin",
        "diabetes",
        "oobintubday1",
        "-eos",
        "oobventday1",
        "immunosuppression",
        "eyes",
        "motor",
        "verbal",
        "age",
        "glucose_y",
        "bun",
        "fio2_x",
        "ph",
        "BUN",
        "Base Excess",
        "heartrate",
        "sodium_y",
        "respiratoryrate",
        "temperature",
        "hematocrit",
        "O2 Sat (%)",
        "meanbp"]
    feat_37 = [
        "platelets x 1000",
        "TV",
        "Temperature",
        "-eos",
        "age",
        "respiratoryrate",
        "temperature",
        "PT",
        "pH",
        "lactate",
        "eyes",
        "motor",
        "paCO2",
        "BUN",
        "WBC x 1000",
        "myoglobin",
        "ph",
        "RBC",
        "MCHC",
        "Hgb",
        "Hct",
        "bun",
        "RDW",
        "heartrate",
        "glucose_y",
        "fio2_y",
        "diabetes",
        "oobventday1",
        "chloride",
        "bedside glucose",
        "glucose_x",
        "bicarbonate",
        "fio2_x",
        "total bilirubin",
        "hematocrit",
        "sodium_x",
        "albumin_x"]
    anovalist = ['lactate', 'motor', 'eyes', 'verbal', 'oobintubday1', 'oobventday1', 'AST (SGOT)',
                 'PT - INR', 'anion gap', 'ALT (SGPT)', 'PT', 'intubated', 'BUN', 'bicarbonate',
                 'ph', 'phosphate', 'albumin_x', 'fio2_x', 'fio2_y', 'total protein', 'Base Excess',
                 'temperature', 'WBC x 1000', 'HCO3', 'bun', 'potassium', 'RDW', 'total bilirubin',
                 'albumin_y', 'calcium', 'wbc', 'PTT', 'Base Deficit', 'creatinine_x', 'heartrate',
                 'glucose_x', 'creatinine_y', 'creatinine', 'ammonia', 'MCHC', '-monos', 'bilirubin',
                 '-eos', 'alkaline phos.', 'platelets x 1000', 'Hgb', 'respiratoryrate', 'direct bilirubin',
                 'RBC', 'O2 Sat (%)', 'magnesium', 'age', 'Hct', '-bands', 'glucose_y', 'sodium_x', 'cortisol',
                 'MCV', 'hematocrit', '-polys', 'bedside glucose', 'LDH', 'Total CO2', 'CPK-MB', 'Vancomycin - trough',
                 'MPV', 'meanbp', 'metastaticcancer', 'LPM O2', 'amylase', 'FiO2', 'hepaticfailure', 'cirrhosis',
                 'chloride', 'urinary sodium', 'Ferritin', 'total cholesterol', 'BNP', 'sodium_y', 'immunosuppression',
                 'Pressure Support', 'O2 Content', 'Respiratory Rate', 'prealbumin', 'leukemia', 'diabetes', 'Digoxin',
                 'haptoglobin', 'Fe/TIBC Ratio', 'pco2', 'Methemoglobin', 'Temperature', 'ionized calcium', 'uric acid',
                 'HDL', 'serum osmolality', 'CPK', 'urinary specific gravity', 'TIBC', 'Peak Airway/Pressure', 'troponin - I',
                 'paO2', 'troponin - T', 'CRP-hs', 'Vancomycin - random', 'Tacrolimus-FK506', 'Carboxyhemoglobin', 'Vitamin B12',
                 'transferrin', '-basos', 'urinary creatinine', 'HSV 1&2 IgG AB titer', '''WBC's in body fluid''', 'Oxyhemoglobin',
                 'lymphoma', 'Acetaminophen', 'Fe', 'Amikacin - random', 'Tobramycin - random', '''WBC's in pericardial fluid''',
                 'PTT ratio', 'aids', 'Pressure Control', 'Vent Other', 'T3', 'Tobramycin - peak', 'Gentamicin - trough',
                 'urinary osmolality', 'lipase', 'protein C', 'reticulocyte count', '''WBC's in cerebrospinal fluid''',
                 '''WBC's in peritoneal fluid''', 'PEEP', 'folate', 'ethanol', 'triglycerides', 'Amikacin - trough',
                 '''WBC's in urine''', 'fibrinogen', 'CRP', 'prolactin', 'TSH', 'Theophylline', 'Device',
                 'Cyclosporin', '24 h urine protein', 'Tobramycin - trough', 'Gentamicin - peak', 'Phenytoin',
                 'midur', '-lymphs', 'LDL', '''WBC's in pleural fluid''', 'pH', 'TV', 'free T4', 'T3RU',
                 'cd 4', 'ANF/ANA', 'gender', 'protein S', 'MCH', 'Lithium', 'Vancomycin - peak', 'Phenobarbital',
                 'protein - CSF', 'Carbamazepine', 'glucose - CSF', 'Lidocaine', 'Vent Rate', '''WBC's in synovial fluid''',
                 'salicylate', 'Gentamicin - random', 'T4', 'pao2_x', 'pao2_y', 'myoglobin', 'ESR', 'Spontaneous Rate',
                 '24 h urine urea nitrogen', 'serum ketones', 'CPK-MB INDEX', 'paCO2', 'Mode', 'NAPA', 'Site',
                 'Clostridium difficile toxin A+B', 'Procainamide', 'Amikacin - peak', 'Legionella pneumophila Ab',
                 'HSV 1&2 IgG AB', 'HIV 1&2 AB', 'RPR titer']

    device = training_device()
    mean_1 = []
    std_1 = []
    roc_mean = []
    roc_confidence_interval = []
    counter = 1
    random_state = 1
    n = 1
    kfold = StratifiedKFold(n_splits=10)

    x = np.arange(195)
    y = []
    for i in range(195):
        y.append(0.918066)
    plt.plot(x, y, 'r-', label='Benchmark')
    plt.fill_between(x, 0.915318, 0.920814, color='r', alpha=0.2)
    plt.xlabel('Features')
    plt.ylabel('ROC_AUC')

    # ####################### #
    #       ELASTIC NET       #
    # ####################### #
    for i in range(139):
        x, y, features = load_data(feature_list=elastic_features[0:i+1])
        logits_all = []
        labels_all = []
        for train_index, test_index in kfold.split(x, y):
            print('\nK-Fold {}/10'.format(counter))

            # TRAIN-TEST SPLIT
            x_train, y_train = x[train_index], y[train_index]
            x_test, y_test = x[test_index], y[test_index]

            # IMPUTE & SCALE
            x_train, x_test = impute_scale(x_train, x_test, random_state)

            # CONVERT TO TENSOR FOR MODEL TRAINING
            x_train, y_train = to_tensor(x_train, y_train, device)

            # MODEL TRAINING
            num_features = x_train.size()[1]
            print("Training on ", num_features, "Features")
            model = train_model(num_features, device, x_train, y_train)

            # MODEL EVALUATION
            logits = test_model(model, x_test, y_test, device)
            logits_all.append(logits.reshape(-1))
            labels_all.append(y_test)
            counter += 1
            # ---------------------------------------------------------------
        # METRICS
        print('\n[METRICS]')
        roc_mean, roc_confidence_interval = make_metrics(
            logits_all, labels_all, num_features)
        mean_1.append(roc_mean)
        std_1.append(roc_confidence_interval)

    x1 = np.arange(len(mean_1)+1)
    mean_1.insert(0, 0)
    std_1.insert(0, 0)
    np.savetxt("mean_ElasticNet.csv", mean_1, delimiter=",")
    np.savetxt("std_ElasticNet.csv", std_1, delimiter=",")
    mean_1 = np.array(mean_1)
    std_1 = np.array(std_1)
    plt.plot(x1, mean_1, 'b-', label='Elastic Net')
    plt.fill_between(x1, mean_1 - std_1, mean_1 +
                     std_1, color='b', alpha=0.2)

    # ####################### #
    #           PCA           #
    # ####################### #
    mean_1 = []
    std_1 = []
    roc_mean = []
    roc_confidence_interval = []
    for i in range(194):
        x, y, features = load_data(feature_list=None)
        logits_all = []
        labels_all = []
        for train_index, test_index in kfold.split(x, y):
            print('\nK-Fold {}/10'.format(counter))

            # TRAIN-TEST SPLIT
            x_train, y_train = x[train_index], y[train_index]
            x_test, y_test = x[test_index], y[test_index]

            # IMPUTE & SCALE
            x_train, x_test = impute_scale(x_train, x_test, random_state)

            # FEATURE SELECTION
            x_train, x_test, fs = scratchPCA(x_train, y_train, x_test, n=n)

            # CONVERT TO TENSOR FOR MODEL TRAINING
            x_train, y_train = to_tensor(x_train, y_train, device)

            # MODEL TRAINING
            num_features = x_train.size()[1]
            print("Training on ", num_features, "Features")
            model = train_model(num_features, device, x_train, y_train)

            # MODEL EVALUATION
            logits = test_model(model, x_test, y_test, device)
            logits_all.append(logits.reshape(-1))
            labels_all.append(y_test)
            counter += 1
            # ---------------------------------------------------------------
        # METRICS
        print('\n[METRICS]')
        roc_mean, roc_confidence_interval = make_metrics(
            logits_all, labels_all, num_features)
        mean_1.append(roc_mean)
        std_1.append(roc_confidence_interval)
        n += 1

    x1 = np.arange(len(mean_1)+1)
    mean_1.insert(0, 0)
    std_1.insert(0, 0)
    np.savetxt("mean_PCA.csv", mean_1, delimiter=",")
    np.savetxt("std_PCA.csv", std_1, delimiter=",")
    mean_1 = np.array(mean_1)
    std_1 = np.array(std_1)
    plt.plot(x1, mean_1, 'g-', label='PCA')
    plt.fill_between(x1, mean_1 - std_1, mean_1 +
                     std_1, color='g', alpha=0.2)

    # ####################### #
    #           RFE           #
    # ####################### #
    mean_1 = []
    std_1 = []
    roc_mean = []
    roc_confidence_interval = []
    n = 1
    for i in range(50):
        x, y, features = load_data(feature_list=RFE_50[0:i+1])
        logits_all = []
        labels_all = []
        for train_index, test_index in kfold.split(x, y):
            print('\nK-Fold {}/10'.format(counter))

            # TRAIN-TEST SPLIT
            x_train, y_train = x[train_index], y[train_index]
            x_test, y_test = x[test_index], y[test_index]

            # IMPUTE & SCALE
            x_train, x_test = impute_scale(x_train, x_test, random_state)

            # CONVERT TO TENSOR FOR MODEL TRAINING
            x_train, y_train = to_tensor(x_train, y_train, device)

            # MODEL TRAINING
            num_features = x_train.size()[1]
            print("Training on ", num_features, "Features")
            model = train_model(num_features, device, x_train, y_train)

            # MODEL EVALUATION
            logits = test_model(model, x_test, y_test, device)
            logits_all.append(logits.reshape(-1))
            labels_all.append(y_test)
            counter += 1
            # ---------------------------------------------------------------
        # METRICS
        print('\n[METRICS]')
        roc_mean, roc_confidence_interval = make_metrics(
            logits_all, labels_all, num_features)
        mean_1.append(roc_mean)
        std_1.append(roc_confidence_interval)
        n += 1

    x1 = np.arange(len(mean_1)+1)
    mean_1.insert(0, 0)
    std_1.insert(0, 0)
    np.savetxt("mean_RFE.csv", mean_1, delimiter=",")
    np.savetxt("std_RFE.csv", std_1, delimiter=",")
    mean_1 = np.array(mean_1)
    std_1 = np.array(std_1)
    plt.plot(x1, mean_1, 'c-', label='RFE')
    plt.fill_between(x1, mean_1 - std_1, mean_1 +
                     std_1, color='c', alpha=0.2)

    # ####################### #
    #         ANOVA           #
    # ####################### #
    mean_1 = []
    std_1 = []
    roc_mean = []
    roc_confidence_interval = []
    n = 1
    for i in range(194):
        x, y, features = load_data(feature_list=anovalist[0:i+1])
        logits_all = []
        labels_all = []
        for train_index, test_index in kfold.split(x, y):
            print('\nK-Fold {}/10'.format(counter))

            # TRAIN-TEST SPLIT
            x_train, y_train = x[train_index], y[train_index]
            x_test, y_test = x[test_index], y[test_index]

            # IMPUTE & SCALE
            x_train, x_test = impute_scale(x_train, x_test, random_state)

            # FEATURE SELECTION
            x_train, x_test, fs = anova(x_train, y_train, x_test)

            # CONVERT TO TENSOR FOR MODEL TRAINING
            x_train, y_train = to_tensor(x_train, y_train, device)

            # MODEL TRAINING
            num_features = x_train.size()[1]
            print("Training on ", num_features, "Features")
            model = train_model(num_features, device, x_train, y_train)

            # MODEL EVALUATION
            logits = test_model(model, x_test, y_test, device)
            logits_all.append(logits.reshape(-1))
            labels_all.append(y_test)
            counter += 1
            # ---------------------------------------------------------------
        # METRICS
        print('\n[METRICS]')
        roc_mean, roc_confidence_interval = make_metrics(
            logits_all, labels_all, num_features)
        mean_1.append(roc_mean)
        std_1.append(roc_confidence_interval)
        n += 1

    x1 = np.arange(len(mean_1)+1)
    mean_1.insert(0, 0)
    std_1.insert(0, 0)
    np.savetxt("mean_Anova.csv", mean_1, delimiter=",")
    np.savetxt("std_Anova.csv", std_1, delimiter=",")
    mean_1 = np.array(mean_1)
    std_1 = np.array(std_1)
    plt.plot(x1, mean_1, 'k-', label='ANOVA')
    plt.fill_between(x1, mean_1 - std_1, mean_1 +
                     std_1, color='k', alpha=0.2)

    # ####################### #
    #   Variance Threshold    #
    # ####################### #
    mean_1 = []
    std_1 = []
    roc_mean = []
    roc_confidence_interval = []
    x1 = []
    thresholdValues = [0.0005, 0.005, 0.1, 0.5, 1, 10, 50, 100,
                       200, 400, 1600, 3200, 6400]
    k = 0

    while k <= 12:
        x, y, features = load_data(feature_list=None)
        logits_all = []
        labels_all = []
        j = thresholdValues[k]
        for train_index, test_index in kfold.split(x, y):
            print('\nK-Fold {}/10'.format(counter))

            # TRAIN-TEST SPLIT
            x_train, y_train = x[train_index], y[train_index]
            x_test, y_test = x[test_index], y[test_index]

            # IMPUTE & SCALE
            x_train, x_test = impute(x_train, x_test, random_state)

            # FEATURE SELECTION
            x_train, x_test = variance_thresh(
                x_train, y_train, x_test, n=j)

            # CONVERT TO TENSOR FOR MODEL TRAINING
            x_train, y_train = to_tensor(x_train, y_train, device)

            # MODEL TRAINING
            num_features = x_train.size()[1]
            print("Training on ", num_features, "Features")
            model = train_model(num_features, device, x_train, y_train)

            # MODEL EVALUATION
            logits = test_model(model, x_test, y_test, device)
            logits_all.append(logits.reshape(-1))
            labels_all.append(y_test)
            counter += 1
            # ---------------------------------------------------------------
        # METRICS
        print('\n[METRICS]')
        roc_mean, roc_confidence_interval = make_metrics(
            logits_all, labels_all, num_features)
        mean_1.append(roc_mean)
        std_1.append(roc_confidence_interval)
        x1.append(num_features)
        k += 1

    x1 = x1[::-1]
    mean_1 = mean_1[::-1]

    x1.insert(0, 0)
    mean_1.insert(0, 0)
    std_1.insert(0, 0)

    std_1 = np.array(std_1)
    mean_1 = np.array(mean_1)

    np.savetxt("mean_VT.csv", mean_1, delimiter=",")
    np.savetxt("std_VT.csv", std_1, delimiter=",")

    plt.plot(x1, mean_1, 'm-', label='Variance Threshold')
    plt.fill_between(x1, mean_1 - std_1, mean_1 +
                     std_1, color='m', alpha=0.2)

    plt.legend()
    plt.show()
    milist = np.load(os.getcwd()+'/../results/mi_ordered_feats.npy', allow_pickle=True)
    kfold = StratifiedKFold(n_splits=10)
    random_state = 1
    device = 'cuda:0'
    n = 0
    mean_1, std_1 = list(), list()
    milist = elastic_features
    for i in range(194):
        x, y, features = load_data(feature_list=milist[0:i+1])
        logits_all = []
        labels_all = []
        counter = 0
        for train_index, test_index in kfold.split(x, y):
            print('\nK-Fold {}/10'.format(counter))

            # TRAIN-TEST SPLIT
            x_train, y_train = x[train_index], y[train_index]
            x_test, y_test = x[test_index], y[test_index]

            # IMPUTE & SCALE
            x_train, x_test = impute_scale(x_train, x_test, random_state)

            # FEATURE SELECTION
            x_train, x_test, fs = anova(x_train, y_train, x_test)

            # CONVERT TO TENSOR FOR MODEL TRAINING
            x_train, y_train = to_tensor(x_train, y_train, device)

            # MODEL TRAINING
            num_features = x_train.size()[1]
            print("Training on ", num_features, "Features")
            model = train_model(num_features, device, x_train, y_train)

            # MODEL EVALUATION
            logits = test_model(model, x_test, y_test, device)
            logits_all.append(logits.reshape(-1))
            labels_all.append(y_test)
            counter += 1
            if num_features % 5 == 0:
                np.savetxt("mean_mi.csv", mean_1, delimiter=",")
                np.savetxt("std_mi.csv", std_1, delimiter=",")
            # ---------------------------------------------------------------
        # METRICS
        print('\n[METRICS]')
        roc_mean, roc_confidence_interval = make_metrics(
            logits_all, labels_all)
        mean_1.append(roc_mean)
        std_1.append(roc_confidence_interval)
        n += 1

    x1 = np.arange(len(mean_1)+1)
    mean_1.insert(0, 0)
    std_1.insert(0, 0)
    np.savetxt("mean_mi.csv", mean_1, delimiter=",")
    np.savetxt("std_mi.csv", std_1, delimiter=",")


if __name__ == '__main__':
    main()

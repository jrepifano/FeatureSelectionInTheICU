import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics


class Load:
    def __init__(self, path):
        self.path = path

    def vt(self):
        VTmean = pd.read_csv(self.path+'mean_VT.csv')
        VTmean = np.array(VTmean)
        VTmean = VTmean.reshape((13,))
        VTmean = np.insert(VTmean, 0, 0)
        VTmean = np.append(VTmean, 0.918066)
        VTstd = pd.read_csv(self.path+'std_VT.csv')
        VTstd = np.array(VTstd)
        VTstd = VTstd.reshape((13,))
        VTstd = np.insert(VTstd, 0, 0)
        VTstd = np.append(VTstd, 0.002748)
        return VTmean, VTstd

    def anova(self):
        Anovamean = pd.read_csv(self.path+'mean_Anova.csv')
        Anovamean = np.array(Anovamean)
        Anovamean = Anovamean.reshape((194,))
        Anovamean = np.insert(Anovamean, 0, 0)
        Anovamean = np.append(Anovamean, 0.918066)
        Anovastd = pd.read_csv(self.path+'std_Anova.csv')
        Anovastd = np.array(Anovastd)
        Anovastd = Anovastd.reshape((194,))
        Anovastd = np.insert(Anovastd, 0, 0)
        Anovastd = np.append(Anovastd, 0.002748)
        return Anovamean, Anovastd

    def enet(self):
        ENETmean = pd.read_csv(self.path+'mean_ElasticNet.csv')
        ENETmean = np.array(ENETmean)
        ENETmean = ENETmean.reshape((139,))
        ENETmean = np.insert(ENETmean, 0, 0)
        ENETmean = np.append(ENETmean, 0.918066)
        ENETstd = pd.read_csv(self.path+'std_ElasticNet.csv')
        ENETstd = np.array(ENETstd)
        ENETstd = ENETstd.reshape((139,))
        ENETstd = np.insert(ENETstd, 0, 0)
        ENETstd = np.append(ENETstd, 0.002748)
        return ENETmean, ENETstd

    def pca(self):
        PCAmean = pd.read_csv(self.path+'mean_PCA.csv')
        PCAmean = np.array(PCAmean)
        PCAmean = PCAmean.reshape((194,))
        PCAmean = np.insert(PCAmean, 0, 0)
        PCAstd = pd.read_csv(self.path+'std_PCA.csv')
        PCAstd = np.array(PCAstd)
        PCAstd = PCAstd.reshape((194,))
        PCAstd = np.insert(PCAstd, 0, 0)
        return PCAmean, PCAstd

    def rfe(self):
        RFEmean = pd.read_csv(self.path+'mean_RFE.csv')
        RFEmean = np.array(RFEmean)
        RFEmean = RFEmean.reshape((50,))
        RFEmean = np.insert(RFEmean, 0, 0)
        RFEmean = np.append(RFEmean, 0.918066)
        RFEstd = pd.read_csv(self.path+'std_RFE.csv')
        RFEstd = np.array(RFEstd)
        RFEstd = RFEstd.reshape((50,))
        RFEstd = np.insert(RFEstd, 0, 0)
        RFEstd = np.append(RFEstd, 0.002748)
        return RFEmean, RFEstd

    def mi(self):
        mimean = pd.read_csv(self.path+'mean_mi.csv')
        mimean = np.array(mimean)
        mimean = mimean.reshape((-1))
        mimean = np.insert(mimean, 0, 0)
        mimean = np.append(mimean, 0.918066)
        mistd = pd.read_csv(self.path+'std_mi.csv')
        mistd = np.array(mistd)
        mistd = mistd.reshape((-1))
        mistd = np.insert(mistd, 0, 0)
        mistd = np.append(mistd, 0.002748)
        return mimean, mistd

# matplotlib.rcParams['text.usetex'] = True
load = Load(os.getcwd()+'/../results/')
PCAmean, PCAstd = load.pca()
RFEmean, RFEstd = load.rfe()
ENETmean, ENETstd = load.enet()
Anovamean, Anovastd = load.anova()
VTmean, VTstd = load.vt()
mimean, mistd = load.mi()

# BENCHMARK
x = np.arange(195)
y = []
for i in range(195):
    y.append(0.918066)
plt.plot(x, y, 'r-', label='All Features')
plt.fill_between(x, 0.915318, 0.920814, color='r', alpha=0.2)
plt.xlabel('Number of Features')
plt.ylabel('ROC AUC')

# VARIANCE THRESHOLD
x1 = [0, 25, 27, 34, 43, 53, 64, 71, 102, 131, 141, 158, 179, 187, 195]
plt.plot(x1, VTmean, 'm-',
         label='Variance Threshold; AUC: {:.2f}'.format(metrics.auc(x1, VTmean)))
plt.fill_between(x1, VTmean - VTstd, VTmean +
                 VTstd, color='m', alpha=0.2)

# ANOVA
x1 = np.arange(len(Anovamean))
plt.plot(x1, Anovamean, 'k-',
         label='ANOVA; AUC: {:.2f}'.format(metrics.auc(x1, Anovamean)))
plt.fill_between(x1, Anovamean - Anovastd, Anovamean +
                 Anovastd, color='k', alpha=0.2)

# MI
x1 = np.arange(len(mimean))
x1[-1] = 195
plt.plot(x1, mimean, 'y-',
         label='Mutual Information; AUC: {:.2f}'.format(metrics.auc(x1, mimean)))
plt.fill_between(x1, mimean - mistd, mimean +
                 mistd, color='y', alpha=0.2)

# RFE
x1 = np.arange(len(RFEmean)-1)
x1 = np.append(x1, 195)
plt.plot(x1, RFEmean, 'c-',
         label='RFE; AUC: {:.2f}'.format(metrics.auc(x1, RFEmean)))
plt.fill_between(x1, RFEmean - RFEstd, RFEmean +
                 RFEstd, color='c', alpha=0.2)

# PCA
x1 = np.arange(len(PCAmean))
plt.plot(x1, PCAmean, 'g-',
         label='PCA; AUC: {:.2f}'.format(metrics.auc(x1, PCAmean)))
plt.fill_between(x1, PCAmean - PCAstd, PCAmean +
                 PCAstd, color='g', alpha=0.2)

# ELASTIC NET
x1 = np.arange(len(ENETmean)-1)
x1 = np.append(x1, 195)
plt.plot(x1, ENETmean, 'b-',
         label='Elastic Net; AUC: {:.2f}'.format(metrics.auc(x1, ENETmean)))
plt.fill_between(x1, ENETmean-ENETstd, ENETmean+ENETstd,
                 color='b', alpha=0.2)


plt.legend()
plt.show()

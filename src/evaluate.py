import torch
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score


def test_model(model, x_test, y_test, device):
    model.eval()
    outputs = model.forward(torch.from_numpy(x_test).float().to(device))
    logits = torch.sigmoid(outputs).detach().cpu().numpy()
    return logits


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def make_metrics(logits_all, labels_all):
    accuracy = []
    precision = []
    sensitivity = []
    specificity = []
    roc_auc = []
    prc_auc = []
    balanced_acc = []
    for i in range(len(logits_all)):
        tn, fp, fn, tp = confusion_matrix(
            labels_all[i], np.round(logits_all[i])).ravel()
        accuracy.append((tp + tn) / (tp + tn + fp + fn))
        precision.append(tp / (tp + fp))
        sensitivity.append(tp / (tp + fn))
        specificity.append(tn / (tn + fp))
        roc_auc.append(roc_auc_score(labels_all[i], logits_all[i]))
        prc_auc.append(average_precision_score(labels_all[i], logits_all[i]))
        balanced_acc.append(balanced_accuracy_score(
            labels_all[i], np.round(logits_all[i])))

    acc_mean, acc_confidence_interval = mean_confidence_interval(accuracy)
    print('Accuracy Mean and confidence interval: {:4f}, {:4f}'.format(
        acc_mean, acc_confidence_interval))

    prec_mean, prec_confidence_interval = mean_confidence_interval(precision)
    print('Precision Mean and confidence interval: {:4f}, {:4f}'.format(
        prec_mean, prec_confidence_interval))

    sens_mean, sens_confidence_interval = mean_confidence_interval(sensitivity)
    print('Sensitivity Mean and confidence interval: {:4f}, {:4f}'.format(
        sens_mean, sens_confidence_interval))

    spec_mean, spec_confidence_interval = mean_confidence_interval(specificity)
    print('Specificity Mean and confidence interval: {:4f}, {:4f}'.format(
        spec_mean, spec_confidence_interval))

    roc_mean, roc_confidence_interval = mean_confidence_interval(roc_auc)
    print('ROC_AUC Mean and confidence interval: {:4f}, {:4f}'.format(
        roc_mean, roc_confidence_interval))

    prc_mean, prc_confidence_interval = mean_confidence_interval(prc_auc)
    print('PRC_AUC Mean and confidence interval: {:4f}, {:4f}'.format(
        prc_mean, prc_confidence_interval))

    bacc_mean, bacc_confidence_interval = mean_confidence_interval(
        balanced_acc)
    print('Balanced Accuracy Mean and confidence interval: {:4f}, {:4f}'.format(
        bacc_mean, bacc_confidence_interval))

    all_low = 0.918066 - 1.2*0.002748
    all_high = 0.918066 + 1.2*0.002748
    reduced_low = roc_mean - 1.2*roc_confidence_interval
    reduced_high = roc_mean + 1.2*roc_confidence_interval

    if all_low < reduced_low:
        low = all_low
    else:
        low = reduced_low
    if all_high > reduced_high:
        high = all_high
    else:
        high = reduced_high
    '''
    labels = ['195', num_features]
    x_pos = np.arange(len(labels))
    ROC = [0.918696, roc_mean]
    CI = [0.003557, roc_confidence_interval]
    fig, ax = plt.subplots()
    ax.bar(x_pos, ROC, yerr=CI, capsize=25)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_xlabel('# of Features')
    ax.set_ylabel('%')
    ax.set_title('Mean ROC_AUC')
    ax.set_ylim(low, high)
    plt.tight_layout()
    plt.show()
    '''
    return roc_mean, roc_confidence_interval

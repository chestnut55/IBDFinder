import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score, auc
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import utils
from scipy import interp

from GEDFN import gedfn
from DFN import dfn


def plot_beauty():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    X, y, left, right = utils.load()

    gedfn_tprs = []
    gedfn_aucs = []
    dfn_tprs = []
    dfn_aucs = []
    rf_tprs = []
    rf_aucs = []

    mean_fpr = np.linspace(0, 1, 100)

    y_true = []
    gedfn_y_score = []
    dfn_y_score = []
    rf_y_score = []

    cv = StratifiedKFold(n_splits=5, random_state=0)
    # for train_idx,test_idx in cv.split(X,y):
    #     X_train, X_test, y_train, y_test = X.ix[train_idx], X.ix[test_idx], y[train_idx], y[test_idx]
    for i in np.arange(0, 10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
        y_true.append(y_test)
        ################GEDFN################################
        _, _, _, _, _, y_score = gedfn(X_train, X_test,
                                       to_categorical(y_train), to_categorical(y_test), left, right)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        gedfn_tprs.append(interp(mean_fpr, fpr, tpr))
        gedfn_tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        gedfn_aucs.append(roc_auc)

        gedfn_y_score.append(y_score)
        precision, recall, _ = precision_recall_curve(y_test, y_score)

        ################DFN################################
        _, _, _, _, _, y_score = dfn(X_train, X_test,
                                     to_categorical(y_train), to_categorical(y_test))
        fpr, tpr, _ = roc_curve(y_test, y_score)
        dfn_tprs.append(interp(mean_fpr, fpr, tpr))
        dfn_tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        dfn_aucs.append(roc_auc)

        dfn_y_score.append(y_score)
        precision, recall, _ = precision_recall_curve(y_test, y_score)

        ################Random Forest################################
        _, _, _, _, _, y_score = utils.rf(X_train, X_test, y_train, y_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        rf_tprs.append(interp(mean_fpr, fpr, tpr))
        rf_tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        rf_aucs.append(roc_auc)

        rf_y_score.append(y_score)
        precision, recall, _ = precision_recall_curve(y_test, y_score)

    y_true = np.concatenate(y_true, axis=0)

    gedfn_y_score = np.concatenate(gedfn_y_score, axis=0)
    dfn_y_score = np.concatenate(dfn_y_score, axis=0)
    rf_y_score = np.concatenate(rf_y_score, axis=0)

    axes[0].plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=.8)
    ################GEDFN################################
    mean_tpr = np.mean(gedfn_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(gedfn_aucs)
    axes[0].plot(mean_fpr, mean_tpr, color='r',
                 label=r'GEDFN (%0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

    precision, recall, _ = precision_recall_curve(y_true, gedfn_y_score)
    ap = round(average_precision_score(y_true, gedfn_y_score), 3)
    axes[1].plot(recall, precision, label='GEDFN:AP=' + str(ap), color='r')
    ################DFN################################
    mean_tpr = np.mean(dfn_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(dfn_aucs)
    axes[0].plot(mean_fpr, mean_tpr, color='g',
                 label=r'DFN (%0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

    precision, recall, _ = precision_recall_curve(y_true, dfn_y_score)
    ap = round(average_precision_score(y_true, dfn_y_score), 3)
    axes[1].plot(recall, precision, label='DFN:AP=' + str(ap), color='g')
    ################Random Forest################################
    mean_tpr = np.mean(rf_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(rf_aucs)
    axes[0].plot(mean_fpr, mean_tpr, color='b',
                 label=r'RF (%0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

    precision, recall, _ = precision_recall_curve(y_true, rf_y_score)
    ap = round(average_precision_score(y_true, rf_y_score), 3)
    axes[1].plot(recall, precision, label='RF:AP=' + str(ap), color='b')
    ##############################################################
    axes[0].set_xlim([-0.05, 1.05])
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('AUC curve')
    axes[0].legend(loc='best')

    axes[1].set_xlim([-0.05, 1.05])
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall curve')
    axes[1].legend(loc='best')

    fig.tight_layout()
    fig.savefig('output/roc_curve.png')
    plt.show()


def plot():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    X, y, left, right = utils.load()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=0)

    _, _, _, _, _, y_score = gedfn(X_train, X_test,
                                   to_categorical(y_train), to_categorical(y_test), left, right)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    ap = round(average_precision_score(y_test, y_score), 3)
    axes[0].plot(recall, precision, label='GEDFN:AP=' + str(ap), color='r')
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = round(roc_auc_score(y_test, y_score), 3)
    axes[1].plot(fpr, tpr, label='GEDFN:' + str(auc), color='r')

    _, _, _, _, _, y_score = dfn(X_train, X_test, to_categorical(y_train),
                                 to_categorical(y_test))

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    ap = round(average_precision_score(y_test, y_score), 3)
    axes[0].plot(recall, precision, label='DFN:AP=' + str(ap), color='g')

    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = round(roc_auc_score(y_test, y_score), 3)
    axes[1].plot(fpr, tpr, label='DFN:' + str(auc), color='g')

    _, _, _, _, _, y_score = utils.rf(X_train, X_test, y_train, y_test)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    ap = round(average_precision_score(y_test, y_score), 3)
    axes[0].plot(recall, precision,label='RF:AP=' + str(ap), color='b')

    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = round(roc_auc_score(y_test, y_score), 3)
    axes[1].plot(fpr, tpr,label='RF:' + str(auc), color='b')

    axes[0].legend(loc='best')
    axes[0].set_title('Precision-Recall curve')
    axes[0].set_ylim([0, 1])
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')

    axes[1].legend(loc='best')
    axes[1].set_title('AUC curve')
    axes[1].set_ylim([0, 1])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Postive Rate')

    fig.tight_layout()
    fig.savefig('roc_curve.png')
    plt.show()


if __name__ == "__main__":
    plot_beauty()
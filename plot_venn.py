import pymrmr

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from matplotlib_venn import venn3
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from GEDFN import gedfn
import utils


def plot_venn(X, y, left, right):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    # RFE
    model = LogisticRegression()
    rfe = RFE(model, 100, step=5)
    fit = rfe.fit(X_train, y_train)
    rfe_results = X.columns.values[fit.support_]

    # mRMR
    mRMR_df = X.copy()
    mRMR_df.insert(loc=0, column='class', value=y)
    # mRMR_df.to_csv('output/mRMR_df.csv',index=False)
    mRMR_results = pymrmr.mRMR(mRMR_df, 'MID', 100)

    # random forest
    rf = RandomForestClassifier(random_state=0, n_estimators=200)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    rf_results = pd.Series(importances, index=X.columns.values).sort_values(ascending=False).index.values

    # graph embedding feedforward neural network
    gedfn(X_train, X_test, to_categorical(y_train), to_categorical(y_test), left, right)
    var_ibd = np.loadtxt('output/var_ibd.csv', delimiter=",")
    nodes = X.columns.values.tolist()
    gedfn_results = pd.Series(var_ibd, index=nodes).sort_values(ascending=False).index.values

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    venn3([set(gedfn_results[:100]), set(rf_results[:100]), set(rfe_results)],
          set_labels=('GEDFN', 'RF', 'RFE'), set_colors=('r', 'g', 'c'), ax=axes[0][0])

    venn3([set(gedfn_results[:100]), set(rf_results[:100]), set(mRMR_results)],
          set_labels=('GEDFN', 'RF', 'mRMR'), set_colors=('r', 'g', 'y'), ax=axes[0][1])

    venn3([set(gedfn_results[:100]), set(rfe_results), set(mRMR_results)],
          set_labels=('GEDFN', 'RFE', 'mRMR'), set_colors=('r', 'c', 'y'), ax=axes[1][0])

    venn3([set(rf_results[:100]), set(rfe_results), set(mRMR_results)],
          set_labels=('RF', 'RFE', 'mRMR'), set_colors=('g', 'c', 'y'), ax=axes[1][1])

    fig.tight_layout()
    plt.savefig('output/venn.png')
    plt.show()


if __name__ == "__main__":
    X, y, left, right = utils.load()
    plot_venn(X, y, left, right)

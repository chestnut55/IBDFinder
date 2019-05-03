import pymrmr
import venn
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from matplotlib_venn import venn3
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from keras_gemlp import gemlp
from GEDFN import gedfn
import utils
import seaborn as sns


def generate_venn(X, y, left, right):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    top_selected_features = 50
    # RFE
    model = LogisticRegression()
    rfe = RFE(model, top_selected_features, step=5)
    fit = rfe.fit(X_train, y_train)
    rfe_results = X.columns.values[fit.support_]

    # mRMR
    mRMR_df = X.copy()
    mRMR_df.insert(loc=0, column='class', value=y)
    # mRMR_df.to_csv('output/mRMR_df.csv',index=False)
    mRMR_results = pymrmr.mRMR(mRMR_df, 'MID', top_selected_features)

    # random forest
    rf = RandomForestClassifier(random_state=0, n_estimators=100)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    rf_results = pd.Series(importances, index=X.columns.values).sort_values(ascending=False).index.values

    # graph embedding feedforward neural network
    gedfn(X_train, X_test, to_categorical(y_train), to_categorical(y_test), left, right)
    var_ibd = np.loadtxt('output/var_ibd.csv', delimiter=",")
    nodes = X.columns.values.tolist()
    gedfn_results = pd.Series(var_ibd, index=nodes).sort_values(ascending=False).index.values

    df_top_features = pd.DataFrame(columns=['rfe', 'mRMR', 'rf', 'gedfn'])
    df_top_features['rfe'] = rfe_results
    df_top_features['mRMR'] = mRMR_results
    df_top_features['rf'] = rf_results[:top_selected_features]
    df_top_features['gedfn'] = gedfn_results[:top_selected_features]

    df_top_features.to_csv('output/venn.txt')


def plot_venn3():
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    df = pd.read_csv('output/venn.txt', sep=',')
    venn3([set(df['gedfn'].values), set(df['rf'].values), set(df['rfe'].values)],
          set_labels=('GEDFN', 'RF', 'RFE'), set_colors=('r', 'g', 'c'), ax=axes[0][0])

    venn3([set(df['gedfn'].values), set(df['rf'].values), set(df['mRMR'].values)],
          set_labels=('GEDFN', 'RF', 'mRMR'), set_colors=('r', 'g', 'y'), ax=axes[0][1])

    venn3([set(df['gedfn'].values), set(df['rfe'].values), set(df['mRMR'].values)],
          set_labels=('GEDFN', 'RFE', 'mRMR'), set_colors=('r', 'c', 'y'), ax=axes[1][0])

    venn3([set(df['rf'].values), set(df['rfe'].values), set(df['mRMR'].values)],
          set_labels=('RF', 'RFE', 'mRMR'), set_colors=('g', 'c', 'y'), ax=axes[1][1])

    fig.tight_layout()
    plt.savefig('output/venn3.png')
    plt.show()


def plot_venn4():
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    df = pd.read_csv('output/venn.txt', sep=',')
    labels = venn.get_labels([df['rfe'].values, df['mRMR'].values, df['rf'].values, df['gedfn'].values])
    venn.venn4(labels, names=['RFE', 'mRMR', 'RF', 'GEMLP'], fig=fig,  ax=axes[0])

    df_gedfn = pd.read_csv('output/feature_selection_result.csv', header=0, sep=',', index_col=0)
    df_gedfn = df_gedfn.head(9)
    _df_gedfn = df_gedfn[['Accuracy', 'AUC', 'Precision', 'Recall']]
    _df_gedfn.plot(linestyle='-',marker='.', ax=axes[1],color=['r','g','b','y'])
    axes[1].set_ylim([0.5, 1])
    axes[1].set_xticklabels(np.arange(10, 50, 5))
    axes[1].set_xlabel('#feature')
    axes[1].set_title('GEMLP')

    fig.tight_layout()
    plt.savefig('output/venn4.png')
    plt.show()


if __name__ == "__main__":
    X, y, left, right = utils.load()
    # generate_venn(X, y, right, right)
    # plot_venn3()
    plot_venn4()

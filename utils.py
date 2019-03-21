import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from io import StringIO
from Bio import Phylo


def load():
    '''
    load the training data and adj matrix
    :return:
    '''
    sparcc = pd.read_csv('sparcc/sparcc_otu_adj.txt', sep='\t', index_col=0).values
    mic = pd.read_csv('MIC/mic_otu_adj.txt', sep='\t', index_col=0).values
    spieceasi = pd.read_csv('Spieceasi/spieceasi_adj_out.txt', sep='\t', index_col=0).values

    identity_mat = np.identity(n=spieceasi.shape[0], dtype=np.int)
    spieceasi = spieceasi + identity_mat

    merged = sparcc + mic + spieceasi
    merged[merged > 1] = 1

    # U = union_Adjac_matrix(left, right)
    X = pd.read_csv('output/ibd_otus.txt', sep='\t', index_col=0, header=0)
    X = shuffle(X)
    y = X['label'].values
    X = X.drop(columns=['label'])
    # X = preprocessing.normalize(X, axis=1)
    # print(check_symmetric(spieceasi))
    merged_flat_list = [item for sublist in merged.tolist() for item in sublist]

    sparcc_flat_list = [item for sublist in sparcc.tolist() for item in sublist]

    spieceasi_flat_list = [item for sublist in spieceasi.tolist() for item in sublist]

    print("merged=" + str(merged_flat_list.count(1)), "sparcc=" + str(sparcc_flat_list.count(1)),
          "spieceasi=" + str(spieceasi_flat_list.count(1)))

    return X, y, sparcc, merged


def parse_CoNet():
    samples = pd.read_csv('data/coNet_id_file.txt', sep='\t')['SampleID'].values
    df = pd.read_csv('data/coNet_metadata_file.txt', sep='\t', index_col=0)
    df = df.ix[samples, :]
    df[df.diagnosis == 'no'] = 0
    df[df.diagnosis == 'IC'] = 1
    df[df.diagnosis == 'CD'] = 1
    df[df.diagnosis == 'UC'] = 1
    df.to_csv('data/coNet_metadata_file.txt', sep='\t')


def apply_column_filter(row):
    taxa = row[1]
    taxonomy = taxa.rfind(';')
    taxa = taxa[:taxonomy]
    return taxa


def parse_taxa():
    df = pd.read_csv('data/exported_out_feature_table/taxonomy.tsv', sep='\t')
    se = set(df.apply(apply_column_filter, axis=1).values)
    print(len(se))


# random forest
def rf(x_train, x_test, y_train, y_test):
    rf = RandomForestClassifier(random_state=0, n_estimators=200)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    y_score = rf.predict_proba(x_test)[:, 1]

    acc = round(accuracy_score(y_test, y_pred), 3)
    f1 = round(f1_score(y_test, y_pred), 3)
    precision = round(precision_score(y_test, y_pred), 3)
    recall = round(recall_score(y_test, y_pred), 3)

    auc = round(roc_auc_score(y_test, y_score), 3)

    print("Random Forest Testing accuracy: ", acc, " Testing auc: ", auc, " Testing f1: ",
          f1, " Testing precision: ", precision, " Testing recall: ", recall)
    return acc, auc, f1, precision, recall, y_score


def rf_ranked_feature_selection(X_train, y_train):
    rf = RandomForestClassifier(random_state=0, n_estimators=200)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    results = pd.Series(importances,
                        index=X_train.columns.values).sort_values(ascending=False).index.values

    return results


# SVM
def svm(x_train, x_test, y_train, y_test):
    svm = SVC(probability=True)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    y_score = svm.predict_proba(x_test)[:, 1]

    acc = round(accuracy_score(y_test, y_pred), 3)
    f1 = round(f1_score(y_test, y_pred), 3)
    precision = round(precision_score(y_test, y_pred), 3)
    recall = round(recall_score(y_test, y_pred), 3)

    auc = round(roc_auc_score(y_test, y_score), 3)

    print("SVM Testing accuracy: ", acc, " Testing auc: ", auc, " Testing f1: ",
          f1, " Testing precision: ", precision, " Testing recall: ", recall)
    return acc, auc, f1, precision, recall, y_score


def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)


def union_Adjac_matrix(A, B):
    '''
    union two Adjacency matrix
    :return:
    '''
    U = A + B
    U[U == 2] = 1
    return U


def read_phylo_tree():
    tree = Phylo.read("data/tree.nwk", "newick")
    Phylo.draw(tree)


if __name__ == "__main__":
    # A = pd.read_csv('sparcc/sparcc_otu_adj.txt', sep='\t', index_col=0).values
    # B = pd.read_csv('MIC/mic_otu_adj.txt', sep='\t', index_col=0).values
    # U = union_Adjac_matrix(A, B)
    # print(U)
    load()

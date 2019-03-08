import numpy as np
import pandas as pd
from sklearn import preprocessing


def load():
    '''
    load the training data and adj matrix
    :return:
    '''
    left = pd.read_csv('sparcc/sparcc_otu_adj.txt', sep='\t', index_col=0).values
    right = pd.read_csv('MIC/mic_otu_adj.txt', sep='\t', index_col=0).values

    X = pd.read_csv('output/ibd_otus.txt', sep='\t', index_col=0, header=0)
    y = X['label'].values
    X = X.drop(columns=['label'])
    # X = preprocessing.normalize(X, axis=1)

    return X, y, left, right


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


if __name__ == "__main__":
    load()

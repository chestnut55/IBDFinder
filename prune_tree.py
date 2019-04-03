from ete3 import Tree
import numpy as np
import pandas as pd
import nearest_neighbors
import utils
from sklearn.model_selection import train_test_split


def gemlp_prune_tree():
    with open('data/tree.nwk', 'r') as myfile:
        str_tree = myfile.read()
    tre = Tree(str_tree, format=5)

    # pruning
    top_features = 50
    var_ibd = np.loadtxt('output/var_ibd.csv', delimiter=",")
    df = pd.read_csv('output/ibd_otus.txt', sep='\t', index_col=0, header=0)
    nodes = df.columns.values.tolist()[:-1]
    gemlp_results = pd.Series(var_ibd, index=nodes).sort_values(ascending=False).index.values[:top_features]

    # for genus level, there are many id for one species
    id_paires = nearest_neighbors.taxon_id_name()
    taxa_id_list = []
    unique_species_name = set()

    for taxa_id, name in id_paires.items():
        if name in unique_species_name:
            continue
        unique_species_name.add(name)
        if name in gemlp_results:
            taxa_id_list.append(taxa_id)
    tre.prune(taxa_id_list)
    tre.write(format=5, outfile="output/filtered_gemlp_tree.nwk")


def rf_prune_tree():
    X, y, _left, _right = utils.load()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, stratify=y, random_state=0)

    results = utils.rf_ranked_feature_selection(X_train, y_train)

    with open('data/tree.nwk', 'r') as myfile:
        str_tree = myfile.read()
    tre = Tree(str_tree, format=5)
    top_features = 50
    results = results[:top_features]
    id_paires = nearest_neighbors.taxon_id_name()
    taxa_id_list = []
    unique_species_name = set()

    for taxa_id, name in id_paires.items():
        if name in unique_species_name:
            continue
        unique_species_name.add(name)
        if name in results:
            taxa_id_list.append(taxa_id)
    tre.prune(taxa_id_list)
    tre.write(format=5, outfile="output/filtered_rf_tree.nwk")


if __name__ == "__main__":
    rf_prune_tree()

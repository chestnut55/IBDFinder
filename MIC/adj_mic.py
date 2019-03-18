import pandas as pd
import numpy as np

mic_strength = pd.read_csv('results/strength.txt', delimiter="\t")
mic_strength = mic_strength.drop(['Class'], axis=1)
mic_strength = mic_strength[mic_strength.MICe > 0.1]
mic_strength = mic_strength.sort_values(by=['MICe'])
unique_nodes = list(set().union(mic_strength.Var1.values, mic_strength.Var2.values))
print('unique node length is :', len(unique_nodes))
# adj_nodes = pd.read_csv('ibd_otus_mic.txt', sep='\t', index_col=0, header=0).index.values.tolist()

adj_nodes = pd.read_csv('../output/ibd_otus.txt',sep='\t', index_col=0,header=0).columns.values.tolist()
adj_nodes = adj_nodes[:-1]

identity_mat = np.identity(n=len(adj_nodes), dtype=np.int)
df_ = pd.DataFrame(data=identity_mat, index=adj_nodes, columns=adj_nodes)
nodesA, nodesB = mic_strength.Var1.values, mic_strength.Var2.values

# phylogenetic tree
nearest_neighbors_matrix = pd.read_csv('../output/phy_neighbors.csv',sep='\t',index_col=0)
joined_nodes = set(adj_nodes).intersection(set(nearest_neighbors_matrix.columns.values))

for index, (value1, value2) in enumerate(zip(nodesA, nodesB)):
    df_[value1][value2] = 1
    df_[value2][value1] = 1

    # add neighbors of phylogenetic tree
    if value1 in joined_nodes:
        check_neighbors = nearest_neighbors_matrix.loc[nearest_neighbors_matrix[value1] == 1]
        if check_neighbors is not None:
            individual_neighbors = set(check_neighbors.index.values).intersection(joined_nodes)
            for ind in individual_neighbors:
                df_[value2][ind] = 1
                df_[ind][value2] = 1
    if value2 in joined_nodes:
        check_neighbors2 = nearest_neighbors_matrix.loc[nearest_neighbors_matrix[value2] == 1]
        if check_neighbors2 is not None:
            individual_neighbors = set(check_neighbors2.index.values).intersection(joined_nodes)
            for ind in individual_neighbors:
                df_[value1][ind] = 1
                df_[ind][value1] = 1
# df_ = df_ - np.diag(np.diag(df_))
df_.to_csv('mic_otu_adj.txt', sep='\t')

flat_list = [item for sublist in df_.values.tolist() for item in sublist]

print(flat_list.count(1))
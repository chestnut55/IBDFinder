import numpy as np
import pandas as pd
import nearest_neighbors
import utils



df = pd.read_csv('cor_sparcc.out_edges.txt', sep='\t')
nodesA = df['from'].values.tolist()
nodesB = df['to'].values.tolist()
# nodes = list(set().union(nodesA,nodesB))
nodes = pd.read_csv('../output/ibd_otus.txt',sep='\t', index_col=0,header=0).columns.values.tolist()
labels = nodes[-1]
nodes = nodes[:-1]
identity_mat = np.identity(n=len(nodes), dtype=np.int)
df_ = pd.DataFrame(data=identity_mat, index=nodes, columns=nodes)

nearest_neighbors_matrix = pd.read_csv('../output/phy_neighbors.csv',sep='\t',index_col=0)

joined_nodes = set(nodes).intersection(set(nearest_neighbors_matrix.columns.values))

for index, (value1, value2) in enumerate(zip(nodesA, nodesB)):
    df_[value1][value2] = 1
    df_[value2][value1] = 1

    # add neighbors of phylogenetic tree
    # if value1 in joined_nodes:
    #     check_neighbors = nearest_neighbors_matrix.loc[nearest_neighbors_matrix[value1] == 1]
    #     if check_neighbors is not None:
    #         individual_neighbors = set(check_neighbors.index.values).intersection(joined_nodes)
    #         for ind in individual_neighbors:
    #             df_[value2][ind] = 1
    #             df_[ind][value2] = 1
    # if value2 in joined_nodes:
    #     check_neighbors2 = nearest_neighbors_matrix.loc[nearest_neighbors_matrix[value2] == 1]
    #     if check_neighbors2 is not None:
    #         individual_neighbors = set(check_neighbors2.index.values).intersection(joined_nodes)
    #         for ind in individual_neighbors:
    #             df_[value1][ind] = 1
    #             df_[ind][value1] = 1

df_ = df_ - np.diag(np.diag(df_))
print('finish---------------------------')
df_.to_csv('sparcc_otu_adj.txt', sep='\t')
print(utils.check_symmetric(df_.values))

flat_list = [item for sublist in df_.values.tolist() for item in sublist]

print(flat_list.count(1))

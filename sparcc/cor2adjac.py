import numpy as np
import pandas as pd

df = pd.read_csv('cor_sparcc.out_edges.txt', sep='\t')
nodesA = df['from'].values.tolist()
nodesB = df['to'].values.tolist()
# nodes = list(set().union(nodesA,nodesB))
nodes = pd.read_csv('../output/ibd_otus.txt',sep='\t', index_col=0,header=0).columns.values.tolist()
labels = nodes[-1]
nodes = nodes[:-1]
identity_mat = np.identity(n=len(nodes), dtype=np.int)
df_ = pd.DataFrame(data=identity_mat, index=nodes, columns=nodes)

for index, (value1, value2) in enumerate(zip(nodesA, nodesB)):
    df_[value1][value2] = 1

print(df_.shape)
df_.to_csv('sparcc_otu_adj.txt', sep='\t')

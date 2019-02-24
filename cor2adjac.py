import numpy as np
import pandas as pd

df = pd.read_csv('output/cor_sparcc.out_edges.txt', sep='\t')
nodesA = df['from'].values.tolist()
nodesB = df['to'].values.tolist()
# nodes = list(set().union(nodesA,nodesB))
nodes = pd.read_csv('output/ibd_otus.csv',sep='\t', index_col=0,header=0).columns.values.tolist()
print(nodes)
print(len(nodes))
labels = nodes[-1]
nodes = nodes[:-1]
print(len(nodes))
identity_mat = np.identity(n=len(nodes), dtype=np.int)
df_ = pd.DataFrame(data=identity_mat, index=nodes, columns=nodes)

for index, (value1, value2) in enumerate(zip(nodesA, nodesB)):
    df_[value1][value2] = 1

print(df_.shape)
df_.to_csv('output/otu_adj', sep='\t')
np.savetxt('output/otu_adj.txt',df_.values,fmt='%d')

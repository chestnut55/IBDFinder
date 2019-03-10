import pandas as pd
import numpy as np

mic_strength = pd.read_csv('results/strength.txt', delimiter="\t")
mic_strength = mic_strength.drop(['Class'], axis=1)
mic_strength = mic_strength[mic_strength.MICe > 0.05]
mic_strength = mic_strength.sort_values(by=['MICe'])
unique_nodes = list(set().union(mic_strength.Var1.values, mic_strength.Var2.values))
print('unique node length is :', len(unique_nodes))
adj_nodes = pd.read_csv('ibd_otus_mic.txt', sep='\t', index_col=0, header=0).index.values.tolist()
identity_mat = np.identity(n=len(adj_nodes), dtype=np.int)
df_ = pd.DataFrame(data=identity_mat, index=adj_nodes, columns=adj_nodes)
nodesA, nodesB = mic_strength.Var1.values, mic_strength.Var2.values
for index, (value1, value2) in enumerate(zip(nodesA, nodesB)):
    df_[value1][value2] = 1
print(df_.shape)
df_.to_csv('mic_otu_adj.txt', sep='\t')
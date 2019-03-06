import pandas as pd
import numpy as np

mic_strength = pd.read_csv('output/mic_results/strength.txt', delimiter="\t")
mic_strength = mic_strength.drop(['Class'], axis=1)
# mic_strength = mic_strength[mic_strength.MICe > 0.01]
mic_strength = mic_strength.sort_values(by=['MICe'])
# nodes = list(set().union(mic_strength.Var1.values, mic_strength.Var2.values))

nodes = pd.read_csv('output/ibd_otus.csv',sep='\t', index_col=0,header=0).columns.values.tolist()
nodes = nodes[:-1]
identity_mat = np.identity(n=len(nodes), dtype=np.int)
df_ = pd.DataFrame(data=identity_mat, index=nodes, columns=nodes)
nodesA, nodesB = mic_strength.Var1.values, mic_strength.Var2.values
for index, (value1, value2) in enumerate(zip(nodesA, nodesB)):
    df_[value1][value2] = 1
print(df_.shape)
df_.to_csv('output/mic_otu_adj', sep='\t')
np.savetxt('output/mic_otu_adj.txt',df_.values,fmt='%d')
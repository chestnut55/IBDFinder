import pandas as pd
import numpy as np


adj_nodes = pd.read_csv('ibd_otus_mic.txt', sep='\t', index_col=0, header=0).index.values.tolist()
identity_mat = np.identity(n=len(adj_nodes), dtype=np.int)
df_ = pd.DataFrame(data=identity_mat, index=adj_nodes, columns=adj_nodes)

# diseased group
diseased_mic_strength = pd.read_csv('diseased_results/strength.txt', delimiter="\t")
diseased_mic_strength = diseased_mic_strength.drop(['Class'], axis=1)
diseased_mic_strength = diseased_mic_strength[diseased_mic_strength.MICe > 0.05]
diseased_mic_strength = diseased_mic_strength.sort_values(by=['MICe'])
diseased_unique_nodes = list(set().union(diseased_mic_strength.Var1.values, diseased_mic_strength.Var2.values))
print('diseased group unique node length is :', len(diseased_unique_nodes))

diseased_nodesA, diseased_nodesB = diseased_mic_strength.Var1.values, diseased_mic_strength.Var2.values
for index, (value1, value2) in enumerate(zip(diseased_nodesA, diseased_nodesB)):
    df_[value1][value2] = 1

# normal group
normal_mic_strength = pd.read_csv('normal_results/strength.txt', delimiter="\t")
normal_mic_strength = normal_mic_strength.drop(['Class'], axis=1)
normal_mic_strength = normal_mic_strength[normal_mic_strength.MICe > 0.05]
normal_mic_strength = normal_mic_strength.sort_values(by=['MICe'])
normal_unique_nodes = list(set().union(normal_mic_strength.Var1.values, normal_mic_strength.Var2.values))
print('normal group unique node length is :', len(normal_unique_nodes))

normal_nodesA, normal_nodesB = normal_mic_strength.Var1.values, normal_mic_strength.Var2.values
for index, (value1, value2) in enumerate(zip(normal_nodesA, normal_nodesB)):
    df_[value1][value2] = 1


print('whole unique node length is :', len(list(set().union(normal_unique_nodes, diseased_unique_nodes))))

df_.to_csv('mic_otu_adj.txt', sep='\t')
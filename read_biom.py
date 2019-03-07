from __future__ import print_function

import numpy as  np
import pandas as pd
from biom import load_table
from sklearn import preprocessing

metadata = pd.read_csv('data/mapping_file.txt', sep='\t', encoding='utf-8')

metadata = metadata[['#SampleID', 'sample_type', 'age',
                     'antibiotics', 'collection',
                     'biologics', 'biopsy_location', 'body_habitat',
                     'body_product', 'body_site',
                     'diagnosis', 'disease_extent', 'disease_stat',
                     'diseasesubtype', 'gastrointest_disord']]

metadata = metadata[metadata.sample_type == 'biopsy']
metadata = metadata[metadata.age.notnull()]
metadata = metadata[metadata.age <= 18.0]

# coNet_metadata = metadata[['#SampleID', 'diagnosis']]
# coNet_metadata.columns = ['SampleID', 'diagnosis']
# coNet_metadata.to_csv('data/coNet_metadata_file.txt', sep='\t', index=False)

ibd = metadata[metadata.diagnosis != 'no']['#SampleID'].values
control = metadata[metadata.diagnosis == 'no']['#SampleID'].values
#
#
ibd_abundance_table = load_table('data/otu_table.biom')
# ibd_abundance_table = load_table('data/exported_out_feature_table/feature-table.biom')
#
# collapse to genus level
genus_idx = 5
collapse_f = lambda id_, md: ';'.join(md['taxonomy'][:genus_idx + 1])
collapsed = ibd_abundance_table.collapse(collapse_f, axis='observation',norm=False)

# normalise
collapsed = collapsed.norm(axis='sample', inplace=False)
# ibd group
filter_ibd = lambda v, i_, m: [x for x in ibd if x == i_]
ibd_group = collapsed.filter(filter_ibd, axis='sample', inplace=False)
df_ibd = ibd_group.to_dataframe(dense=True)
# control group
filter_control = lambda v, i_, m: [x for x in control if x == i_]
control_group = collapsed.filter(filter_control, axis='sample', inplace=False)
df_control = control_group.to_dataframe(dense=True)

labels = [1] * len(df_ibd.columns.values) + [0] * len(df_control.columns.values)

results = pd.concat([df_ibd, df_control], axis=1).T
# values with all zero for the variable are removed
results = results.loc[:, (results != 0).any(axis=0)]

results.T.to_csv('output/ibd_otus_mic.txt', sep="\t")

results["label"] = labels

results.to_csv('output/ibd_otus.csv', sep="\t") # output for sparcc in order to calculate correlation

np.savetxt('output/otus.csv',results.values,fmt='%.2f',delimiter=",") # output for machine learning models
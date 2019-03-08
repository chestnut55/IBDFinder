import pandas as pd
from biom import load_table


def biom2ml():
    '''
    after original biom file collapsed to genus level,
    split the data into diseased and normal group for machine learning
    :return:
    '''
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

    ibd_abundance_table = load_table('data/otu_genus_table.biom')
    # normalise
    ibd_abundance_table = ibd_abundance_table.norm(axis='sample', inplace=False)
    # ibd group
    filter_ibd = lambda v, i_, m: [x for x in ibd if x == i_]
    ibd_group = ibd_abundance_table.filter(filter_ibd, axis='sample', inplace=False)
    df_ibd = ibd_group.to_dataframe(dense=True)
    # control group
    filter_control = lambda v, i_, m: [x for x in control if x == i_]
    control_group = ibd_abundance_table.filter(filter_control, axis='sample', inplace=False)
    df_control = control_group.to_dataframe(dense=True)

    labels = [1] * len(df_ibd.columns.values) + [0] * len(df_control.columns.values)

    results = pd.concat([df_ibd, df_control], axis=1).T
    # values with all zero for the variable are removed
    results = results.loc[:, (results != 0).any(axis=0)]

    results["label"] = labels
    # output for machine learning models
    results.to_csv('output/ibd_otus.txt', sep="\t")


if __name__ == "__main__":
    biom2ml()

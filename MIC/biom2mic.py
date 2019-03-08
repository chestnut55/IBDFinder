from biom import load_table


def biom2mic():
    '''
    biom file to MIC input :
    rows are variables, columns are samples
    :return:
    '''
    ibd_abundance_table = load_table('../data/otu_genus_table.biom')

    ibd_abundance_table = ibd_abundance_table.norm(axis='sample', inplace=False)
    ibd_abundance_table = ibd_abundance_table.to_dataframe(dense=True)
    ibd_abundance_table.to_csv('ibd_otus_mic.txt', sep="\t")


if __name__ == "__main__":
    biom2mic()

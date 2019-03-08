from biom import load_table


def biom2sparcc():
    '''
    biom file to sparcc input
    :return:
    '''
    ibd_abundance_table = load_table('../data/otu_genus_table.biom')

    ibd_abundance_table = ibd_abundance_table.to_dataframe(dense=True)
    ibd_abundance_table.to_csv('ibd_otus_sparcc.txt', sep="\t")


if __name__ == "__main__":
    biom2sparcc()

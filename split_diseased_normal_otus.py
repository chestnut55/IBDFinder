import pandas as pd


def split_diseased_normal():
    '''
    split the samples to diseased group and normal group
    :return:
    '''
    otus = pd.read_csv('output/ibd_otus.txt', sep="\t", index_col=0)

    diseased_group = otus[otus['label'] == 1].drop(['label'], axis=1).T
    normal_group = otus[otus['label'] == 0].drop(['label'], axis=1).T

    diseased_group.to_csv('output/diseased_otu.txt', sep="\t")
    normal_group.to_csv('output/normal_otu.txt', sep="\t")


if __name__ == "__main__":
    split_diseased_normal()

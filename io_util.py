import pandas as pd


def parse_CoNet():
    samples = pd.read_csv('data/coNet_id_file.txt',sep='\t')['SampleID'].values
    df = pd.read_csv('data/coNet_metadata_file.txt',sep='\t',index_col=0)
    df = df.ix[samples,:]
    df[df.diagnosis == 'no'] = 0
    df[df.diagnosis == 'IC'] = 1
    df[df.diagnosis == 'CD'] = 1
    df[df.diagnosis == 'UC'] = 1
    df.to_csv('data/coNet_metadata_file.txt',sep='\t')


def apply_column_filter(row):
    taxa = row[1]
    taxonomy = taxa.rfind(';')
    taxa = taxa[:taxonomy]
    return taxa


def parse_taxa():
    df = pd.read_csv('data/exported_out_feature_table/taxonomy.tsv',sep='\t')
    se = set(df.apply(apply_column_filter,axis=1).values)
    print(len(se))


if __name__ == "__main__":
    parse_taxa()
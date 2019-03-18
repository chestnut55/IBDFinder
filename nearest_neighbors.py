import pandas as pd
import numpy as np
import utils


def taxon_id_name():
    taxonomy = pd.read_csv('data/taxonomy.tsv', sep='\t')
    taxa_mapping_file = dict()
    for key, value in zip(taxonomy['Feature ID'].values, taxonomy['Taxon'].values):
        taxonomy = value.rfind(';')
        taxa = value[:taxonomy].replace(" ","")
        taxa_mapping_file[str(key)] = taxa

    return taxa_mapping_file


def apply_column_filter(row):
    taxa = row[1]
    taxonomy = taxa.rfind(';')
    taxa = taxa[:taxonomy]
    return taxa


def top_k_neighbors(k):
    taxa_mapping_file = taxon_id_name()

    df = pd.read_csv('output/tips_distances.csv', sep='\t', index_col=0)

    taxa_names = [taxa_mapping_file[name.strip()] for name in df.columns.values]

    results = pd.DataFrame(np.identity(len(taxa_names)),
                           columns=taxa_names, index=taxa_names, dtype=np.int)
    for col_id in df.columns.values:
        neighbors = df[col_id].sort_values()[1:k + 1].index.values
        column_name = taxa_mapping_file[col_id]
        for neighbor in neighbors:
            row_name = taxa_mapping_file[str(neighbor)]
            results[column_name][row_name] = 1
            results[row_name][column_name] = 1

    results.to_csv('output/phy_neighbors.csv',sep='\t')
    return results


if __name__ == "__main__":
    top_k_neighbors(10)

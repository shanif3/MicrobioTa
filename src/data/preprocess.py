import pandas as pd
import MIPMLP
import numpy as np
import samba


def clean_taxonomy(bact_name):
    """Remove taxonomy level prefixes from a name."""
    # Split the taxonomy string into levels
    levels = bact_name.split(";")

    # Remove known prefixes from each level
    cleaned_levels = [level.split("__")[-1] if "__" in level else level for level in levels]
    # remove empty levels
    cleaned_levels = [level for level in cleaned_levels if level]

    # Reconstruct the cleaned taxonomy string
    return ";".join(cleaned_levels)


def return_leaf_names(bacteria_names):
    """
    Builds a taxonomy matrix where each column represents a leaf bacterium
    and each row contains cumulative hierarchical levels.

    :param bacteria_names: List of bacterial names in "bacteria;phylum;...;species" format
    :return: Numpy array (8 rows x number of leaf bacteria)
    """
    max_levels = 8  # Maximum hierarchy levels
    processed_names = []

    full_names_set = set(bacteria_names)  # Set for fast lookup

    # Step 1: Identify non-leaf taxa (those that are prefixes of longer taxa)
    non_leaf_taxa = set()
    for name in bacteria_names:
        parts = name.split(";")
        for i in range(1, len(parts)):
            parent = ";".join(parts[:i])
            non_leaf_taxa.add(parent)

    # Step 2: Filter out non-leaf names (i.e. keep only real leaves)
    leaf_names = [name for name in bacteria_names if name not in non_leaf_taxa]

    return leaf_names



def preprocess_dataframe(df):
    """Preprocess a DataFrame with MIPMLP and filter taxonomy."""

    # Rename first column to 'ID'
    df = df.rename(columns={df.columns[0]: 'ID'})

    # Apply MIPMLP preprocessing
    df = MIPMLP.preprocess(df, normalization='log', taxnomy_group='mean')


    # TODO for gali norm
    # df = MIPMLP.preprocess(df, normalization='relative', taxnomy_group='mean')
    # ranks = df.rank(axis=1, method='min', ascending=False)
    #
    # # Compute the average rank for each row
    # avg_rank = ranks.mean(axis=1)
    #
    # # Compute (R_ij - R_i) / R_i
    # # normalized_ranks = (ranks.sub(avg_rank, axis=0)) / avg_rank.to_numpy()[:, None]
    # normalized_ranks = ranks / avg_rank.to_numpy()[:, None]
    #
    # # Apply log transformation
    # df = np.log(normalized_ranks)

    # Clean taxonomy names (use list comprehension instead of .apply())
    bacteria_names_filtered = [clean_taxonomy(col) for col in df.columns]
    df.columns = bacteria_names_filtered

    # Build taxonomy matrix
    # bact_names_leafs = return_leaf_names(bacteria_names_filtered)

    # Filter DataFrame to keep only columns corresponding to leaf bacteria
    # df = df[bact_names_leafs]
    # stay just with the leaf that are genus,specice or strain
    df = df.loc[:, (df.columns.str.split(';').map(len) == 6) | (df.columns.str.split(';').map(len) == 7)| (df.columns.str.split(';').map(len) == 8)]

    # drop columns with len 1
    # df = df.loc[:, df.columns.str.split(';').map(len) > 1]
    return df

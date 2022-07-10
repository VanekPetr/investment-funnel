import pandas as pd
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA


def minimum_spanning_tree(dataset):
    corr = dataset.corr(method="spearman")              # calculate the correlation
    distance_corr = (2 * (1 - corr)) ** 0.5             # calculate the distance
    mask = np.triu(np.ones_like(corr, dtype=np.bool))   # get only the upper half of the matrix
    distance_corr = distance_corr * mask 
   
    # use the correlation matrix to create links
    links = distance_corr.stack().reset_index(level=1)
    links.columns = ["var2", "value"]
    links = links.reset_index()
    links = links.replace(0, np.nan)                                # drop 0 values from the matrix
    links = links.dropna(how='any', axis=0)
    links.columns = ["var1", "var2", "value"]                       # rename the columns
    links_filtered = links.loc[(links["var1"] != links["var2"])]     # filter out self-correlations
    
    # Create the graph
    created_graph = nx.Graph()
    for i in range(len(corr)):                                         # add nodes
        created_graph.add_node(corr.index[i])
    tuples = list(links_filtered.itertuples(index=False, name=None))    # add edges with weight
    created_graph.add_weighted_edges_from(tuples)
    
    # Create a MST from the full graph
    mst = nx.minimum_spanning_tree(created_graph)
    
    # Save the nodes with degree one
    degrees = [val for (node, val) in mst.degree()]
    df = pd.DataFrame(degrees, corr.index)
    df.columns = ["degree"]
    subset = df[df["degree"] == 1].index.tolist()

    # Create a new dataframe with only the assets from the subset
    subset_df = dataset.loc[:, dataset.columns.isin(subset)]
    
    # Calculate the average correlation of the subset
    corr_subset = subset_df.corr(method="spearman")
    corr_avg = corr_subset.mean().mean()
    
    # Calculate the PDI for the subset
    pca = PCA()
    pca.fit(corr_subset)
    value = 0
    for i in range(1, corr_subset.shape[1]):
        value = value + i * pca.explained_variance_ratio_[i-1]
    pdi = 2 * value - 1

    return subset, subset_df, corr_avg, pdi

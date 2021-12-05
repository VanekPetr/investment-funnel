#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:38:17 2020

@author: Petr Vanek
"""

import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, fcluster, complete


# FUNCTION TO CREATE DENDOGRAM
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


# FUNCTION TO CLUSTER DATA
def Cluster(data, nClusters, dendogram):
    corr = data.corr(method="spearman")     # calculate the correlation
    distance_corr = 1-corr                  # distance based on correlation

    # Person corr distance matrix
    con_distance_corr = squareform(distance_corr)   # condence the distance matrix to be able to fit the hierarcal clustering
    complete_corr = complete(con_distance_corr)     # apply hierarchical clustering using the single distance measure
    
    if dendogram == True:
        # draw the dendogram
        plt.figure(figsize=(25, 10))
        fancy_dendrogram(
            complete_corr,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,
            color_threshold = 0.7,# font size for the x axis labels
            labels = distance_corr.index,
        #   max_d=0.35,
            annotate_above=10
        )
        plt.title('Hierarchical Clustering Dendrogram: Complete Linkage, Spearman Correlation Distance Mearsure', fontsize = 16)
        plt.xlabel('Assets', fontsize = 16)
        plt.ylabel('Distance', fontsize = 16)
        plt.show()
    
    # And now we want to save the clustering into a dataframe.
    # Create the dataframe
    cluster_df = pd.DataFrame(index=distance_corr.index)

    # Save the Complete_Corr clustering into the dataframe with 8 clusters
    cluster_df["Complete_Corr"] = fcluster(complete_corr, nClusters, criterion="maxclust")

    # Column for plotting
    for index in cluster_df.index:
        cluster_df.loc[index,"Cluster"]="Cluster "+str(cluster_df.loc[index, "Complete_Corr"])
    
    return cluster_df
 
    
# METHOD TO PICK ASSETS FROM A CLUSTER BASED ON PERFORMANCE CRITERIA
def pickCluster(data, stat, ML, nAssets):
    test = pd.concat([stat, ML], axis=1)
    # For each cluster find the asset with the highest Sharpe ratio
    ids = []
    for clus in test["Cluster"].unique():
        # number of elements in each cluster
        sizeMax = len(test[test["Cluster"]== str(clus)])
        # Get indexes
        if nAssets <= sizeMax:
            ids.extend(test[test["Cluster"]== str(clus)].nlargest(nAssets,
                            ["Sharpe Ratio"]).index)
        else:
            ids.extend(test[test["Cluster"]== str(clus)].nlargest(sizeMax,
                            ["Sharpe Ratio"]).index)
            print("In "+str(clus)+" was picked only", sizeMax,"Assets")

    # Get returns
    result = data[ids]
    
    return ids, result
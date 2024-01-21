import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from scipy.cluster.hierarchy import complete, dendrogram, fcluster
from scipy.spatial.distance import squareform


def fancy_dendrogram(*args, **kwargs):
    """
    FUNCTION TO CREATE DENDROGRAM
    """
    max_d = kwargs.pop("max_d", None)
    if max_d and "color_threshold" not in kwargs:
        kwargs["color_threshold"] = max_d
    annotate_above = kwargs.pop("annotate_above", 0)

    d_data = dendrogram(*args, **kwargs)

    if not kwargs.get("no_plot", False):
        plt.title("Hierarchical Clustering Dendrogram (truncated)")
        plt.xlabel("sample index or (cluster size)")
        plt.ylabel("distance")
        for i, d, c in zip(d_data["icoord"], d_data["dcoord"], d_data["color_list"]):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, "o", c=c)
                plt.annotate(
                    "%.3g" % y,
                    (x, y),
                    xytext=(0, -5),
                    textcoords="offset points",
                    va="top",
                    ha="center",
                )
        if max_d:
            plt.axhline(y=max_d, c="k")
    return d_data


def cluster(
    data: pd.DataFrame, n_clusters: int, dendrogram: bool = False
) -> pd.DataFrame:
    """
    FUNCTION TO CLUSTER DATA
    """
    logger.info("💡 Running hierarchical clustering method")

    corr = data.corr(method="spearman")  # calculate the correlation
    distance_corr = 1 - corr  # distance based on correlation

    # Person corr distance matrix
    con_distance_corr = squareform(
        distance_corr
    )  # the distance matrix to be able to fit the hierarchical clustering
    complete_corr = complete(
        con_distance_corr
    )  # apply hierarchical clustering using the single distance measure

    if dendrogram:
        # draw the dendrogram
        plt.figure(figsize=(25, 10))
        fancy_dendrogram(
            complete_corr,
            leaf_rotation=90.0,  # rotates the x-axis labels
            leaf_font_size=8.0,
            color_threshold=0.7,  # font size for the x-axis labels
            labels=distance_corr.index,
            # max_d=0.35,
            annotate_above=10,
        )
        plt.title(
            "Hierarchical Clustering Dendrogram: Complete Linkage, Spearman Correlation Distance Mearsure",
            fontsize=16,
        )
        plt.xlabel("Assets", fontsize=16)
        plt.ylabel("Distance", fontsize=16)
        plt.show()

    # And now we want to save the clustering into a dataframe.
    # Create the dataframe
    cluster_df = pd.DataFrame(index=distance_corr.index)

    # Save the Complete_Corr clustering into the dataframe with 8 clusters
    cluster_df["Complete_Corr"] = fcluster(
        complete_corr, n_clusters, criterion="maxclust"
    )

    # Column for plotting
    for index in cluster_df.index:
        cluster_df.loc[index, "Cluster"] = "Cluster " + str(
            cluster_df.loc[index, "Complete_Corr"]
        )

    return cluster_df


def pick_cluster(
    data: pd.DataFrame, stat: pd.DataFrame, ml: pd.DataFrame, n_assets: int
) -> (list, pd.DataFrame):
    """
    METHOD TO PICK ASSETS FROM A CLUSTER BASED ON PERFORMANCE CRITERIA
    """
    test = pd.concat([stat, ml], axis=1)
    # For each cluster find the asset with the highest Sharpe ratio
    ids = []
    for clus in test["Cluster"].unique():
        # number of elements in each cluster
        max_size = len(test[test["Cluster"] == str(clus)])
        # Get indexes
        if n_assets <= max_size:
            ids.extend(
                test[test["Cluster"] == str(clus)]
                .nlargest(n_assets, ["Sharpe Ratio"])
                .index
            )
        else:
            ids.extend(
                test[test["Cluster"] == str(clus)]
                .nlargest(max_size, ["Sharpe Ratio"])
                .index
            )
            logger.warning(f"⚠️ In {clus} was picked only {max_size} assets")

    # Get returns
    result = data[ids]

    return ids, result

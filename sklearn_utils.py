import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans

#  abs_change(array):
#  auto_kmeans(df, max_clusters=20, excluded_cols=None, plot_scores=False):
#  auto_kmeans_v2(df, max_clusters=20, excluded_cols=None, plot_scores=False, histograms=False):
#  auto_kmeans_v3(df, number_of_clusters=None, max_clusters=20, excluded_cols=None, plot_scores=False, histograms=False):
#  autoselect_cluster_number(array, index_limit=5):
#  find_elbow_index(array, index_limit=5):
#  pct_change(array):
#  visualize_dataframe_columns(df):



def abs_change(array):

    if len(array) < 2:
        return -1

    return array[1:] - array[:-1]


def pct_change(array):

    if len(array) < 2:
        return -1

    return array[1:] / array[:-1] - 1


def find_elbow_index(array, index_limit=5):

    if len(array) < 3:
        return -1

    ds = abs_change(array)
    dds = pct_change(ds)
    dds_argsort = np.argsort(dds)

    for i in range(dds_argsort.size):
        if dds_argsort[i] <= index_limit:
            index = dds_argsort[i] + 1
            break    

    return index


def autoselect_cluster_number(array, index_limit=5):

    if len(array) < 3:
        return -1

    ds = abs_change(array)
    dds = pct_change(ds)
    dds_argsort = np.argsort(dds)

    for i in range(dds_argsort.size):
        if dds_argsort[i] <= index_limit:
            index = dds_argsort[i] + 1
            break    

    return index + 1



def auto_kmeans(df, max_clusters=20, excluded_cols=None, plot_scores=False):
    df_seg = df.copy()
    if excluded_cols:
        df_seg = df.drop(columns=excluded_cols)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_seg)

    scores = []

    for i in range(1, max_clusters):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(scaled_data)
        scores.append(kmeans.inertia_)

    scores = np.array(scores)
    elbow_index = find_elbow_index(scores, index_limit=max_clusters)
    number_of_clusters = elbow_index + 1

    if plot_scores:
        indices = np.arange(1, scores.size + 1)
        plt.plot(indices, scores, 'bx-', np.array([number_of_clusters]), np.array([scores[number_of_clusters]]), 'r^')
        plt.title('Sum of Euclidian Distances vs. Number of Clusters')
        plt.xlabel('Clusters')
        plt.ylabel('Sum of Euclidian Distances')
        plt.show()

    kmeans = KMeans(number_of_clusters)
    kmeans.fit(scaled_data)

    y_kmeans = kmeans.fit_predict(scaled_data)

    scaled_cluster_centers_df = pd.DataFrame(data=kmeans.cluster_centers_, columns=[df_seg.columns])
    cluster_centers_array = scaler.inverse_transform(scaled_cluster_centers_df)
    cluster_centers_df = pd.DataFrame(data=cluster_centers_array, columns=[df_seg.columns])
    cluster_centers_df.where(cluster_centers_df > 0.0001, 0.0, inplace=True)
    df_seg = pd.concat([df_seg, pd.DataFrame({'cluster': kmeans.labels_})], axis=1)

    return df_seg, cluster_centers_df

    # a, b = auto_kmeans(steels_df, max_clusters=8, excluded_cols=['Steel', 'Type', 'Rw'], plot_scores=True)


def auto_kmeans_v2(df, max_clusters=20, excluded_cols=None, plot_scores=False, histograms=False):

    
    df_seg = df.copy()

    if excluded_cols:
        df_seg = df.drop(columns=excluded_cols)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_seg)

    scores = []

    for i in range(1, max_clusters):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(scaled_data)
        scores.append(kmeans.inertia_)

    scores = np.array(scores)
    elbow_index = find_elbow_index(scores, index_limit=max_clusters)
    number_of_clusters = elbow_index + 1

    if plot_scores:
        indices = np.arange(1, scores.size + 1)
        plt.plot(indices, scores, 'bx-', np.array([number_of_clusters]), np.array([scores[number_of_clusters]]), 'r^')
        plt.title('Sum of Euclidian Distances vs. Number of Clusters')
        plt.xlabel('Clusters')
        plt.ylabel('Sum of Euclidian Distances')
        plt.show()

    kmeans = KMeans(number_of_clusters)
    kmeans.fit(scaled_data)

    y_kmeans = kmeans.fit_predict(scaled_data)

    scaled_cluster_centers_df = pd.DataFrame(data=kmeans.cluster_centers_, columns=[df_seg.columns])
    cluster_centers_array = scaler.inverse_transform(scaled_cluster_centers_df)
    cluster_centers_df = pd.DataFrame(data=cluster_centers_array, columns=[df_seg.columns])
    cluster_centers_df.where(cluster_centers_df > 0.0001, 0.0, inplace=True)
    df_cluster = pd.concat([df, pd.DataFrame({'cluster': kmeans.labels_})], axis=1)

    distances_to_centers = distance.cdist(df_seg.to_numpy(), cluster_centers_df.to_numpy(), metric='euclidean')

    dist_col_names = [i + j for i, j in zip(['d']*number_of_clusters, map(str, range(number_of_clusters)))]				#most readable??
    #dist_cols = np.char.add('d', np.vectorize(str)(np.arange(number_of_clusters))).tolist()							#also works
    #dist_col_names = [i + j for i, j in zip(['d']*number_of_clusters, [str(x) for x in range(number_of_clusters)])]	#also works

    df_cluster = pd.concat([df_cluster, pd.DataFrame(distances_to_centers, columns=dist_col_names)], axis=1)
	        # Plot the histogram of various clusters
	
	if histograms:
		for i in df_seg.columns:
			plt.figure(figsize = (35, 10))
			for j in range(number_of_clusters):
				plt.subplot(1,number_of_clusters,j+1)
				cluster = df_cluster[df_cluster['cluster'] == j]
				cluster[i].hist(bins = 20)
				plt.title('{}    \nCluster {} '.format(i, j))
		  
		plt.show()

    return df_cluster, cluster_centers_df

    # a, b = auto_kmeans(steels_df, max_clusters=8, excluded_cols=['Steel', 'Type', 'Rw'], plot_scores=True)


def auto_kmeans_v3(df, number_of_clusters=None, max_clusters=20, excluded_cols=None, plot_scores=False, histograms=False):
''' added option to specify number of clusters '''
    
    df_seg = df.copy()

    if excluded_cols:
        df_seg = df.drop(columns=excluded_cols)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_seg)

    if not number_of_clusters:
	    scores = []

	    for i in range(1, max_clusters):
	        kmeans = KMeans(n_clusters=i)
	        kmeans.fit(scaled_data)
	        scores.append(kmeans.inertia_)

	    scores = np.array(scores)
	    elbow_index = find_elbow_index(scores, index_limit=max_clusters)
	    number_of_clusters = elbow_index + 1

	    if plot_scores:
	        indices = np.arange(1, scores.size + 1)
	        plt.plot(indices, scores, 'bx-', np.array([number_of_clusters]), np.array([scores[number_of_clusters]]), 'r^')
	        plt.title('Sum of Euclidian Distances vs. Number of Clusters')
	        plt.xlabel('Clusters')
	        plt.ylabel('Sum of Euclidian Distances')
	        plt.show()

    kmeans = KMeans(number_of_clusters)
    kmeans.fit(scaled_data)

    y_kmeans = kmeans.fit_predict(scaled_data)

    scaled_cluster_centers_df = pd.DataFrame(data=kmeans.cluster_centers_, columns=[df_seg.columns])
    cluster_centers_array = scaler.inverse_transform(scaled_cluster_centers_df)
    cluster_centers_df = pd.DataFrame(data=cluster_centers_array, columns=[df_seg.columns])
    cluster_centers_df.where(cluster_centers_df > 0.0000001, 0.0, inplace=True)
    df_cluster = pd.concat([df, pd.DataFrame({'cluster': kmeans.labels_})], axis=1)

    distances_to_centers = distance.cdist(df_seg.to_numpy(), cluster_centers_df.to_numpy(), metric='euclidean')

    dist_col_names = [i + j for i, j in zip(['d']*number_of_clusters, map(str, range(number_of_clusters)))]				#most readable??
    #dist_cols = np.char.add('d', np.vectorize(str)(np.arange(number_of_clusters))).tolist()							#also works
    #dist_col_names = [i + j for i, j in zip(['d']*number_of_clusters, [str(x) for x in range(number_of_clusters)])]	#also works

    df_distances = pd.DataFrame(distances_to_centers, columns=dist_col_names)
    df_distances.where(df_distances > 0.000001, 0.0, inplace=True)
    df_cluster = pd.concat([df_cluster, df_distances], axis=1)
	        # Plot the histogram of various clusters
	
	if histograms:
		for i in df_seg.columns:
			plt.figure(figsize = (35, 10))
			for j in range(number_of_clusters):
				plt.subplot(1,number_of_clusters,j+1)
				cluster = df_cluster[df_cluster['cluster'] == j]
				cluster[i].hist(bins = 20)
				plt.title('{}    \nCluster {} '.format(i, j))
		  
		plt.show()

    return df_cluster, cluster_centers_df

    # a, b = auto_kmeans(steels_df, max_clusters=8, excluded_cols=['Steel', 'Type', 'Rw'], plot_scores=True)


def visualize_dataframe_columns(df):
    # distplot combines the matplotlib.hist function with seaborn kdeplot()
    columns = df.describe().columns.to_list()
    N = len(columns)
    plt.figure(figsize = (20, 40))
    for i in range(N):
        plt.subplot(N, 4, i+1)
        sns.distplot(df[columns[i]], color = 'r', kde_kws = {'bw' :1.0})  # larger bandwidth ('bw') -> smoother looking distribution [[NOTE: watch out for impossible negative values]]
        plt.title(columns[i])

    plt.tight_layout()

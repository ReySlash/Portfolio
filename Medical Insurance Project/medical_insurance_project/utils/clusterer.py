import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors
from itertools import product

def plot_explained_variance(df):
    pca = PCA(random_state=42)
    X_reduced = pca.fit_transform(df)
    cumulative_exp_var = pca.explained_variance_ratio_.cumsum()*100
    sns.lineplot(
        x=list(range(1,df.shape[1]+1)),
        y=cumulative_exp_var,
        marker='o',
        color="r"
    )
    plt.yticks(list(range(5,101,5)))
    plt.xticks(list(range(1,df.shape[1]+1)))
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid()
    
def pca_kmeans_elbow_study(df,pca_n_components_range, n_clusters_range):
    # Define the range of PCA components and clusters
    pca_n_components_range = list(range(10, 16))

    # Loop over the number of PCA components
    for n in pca_n_components_range:
        pca = PCA(n_components=n, random_state=42)
        X_pca = pca.fit_transform(df)
        
        # Store inertia and silhouette scores for each number of clusters
        inertia_values = []
        silhouette_scores = []  # Initialize silhouette scores within this loop

        # Loop over the number of clusters
        for k in n_clusters_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_pca)
            
            # Append inertia and silhouette score
            inertia_values.append(kmeans.inertia_)
            score = silhouette_score(X_pca, kmeans.labels_)
            silhouette_scores.append(score)

        # Plot the inertia (Elbow method) and silhouette score for each k
        plt.figure(figsize=(12, 5))
        
        # Elbow Method Plot
        plt.subplot(1, 2, 1)
        plt.plot(n_clusters_range, inertia_values, marker='o')
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.title(f"Elbow Method for Optimal k (PCA components = {n})")
        plt.xticks(n_clusters_range)
        plt.grid()
        
        # Silhouette Score Plot
        plt.subplot(1, 2, 2)
        plt.plot(n_clusters_range, silhouette_scores, marker='o', color='orange')
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.title(f"Silhouette Score vs. k (PCA components = {n})")
        plt.xticks(n_clusters_range)
        plt.grid()
        
        plt.tight_layout()
        plt.show()


def pca_kmeans_study(df, pca_n_components_range = None, n_clusters_range=0):
    results = []
    if pca_n_components_range is None:
        # Nested Loop over number of clusters
        for n_clusters in n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # Initialize KMeans model
            kmeans.fit(df)                                     # Only fit the KMeans model
            
            inertia = kmeans.inertia_                                # Get inertia for the current clustering
            silhouette_avg = silhouette_score(df, kmeans.labels_)  # Compute silhouette score
            
            # Append results into a dictionary for each combination of n_components and n_clusters
            results.append({
                'n_clusters': n_clusters, 
                'inertia': inertia,
                'silhouette_score': silhouette_avg,
            })

        return pd.DataFrame(results)
    
    else:
        # Main Loop over number of components
        for n_components in pca_n_components_range:
            pca = PCA(n_components=n_components, random_state=42)  # Initialize PCA with the current number of components
            data_pca = pca.fit_transform(df)                       # Apply PCA transformation on data
            explained_variance_ratio = pca.explained_variance_ratio_[-1]    # Explained variance for the last component
            cumulative_explained_variance_ratio = pca.explained_variance_ratio_.cumsum()[-1]  # Cumulative variance
            
            # Nested Loop over number of clusters
            for n_clusters in n_clusters_range:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # Initialize KMeans model
                kmeans.fit(data_pca)                                     # Only fit the KMeans model
                
                inertia = kmeans.inertia_                                # Get inertia for the current clustering
                silhouette_avg = silhouette_score(data_pca, kmeans.labels_)  # Compute silhouette score
                
                # Append results into a dictionary for each combination of n_components and n_clusters
                results.append({
                    'n_components': n_components,
                    'n_clusters': n_clusters, 
                    'explained_variance': explained_variance_ratio,
                    'cumulative_explained_variance_ratio': cumulative_explained_variance_ratio,
                    'silhouette_score': silhouette_avg,
                    'inertia': inertia
                })
    
                
        results_df = pd.DataFrame(results)
        
        # Copy of the original results DataFrame for scaling purposes
        tests_df = results_df.copy()

        # Scaling the metrics for comparison
        scaler = MinMaxScaler()
        cols = ['scaled_n_components', 'scaled_n_clusters', 'scaled_explained_variance', 
                'scaled_cumulative_explained_variance_ratio', 'scaled_silhouette_score', 'scaled_inertia']
        
        # Scale metrics and create a new DataFrame with the scaled results
        scaled_results = pd.DataFrame(scaler.fit_transform(tests_df), columns=cols)

        # Create a score for explained variance
        scaled_results['exp_var_Score'] = (scaled_results['scaled_explained_variance'] + 
                                        scaled_results['scaled_cumulative_explained_variance_ratio']) / 2

        # Create a score for silhouette coefficient and number of clusters
        scaled_results['silhouette_n_clust_score'] = (scaled_results['scaled_silhouette_score'] + 
                                                    (1 - scaled_results['scaled_n_clusters'])) / 2

        # Combine both scores to create a total score
        scaled_results['total_score'] = (scaled_results['exp_var_Score'] + 
                                        scaled_results['silhouette_n_clust_score']) / 2

        # Merge the original results with the scaled results
        df_merged = pd.merge(results_df, scaled_results, on=results_df.index, how='inner').drop(columns=['key_0'])

        # Sort by the total score in descending order
        df_merged_sorted = df_merged.sort_values(by='total_score', ascending=False)

        return df_merged_sorted

def hierarchical_dendogram_study(df,pca_n_components_range): 
    # Apply PCA
    for n_components in pca_n_components_range:
        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=n_components,random_state=42)  # Adjust the number of components as needed
        pca_result = pca.fit_transform(df)
        # print(f"N_components: {n_components}")

        # Generate the linkage matrix
        
    #     linkage_matrix = linkage(pca_result, method='ward')

        # Plot the dendrogram
        plt.figure(figsize=(12, 8))
        dendrogram_plot = dendrogram(linkage(pca_result, method='ward'))
        plt.title(f'Dendrogram for Hierarchical Clustering, n_components: {n_components}')
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        plt.show()

def hcl_study(df,pca_n_components_range, n_clusters_range):

    # Initialize a list to store the results
    results = []

    for n_components in pca_n_components_range:
        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=n_components, random_state=42)  # Adjust the number of components as needed
        pca_result = pca.fit_transform(df)

        for n_clusters in n_clusters_range:
            # Perform Agglomerative Clustering
            agg_clust = AgglomerativeClustering(n_clusters=n_clusters)
            agg_labels = agg_clust.fit_predict(pca_result)  # Use pca_result instead of X_pca

            # Calculate the silhouette score
            silhouette_avg = silhouette_score(pca_result, agg_labels)

            # Append results to the list
            results.append({
                'n_components': n_components,
                'n_clusters': n_clusters,
                'silhouette_score': silhouette_avg
            })

    # Convert results to a DataFrame
    return pd.DataFrame(results)

def knee_study(df, max_n_component):
    
    pca = PCA(n_components=max_n_component, random_state=42)
    X_pca = pca.fit_transform(df)
                            
    neighbors = NearestNeighbors(n_neighbors=2)             # Using NearestNeighbors to find the optimal epsilon value
    neighbors_fit = neighbors.fit(X_pca)
    distances, indexes = neighbors_fit.kneighbors(X_pca)
    
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    fig = plt.figure(figsize=(10,10))
    plt.plot(distances)
    plt.title(f"N_components = {max_n_component}")
    plt.show()

def dbscan_study(df,eps_values,min_samples,pca_n_components_range=None):
    # Apply PCA
    results_list = []  # Initialize the list once, outside the loops

    
    if pca_n_components_range:
        for n_component in pca_n_components_range:
            pca = PCA(n_components=n_component, random_state=42)
            X_pca = pca.fit_transform(df)
            
            # Generate all combinations of parameters
            dbscan_params = list(product(eps_values, min_samples))

            # Loop through each combination of eps and min_samples
            for eps, min_s in dbscan_params:
                # Fit DBSCAN
                db = DBSCAN(eps=eps, min_samples=min_s)
                labels = db.fit_predict(X_pca)

                # Number of clusters in labels, ignoring noise if present
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise_ = list(labels).count(-1)
                not_noise_ = len(labels) - n_noise_

                # Calculate silhouette score, ignoring noise points
                if n_clusters_ > 1:  # Silhouette score is not defined for a single cluster
                    sil_score = silhouette_score(X_pca[labels != -1], labels[labels != -1])
                else:
                    sil_score = None  # Use None for clarity

                # Append the results to the list (inside the inner loop)
                results_list.append({
                    'eps': eps,
                    'min_samples': min_s,
                    'n_clusters': n_clusters_,
                    'n_noise': n_noise_,
                    'not_noise': not_noise_,
                    'silhouette_score': sil_score,
                    'n_components': n_component
                })
    else:
        # Generate all combinations of parameters
        dbscan_params = list(product(eps_values, min_samples))
        # Loop through each combination of eps and min_samples
        for eps, min_s in dbscan_params:
            # Fit DBSCAN
            db = DBSCAN(eps=eps, min_samples=min_s)
            labels = db.fit_predict(df)

            # Number of clusters in labels, ignoring noise if present
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            not_noise_ = len(labels) - n_noise_

            # Calculate silhouette score, ignoring noise points
            # if n_clusters_ > 1:  # Silhouette score is not defined for a single cluster
            #     sil_score = silhouette_score(df[labels != -1], labels[labels != -1])
            # else:
            #     sil_score = None  # Use None for clarity

            valid_mask = labels != -1
            if valid_mask.sum() > 0:  # Asegurarse de que no todos sean ruido
                sil_score = silhouette_score(df[valid_mask], labels[valid_mask])
            else:
                sil_score = None

            # Append the results to the list (inside the inner loop)
            results_list.append({
                'eps': eps,
                'min_samples': min_s,
                'n_clusters': n_clusters_,
                'n_noise': n_noise_,
                'not_noise': not_noise_,
                'silhouette_score': sil_score,
            })

    # Convert the results list to a DataFrame
    return pd.DataFrame(results_list)
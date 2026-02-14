#Spotify hidden similarities Trend through clustering﻿
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score

file_path = "dataset.csv"
df = pd.read_csv(file_path)

required_columns = ['danceability', 'loudness', 'valence', 'energy', 'tempo', 'track_genre']
if all(col in df.columns for col in required_columns):
    df = df.dropna(subset=required_columns)
    correlation_min = 0.24
    correlation_max = 0.77
    genre_groups = df.groupby('track_genre')
    relevant_genres = []

    for genre, group in genre_groups:
        if len(group) > 1:
            corr_matrix = group[['danceability', 'loudness', 'valence']].corr()
            danceability_loudness_corr = corr_matrix.loc['danceability', 'loudness']
            danceability_valence_corr = corr_matrix.loc['danceability', 'valence']
            if (correlation_min <= abs(danceability_loudness_corr) <= correlation_max or
                correlation_min <= abs(danceability_valence_corr) <= correlation_max):
                relevant_genres.append(genre)

    filtered_df = df[df['track_genre'].isin(relevant_genres)].copy()
    print(f"Relevant Genres (correlation {correlation_min} to {correlation_max}): {relevant_genres}")
    print(f"Filtered Dataset Shape: {filtered_df.shape}")


    feature_options = ['danceability', 'loudness', 'valence', 'energy', 'tempo']
    print(", ".join(feature_options))
    feature_x = input("Enter the first feature: ").strip().lower()
    feature_y = input("Enter the second feature: ").strip().lower()

    x = filtered_df[feature_x]
    y = filtered_df[feature_y]

    genres = filtered_df['track_genre']

    # this creates the scatterplot
    plt.figure(figsize=(12, 11))
    # .cat.codes used to convert to categorical data with integers
    scatter = plt.scatter(x, y, c = genres.astype('category').cat.codes, cmap = 'tab10', alpha = 0.7)

    # colorbar with genre labels
    cbar = plt.colorbar(scatter, label = 'Genre')
    tick_locs = genres.astype('category').cat.codes.unique()
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(genres.astype('category').cat.categories)

    # creates labels, title, and grid for the data
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title('Spotify Tracks: {feature_x} vs {feature_y} by Relevant Genres')
    plt.grid(True)

    # Show plot
    plt.show()

    print("K-means Clustering")
    print(", ".join(feature_options))
    feature_x = input("Enter the first feature: ").strip().lower()
    feature_y = input("Enter the second feature: ").strip().lower()

    if feature_x not in feature_options or feature_y not in feature_options:
        print("Invalid features selected.")
    else:
        X = filtered_df[[feature_x, feature_y]].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        inertia = []

        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, n_init='auto', random_state=42)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, 11), inertia, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.grid(True)
        plt.show()

        n_clusters = int(input("Enter optimal number of clusters: "))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        cluster_centers = kmeans.cluster_centers_
        filtered_df.loc[:, 'Cluster'] = labels
        true_labels = filtered_df['track_genre'].astype('category').cat.codes
        ari = adjusted_rand_score(true_labels, labels)
        dbs = davies_bouldin_score(X_scaled, labels)
        mis = normalized_mutual_info_score(true_labels, labels)
        sis = silhouette_score(X_scaled, labels)

        print(f"Adjusted Rand Index: {ari}")
        print(f"Davies-Bouldin Score: {dbs}")
        print(f"Mutual Information Score: {mis}")
        print(f"Silhouette Score: {sis}")

        cluster_genre_mapping = {}
        for cluster_label in np.unique(labels):
            genres_in_cluster = filtered_df[filtered_df['Cluster'] == cluster_label]['track_genre'].value_counts()
            cluster_genre_mapping[cluster_label] = genres_in_cluster.index.tolist()

        plt.figure(figsize=(12, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(labels))))
        for label, color in zip(np.unique(labels), colors):
            cluster_data = X[labels == label]
            plt.scatter(
                cluster_data[:, 0], cluster_data[:, 1],
                c=[color], s=100, edgecolor='k', alpha=0.7, label=f"Cluster {label}"
            )
        cluster_centers_unscaled = scaler.inverse_transform(cluster_centers)
        plt.scatter(
            cluster_centers_unscaled[:, 0], cluster_centers_unscaled[:, 1],
            c='red', s=300, marker='X', edgecolor='k', label='Centers'
        )
        plt.title(f'{feature_x.capitalize()} vs {feature_y.capitalize()}')
        plt.xlabel(feature_x.capitalize())
        plt.ylabel(feature_y.capitalize())
        plt.legend()
        plt.grid(True)
        plt.show()

        feature_options = ['danceability', 'loudness', 'valence', 'energy', 'tempo']
        print("Suggested Features:")
        print(", ".join(feature_options))

        print("\nChoose two features for Mean Shift clustering:")
        feature_x = input("Enter the first feature: ").strip().lower()
        feature_y = input("Enter the second feature: ").strip().lower()

        feature_x = next((f for f in feature_options if f.lower() == feature_x), None)
        feature_y = next((f for f in feature_options if f.lower() == feature_y), None)

        if not feature_x or not feature_y:
            print("Invalid features selected. Please choose valid features.")
        else:
            X = filtered_df[[feature_x, feature_y]].values
            X_scaled = scaler.fit_transform(X)

            quant = float(input("Enter quantile: ").strip().lower())
            samples = int(input("Enter sample number: ").strip().lower())

            bandwidth = estimate_bandwidth(X_scaled, quantile=quant, n_samples=samples)
            mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)

            labels = mean_shift.fit_predict(X_scaled)
            cluster_centers = mean_shift.cluster_centers_

            filtered_df['Cluster'] = labels

            print("\nGenres in each cluster:")
            for cluster_label in np.unique(labels):
                genres_in_cluster = filtered_df[filtered_df['Cluster'] == cluster_label]['track_genre'].unique()
                print(f"Cluster {cluster_label}: {genres_in_cluster}")

            true_labels = filtered_df['track_genre'].astype('category').cat.codes
            ari = adjusted_rand_score(true_labels, labels)
            mis = normalized_mutual_info_score(true_labels, labels)

            print(f"\nAdjusted Rand Index for {feature_x} vs {feature_y}: {ari}")
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1:
                dbs = davies_bouldin_score(X_scaled, labels)
                print(f"Davies-Bouldin Score for {feature_x} vs {feature_y}: {dbs}") 
                sis = silhouette_score(X_scaled, labels)
                print(f"Silhouette Score for {feature_x} vs {feature_y}: {sis}")
            print(f"Mutual Information Score for {feature_x} vs {feature_y}: {mis}")
           

            cluster_genre_mapping = {}
            for cluster_label in np.unique(labels):
                genres_in_cluster = filtered_df[filtered_df['Cluster'] == cluster_label]['track_genre'].value_counts()
                cluster_genre_mapping[cluster_label] = genres_in_cluster.index.tolist()

            plt.figure(figsize=(12, 8))
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                cluster_data = X[labels == label]
                plt.scatter(
                    cluster_data[:, 0], cluster_data[:, 1],
                    c=[color], s=100, edgecolor='k', alpha=0.7, label=f"Cluster {label}"
                )

            cluster_centers_unscaled = scaler.inverse_transform(cluster_centers)
            plt.scatter(
                cluster_centers_unscaled[:, 0], cluster_centers_unscaled[:, 1],
                c='red', s=300, marker='X', edgecolor='k', label='Cluster Centers'
            )
            plt.title(f'MeanShift Clustering: {feature_x.capitalize()} vs {feature_y.capitalize()}')
            plt.xlabel(feature_x.capitalize())
            plt.ylabel(feature_y.capitalize())
            plt.legend()
            plt.grid(True)
            plt.show()

            print("K-means clustering by all numeric features:")
            numeric_features = df.select_dtypes(include=[np.number]).columns
            features = df[numeric_features].dropna()

            scaled_features = scaler.fit_transform(features)

            inertia = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, n_init='auto', random_state=42)
                kmeans.fit(scaled_features)
                inertia.append(kmeans.inertia_)

            plt.figure(figsize=(8, 6))
            plt.plot(range(1, 11), inertia, marker='o')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Inertia')
            plt.title('Elbow Method')
            plt.grid(True)
            plt.show()

            n_clusters = int(input("Enter optimal number of clusters: "))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(scaled_features)
            cluster_centers = kmeans.cluster_centers_
            filtered_df = df.loc[features.index].copy()
            filtered_df['Cluster'] = labels

            true_labels = filtered_df['track_genre'].astype('category').cat.codes
            ari = adjusted_rand_score(true_labels, labels)
            dbs = davies_bouldin_score(scaled_features, labels)
            mis = normalized_mutual_info_score(true_labels, labels)
            sis = silhouette_score(scaled_features, labels)

            print(f"Adjusted Rand Index: {ari}")
            print(f"Davies-Bouldin Score: {dbs}")
            print(f"Mutual Information Score: {mis}")
            print(f"Silhouette Score: {sis}")

            pca = PCA(n_components=2)
            pca_features = pca.fit_transform(scaled_features)

            plt.figure(figsize=(12, 8))
            colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(labels))))
            for label, color in zip(np.unique(labels), colors):
                cluster_data = pca_features[labels == label]
                plt.scatter(
                    cluster_data[:, 0], cluster_data[:, 1],
                    c=[color], s=100, edgecolor='k', alpha=0.7, label=f"Cluster {label}"
                )
            cluster_centers_pca = pca.transform(cluster_centers)
            plt.scatter(
                cluster_centers_pca[:, 0], cluster_centers_pca[:, 1],
                c='red', s=300, marker='X', edgecolor='k', label='Centers'
            )
            plt.title('PCA Features with Clusters')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.legend()
            plt.grid(True)
            plt.show()
else:
    print("Dataset not loaded.")

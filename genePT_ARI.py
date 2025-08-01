import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

def load_data(file):
    """
    Loads in pandas dataframe from a csv file
    """
    df = pd.read_csv(file, sep=',')
    df = df.set_index('Unnamed: 0')
    df.index.name = None
    return df


def get_ARI(file_path, num_cells):
    """
    Get ARI values for K-Means and GMM using PCA and UMAP
    """
    cell_df = load_data(file_path)
    y = [cell.split('.')[0] for cell in cell_df.T.index]
    X_train, X_test, y_train, y_test = train_test_split(cell_df.T.values, y, test_size=0.2, stratify=y, random_state=42)

    assert len(np.unique(y_train)) == num_cells
    assert len(np.unique(y_test)) == num_cells

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    pca = PCA()
    pca.fit(X_train_scaled)

    explained_variance = pca.explained_variance_ratio_

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
    plt.title('Explained Variance by PCA Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance')
    plt.grid(True)
    plt.show()

    n_components_lst = [10, 25, 50, 75, 100, 200, 500]
    train_kmeans_ari_score_pca_lst = []
    test_kmeans_ari_score_pca_lst = []
    train_kmeans_ari_score_umap_lst = []
    test_kmeans_ari_score_umap_lst = []

    train_gmm_ari_score_pca_lst = []
    test_gmm_ari_score_pca_lst = []
    train_gmm_ari_score_umap_lst = []
    test_gmm_ari_score_umap_lst = []

    X_test_scaled = scaler.transform(X_test)
    num_unique_cells = len(np.unique(y))

    for n_components in tqdm(n_components_lst):
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X_train_scaled)

        X_train_pca = pca.transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        X_umap = umap.UMAP(min_dist=0.5, spread=1, random_state=42)
        X_umap.fit(X_train_pca)

        X_train_umap = X_umap.transform(X_train_pca)
        X_test_umap = X_umap.transform(X_test_pca)

        kmeans_pca = KMeans(n_clusters=num_unique_cells, random_state=42)
        kmeans_umap = KMeans(n_clusters=num_unique_cells, random_state=42)

        kmeans_pca.fit(X_train_pca)
        kmeans_umap.fit(X_train_umap)

        kmeans_pca_y_train_pred = kmeans_pca.predict(X_train_pca)
        kmeans_pca_y_test_pred = kmeans_pca.predict(X_test_pca)
        kmeans_umap_y_train_pred= kmeans_umap.predict(X_train_umap)
        kmeans_umap_y_test_pred= kmeans_umap.predict(X_test_umap)

        kmeans_pca_train_ari_score = adjusted_rand_score(y_train, kmeans_pca_y_train_pred)
        kmeans_pca_test_ari_score = adjusted_rand_score(y_test, kmeans_pca_y_test_pred)
        kmeans_umap_train_ari_score = adjusted_rand_score(y_train, kmeans_umap_y_train_pred)
        kmeans_umap_test_ari_score = adjusted_rand_score(y_test, kmeans_umap_y_test_pred)

        train_kmeans_ari_score_pca_lst.append(kmeans_pca_train_ari_score)
        train_kmeans_ari_score_umap_lst.append(kmeans_umap_train_ari_score)
        test_kmeans_ari_score_pca_lst.append(kmeans_pca_test_ari_score)
        test_kmeans_ari_score_umap_lst.append(kmeans_umap_test_ari_score)

        gmm_pca = GaussianMixture(n_components=num_unique_cells, max_iter=1000, random_state=42)
        gmm_umap = GaussianMixture(n_components=num_unique_cells, max_iter=1000, random_state=42)

        gmm_pca.fit(X_train_pca)
        gmm_umap.fit(X_train_umap)

        gmm_pca_y_train_pred = gmm_pca.predict(X_train_pca)
        gmm_pca_y_test_pred = gmm_pca.predict(X_test_pca)
        gmm_umap_y_train_pred = gmm_umap.predict(X_train_umap)
        gmm_umap_y_test_pred = gmm_umap.predict(X_test_umap)

        gmm_pca_train_ari_score = adjusted_rand_score(y_train, gmm_pca_y_train_pred)
        gmm_pca_test_ari_score = adjusted_rand_score(y_test, gmm_pca_y_test_pred)
        gmm_umap_train_ari_score = adjusted_rand_score(y_train, gmm_umap_y_train_pred)
        gmm_umap_test_ari_score = adjusted_rand_score(y_test, gmm_umap_y_test_pred)

        train_gmm_ari_score_pca_lst.append(gmm_pca_train_ari_score)
        test_gmm_ari_score_pca_lst.append(gmm_pca_test_ari_score)
        train_gmm_ari_score_umap_lst.append(gmm_umap_train_ari_score)
        test_gmm_ari_score_umap_lst.append(gmm_umap_test_ari_score)

    return pd.DataFrame(data={'n_components': n_components_lst,
                              'Train K-Means PCA': train_kmeans_ari_score_pca_lst,
                              'Test K-Means PCA': test_kmeans_ari_score_pca_lst,
                              'Train K-Means UMAP': train_kmeans_ari_score_umap_lst,
                              'Test K-Means UMAP': test_kmeans_ari_score_umap_lst,
                              'Train GMM PCA': train_gmm_ari_score_pca_lst,
                              'Test GMM PCA': test_gmm_ari_score_pca_lst,
                              'Train GMM UMAP': train_gmm_ari_score_umap_lst,
                              'Test GMM UMAP': test_gmm_ari_score_umap_lst,
                              }).set_index('n_components')

import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, f1_score

def write_cluster_gene_data(cell_df, n_components, N_TRUNC_GENE):
    """
    Saves cluster gene data to file and returns a dictionary of top N_TRUNC_GENE per cluster 
    and list of predicted cluster per cell using n_components
    """
    y = [cell.split('.')[0] for cell in cell_df.T.index]
    num_unique_cells = len(np.unique(y))

    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=42)
    X_umap = umap.UMAP(min_dist=0.5, spread=1, random_state=42)

    X_scaled = scaler.fit_transform(cell_df.T.values)
    X_pca = pca.fit_transform(X_scaled)
    X_umap = X_umap.fit_transform(X_pca)

    gmm_umap = GaussianMixture(n_components=num_unique_cells, max_iter=1000, random_state=42)
    gmm_umap.fit(X_umap)
    gmm_umap_pred = gmm_umap.predict(X_umap)

    cell_gene_dict = {}
    top_gene_data = get_seq_embed_gpt(np.array(cell_df.T),
                                    np.array(cell_df.T.columns), 
                                    prompt_prefix='',
                                    trunc_index=N_TRUNC_GENE)

    for i, cluster in enumerate(gmm_umap_pred):
        gene_lst = top_gene_data[i].split(' ')
        if cluster in cell_gene_dict:
            for gene in gene_lst:
                if gene in cell_gene_dict[cluster]:
                    cell_gene_dict[cluster][gene] += 1
                else:
                    cell_gene_dict[cluster][gene] = 1
        else:
            cell_gene_dict[cluster] = {}
            for gene in gene_lst:
                if gene in cell_gene_dict[cluster]:
                    cell_gene_dict[cluster][gene] += 1
                else:
                    cell_gene_dict[cluster][gene] = 1

    top_cell_gene_dict = {}
    for cluster in cell_gene_dict:
        top_genes = map(lambda x: x[0], sorted(cell_gene_dict[cluster].items(), key=lambda x: x[1], reverse=True)[:N_TRUNC_GENE])
        top_cell_gene_dict[cluster] = list(top_genes)

    with open('data.txt', 'w') as f:
        for cluster, genes in top_cell_gene_dict.items():
            item = ' '.join(genes)
            f.write(f'{item}\n')

    return top_cell_gene_dict, gmm_umap_pred


def get_genePT_GPT_metric(cell_df, gpt_output_file, top_cell_gene_dict, gmm_umap_pred):
    """
    Prints accuracy and F1 score for GenePT + GPT-4 annotation
    """
    original_cell_lst = [cell.split('.')[0] for cell in cell_df.T.index]

    with open(gpt_output_file, 'r') as f:
        cluster_cell_lst = [cell.strip().replace(' Cell', '').replace(' cell', '') for cell in f.readlines()]

    cluster_annotations_dict = {}
    cluster_annotations_lst = []

    for i, cluster in enumerate(top_cell_gene_dict.keys()):
        cluster_annotations_dict[cluster] = cluster_cell_lst[i]

    for cluster in gmm_umap_pred:
        cluster_annotations_lst.append(cluster_annotations_dict[cluster])

    accuracy = accuracy_score(original_cell_lst, cluster_annotations_lst)
    f1_result = f1_score(original_cell_lst, cluster_annotations_lst, average='weighted')

    print(original_cell_lst)
    print(cluster_annotations_lst)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'F1-Score: {f1_result:.4f}')

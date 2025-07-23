import numpy as np
import pandas as pd
import openai 
from IPython.display import display

def load_data(file):
    """
    Loads in pandas dataframe from a csv file
    """
    df = pd.read_csv(file, sep=',')
    df = df.set_index('Unnamed: 0')
    df.index.name = None
    return df

def data_info(df):
    """
    Information about the dataframe
    """
    cell_counts = {}
    cell_lst = np.array([col.split('.')[0] for col in df.columns])
    unique_cells = np.unique(cell_lst)
    average_row = df.sum(axis=1).mean()
    average_col = df.sum(axis=0).mean()

    for cell in cell_lst:
        if cell in cell_counts:
            cell_counts[cell] += 1
        else:
            cell_counts[cell] = 1

    print(f'Number of genes: {df.shape[0]}')
    print(f'Number of cells: {df.shape[1]}')
    print(f'Number of unique cells: {len(unique_cells)}')
    print(f'Cells: {unique_cells}')
    print(f'Average gene row sum: {average_row}')
    print(f'Average gene column sum: {average_col}')
    print('Cell counts:')
    display(pd.DataFrame(data=cell_counts, index=[0]))


def get_seq_embed_gpt(X, gene_names, prompt_prefix="", trunc_index = None):
    """
    Generate GenePT-s cell embeddings
    """
    n_genes = X.shape[1]
    if trunc_index is not None and not isinstance(trunc_index, int):
        raise Exception('trunc_index must be None or an integer!')
    elif isinstance(trunc_index, int) and trunc_index>=n_genes:
        raise Exception('trunc_index must be smaller than the number of genes in the dataset')
    get_test_array = []
    for cell in (X):
        zero_indices = (np.where(cell==0)[0])
        gene_indices = np.argsort(cell)[::-1]
        filtered_genes = gene_indices[~np.isin(gene_indices, list(zero_indices))]
        if trunc_index is not None:
            get_test_array.append(np.array(gene_names[filtered_genes])[0:trunc_index]) 
        else:
            get_test_array.append(np.array(gene_names[filtered_genes])) 
    get_test_array_seq = [prompt_prefix + ' '.join(x) for x in get_test_array]
    return get_test_array_seq

def get_gpt_embedding(text, model="text-embedding-ada-002"):
    """
    Generate GenePT-s cell embeddings using the updated OpenAI client
    """
    text = text.replace("\n", " ")
    
    client = openai.OpenAI(api_key=key)

    response = client.embeddings.create(
        input=[text],
        model=model
    )

    return np.array(response.data[0].embedding)

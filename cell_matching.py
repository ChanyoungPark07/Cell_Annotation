import numpy as np
import pandas as pd
import requests
from openai import OpenAI
import urllib.parse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel 
from sklearn.metrics.pairwise import cosine_similarity

def soft_match(original_df, gpt_output_file):
    """
    Prints accuracies for both exact and soft match between original and GPT outputted cell names
    """
    original_cell_lst = [cell.split('.')[0] for cell in original_df.T.index]
    accuracy_count = 0
    soft_accuracy_count = 0

    with open(gpt_output_file, 'r') as f:
        gpt_cell_lst = [cell.strip().replace(' Cell', '').replace(' cell', '') for cell in f.readlines()]

    for i in range(len(gpt_cell_lst)):
        if gpt_cell_lst[i] == original_cell_lst[i]:
            accuracy_count += 1
        
        if gpt_cell_lst[i] in original_cell_lst[i] \
            or original_cell_lst[i] in gpt_cell_lst[i]:
            soft_accuracy_count += 1
            
    print(f'Accuracy of GPT-4 Cell Type Annotation: {accuracy_count / len(gpt_cell_lst) * 100:.2f}%')
    print(f'Accuracy of GPT-4 Cell Type Soft Annotation: {soft_accuracy_count / len(gpt_cell_lst) * 100:.2f}%')


def SapBERT_match(original_df, gpt_output_file, threshold):
    """
    Prints accuracy using SapBERT cosine similarity with a defined threshold
    """
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")  
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

    original_cell_lst = [cell.split('.')[0] for cell in original_df.T.index]
    accuracy_count = 0

    with open(gpt_output_file, 'r') as f:
        gpt_cell_lst = [cell.strip().replace(' Cell', '').replace(' cell', '') for cell in f.readlines()]

    # Entity names
    unique_cells = set(original_cell_lst)
    unique_cells.update(set(gpt_cell_lst))
    all_names = list(unique_cells)

    bs = 128 # Batch size during inference
    all_embs = []

    for i in tqdm(np.arange(0, len(all_names), bs)):
        toks = tokenizer.batch_encode_plus(all_names[i:i+bs], 
                                        padding="max_length", 
                                        max_length=25, 
                                        truncation=True,
                                        return_tensors="pt")
        toks_cuda = {}

        for k,v in toks.items():
            toks_cuda[k] = v

        cls_rep = model(**toks_cuda)[0][:,0,:] # Use CLS representation as the embedding
        all_embs.append(cls_rep.cpu().detach().numpy())

    all_embs = np.concatenate(all_embs, axis=0)

    cell_embs = {}

    for i, cell in enumerate(all_names):
        cell_embs[cell] = all_embs[i]

    for i in range(len(gpt_cell_lst)):
        original_cell = original_cell_lst[i]
        gpt_cell = gpt_cell_lst[i]

        original_emb = cell_embs[original_cell].reshape(1, -1)
        gpt_emb = cell_embs[gpt_cell].reshape(1, -1)

        similarity_matrix = cosine_similarity(original_emb, gpt_emb)
        similarity_score = similarity_matrix[0][0]

        if similarity_score >= threshold:
            accuracy_count += 1
        
    print(f'Accuracy of GPT-4 Cell Type Annotation: {accuracy_count / len(gpt_cell_lst) * 100:.2f}%')


def NameResolver_match(original_df, gpt_output_file):
    """
    Prints accuracy using Name Resolver
    """
    original_cell_lst = [cell.split('.')[0] for cell in original_df.T.index]
    accuracy_count = 0

    with open(gpt_output_file, 'r') as f:
        gpt_cell_lst = [cell.strip().replace(' Cell', '').replace(' cell', '') for cell in f.readlines()]

    similar_dict = {}

    for cell in set(original_cell_lst):
        encoded_cell = urllib.parse.quote(cell)
        response = requests.get(f"https://name-resolution-sri.renci.org/lookup?string={encoded_cell}&autocomplete=false&highlighting=false&offset=0&limit=10&biolink_type=biolink%3ACell")

        if response.status_code == 200:
                data = response.json()
                if len(data) != 0:
                    synonyms = set(data[0]['synonyms'])
                    similar_dict[cell] = synonyms
                else:
                    similar_dict[cell] = cell
        else:
            print(f"Error: {response.status_code}")

    for i in range(len(gpt_cell_lst)):
        original_cell = original_cell_lst[i]
        gpt_cell = gpt_cell_lst[i]

        if gpt_cell in similar_dict[original_cell] or \
            gpt_cell + ' Cell' in similar_dict[original_cell] or \
            gpt_cell + ' cell' in similar_dict[original_cell] or \
            gpt_cell == original_cell:
            accuracy_count += 1

    print(f'Accuracy of GPT-4 Cell Type Annotation: {accuracy_count / len(gpt_cell_lst) * 100:.2f}%')


def write_gpt_name_resolver(resolver_df, save_path, original_lst, output_lst, key):
    """
    Write OpenAI's GPT-4o name resolver pairs and similarities to dataframe and save as csv. 
    The dataframe must have one column for the score and be multi-indexed (output_cell, original_cell)
    """
    client = OpenAI(api_key=key)
    original_set = set(original_lst)
    output_set = set(output_lst)
    column_name = resolver_df.columns[0]

    for original_cell in original_set:
        for output_cell in output_set:
            pair = (output_cell, original_cell)
            
            if pair in resolver_df.index:
                print(f'The cell pair {pair} already exists')
            else:
                response = client.responses.create(
                    model='gpt-4o',
                    temperature=0,
                    input=f'Are {output_cell} the same or a type of {original_cell} cell? Respond only with a confidence score between integer 0 and 10.'
                    )
                resolver_df.loc[(output_cell, original_cell), column_name] = int(response.output[0].content[0].text)

    resolver_df.to_csv(save_path)


def NameResolver_match(original_df, gpt_output_file, key):
    """
    Prints accuracy using gpt-4o
    """
    original_cell_lst = [cell.split('.')[0] for cell in original_df.T.index]
    accuracy_count = 0

    # Original write to the gpt resolver dataframe
    # empty_df = pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=['output_cell', 'original_cell']), columns=['similarity_score'])
    # write_gpt_name_resolver(empty_df, 'gpt_resolver_df.csv', original_cell_lst, gpt_output_file, key)
    
    # Use the existing gpt resolver dataframe
    gpt_resolver_df = pd.read_csv('gpt_resolver_df.csv').set_index(['output_cell', 'original_cell'])
    write_gpt_name_resolver(gpt_resolver_df, 'gpt_resolver_df.csv', original_cell_lst, gpt_output_file, key)
    gpt_resolver_df = pd.read_csv('gpt_resolver_df.csv').set_index(['output_cell', 'original_cell'])

    for i in range(len(gpt_output_file)):
        original_cell = original_cell_lst[i]
        output_cell = gpt_output_file[i]
        pair = (output_cell, original_cell)
        similarity_score = gpt_resolver_df.loc[pair]['similarity_score']
        
        if similarity_score >= 9:
            accuracy_count += 1

    print(f'Accuracy of scType Cell Type Annotation: {accuracy_count / len(gpt_output_file) * 100:.2f}%')

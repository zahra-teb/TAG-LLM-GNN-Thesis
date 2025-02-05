import pickle
import torch
import pandas as pd

# def get_dataset(path):
#     # Load the graph from the .pkl file
#     with open(path, "rb") as f:
#         graph = pickle.load(f)
    
#     return graph

def get_products(graph_path, texts_path):
    data = torch.load(graph_path)
    text = pd.read_csv(texts_path)
    text = [f'Product:{ti}; Description: {cont}\n'for ti,
            cont in zip(text['title'], text['content'])]

    data.edge_index = data.adj_t.to_symmetric()

    # if not use_text:
    #     return data, None

    return data, text

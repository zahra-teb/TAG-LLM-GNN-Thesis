import pickle

def get_dataset(path):
    # Load the graph from the .pkl file
    with open(path, "rb") as f:
        graph = pickle.load(f)
    
    return graph
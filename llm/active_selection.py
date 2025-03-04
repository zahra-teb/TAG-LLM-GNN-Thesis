import torch
import networkx as nx
from sklearn.cluster import KMeans  
from torch_geometric.utils import to_edge_index


def single_score(b, x, data, seed, device, pg_rank_score=0.85):
    edge_index = to_edge_index(data.adj_t)[0]  # Converts `adj_t` (SparseTensor) to `edge_index`
    edges = [(int(i), int(j)) for i, j in zip(edge_index[0], edge_index[1])]
    nodes = list(range(x.shape[0]))
    
    # Construct graph
    data.g = nx.Graph()
    data.g.add_nodes_from(nodes)
    data.g.add_edges_from(edges)

    # Compute PageRank
    page = torch.tensor(list(nx.pagerank(data.g, alpha=pg_rank_score).values()), dtype=x.dtype, device=device)

    # Feature Clustering for Density using sklearn KMeans
    kmeans = KMeans(n_clusters=data.y.max().item() + 1, random_state=seed)
    labels = kmeans.fit_predict(x.cpu().numpy())  # Convert tensor to NumPy for sklearn
    centers = torch.tensor(kmeans.cluster_centers_[labels], dtype=x.dtype, device=device)

    x = x.to(device)
    dist_map = torch.linalg.norm(x - centers, dim=1).to(x.dtype)
    density = 1 / (1 + dist_map)

    # Normalize percentile ranks
    percentile = torch.arange(x.shape[0], dtype=x.dtype, device=device) / x.shape[0]
    density[density.argsort(descending=False)] = percentile
    page[page.argsort(descending=False)] = percentile

    # Compute selection score
    alpha, beta = 0.8, 0.2
    selection_score = alpha * page + beta * density  

    # Select top-b nodes
    _, indices = torch.topk(selection_score, k=b)
    return indices

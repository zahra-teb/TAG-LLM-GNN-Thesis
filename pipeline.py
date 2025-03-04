from selection.active_selection import single_score
from llm.annotation import generate_annotations
from embedding.generate_embeddings import generate_embeddings
from training.train_gnn import train

selected_nodes = single_score(500, node_features, graph_data, 42, confidences, device)
generate_annotations(selected_nodes, texts, "Explain why this node belongs to its predicted category: {description}")
h_orig, h_expl = generate_embeddings(texts, explanations)
train()

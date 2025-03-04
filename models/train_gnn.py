# from torch_geometric.nn import GCNConv
# import torch.nn.functional as F
# import torch

# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)

# def evaluate_accuracy(preds, labels):
#     correct = preds.argmax(dim=1).eq(labels).sum().item()
#     return correct / labels.size(0)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = GCN(in_channels=h_orig.shape[1] + h_expl.shape[1], hidden_channels=64, out_channels=len(set(filtered_labels.tolist()))).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# def train():
#     model.train()
#     optimizer.zero_grad()
#     x = torch.cat([h_orig, h_expl], dim=1)  # Combine embeddings
#     out = model(x, graph_data.edge_index)
#     loss = F.nll_loss(out[filtered_nodes], filtered_labels)
#     acc = evaluate_accuracy(out[filtered_nodes], filtered_labels)
#     loss.backward()
#     optimizer.step()
#     return loss.item(), acc

# for epoch in range(100):
#     loss, acc = train()
#     print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

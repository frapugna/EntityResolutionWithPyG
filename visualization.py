import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import networkx as nx

def visualize_graph(g):
    G = nx.Graph()
    edge_list = g.edges
    index_to_token = g.index_to_token

    for i in range(g.number_of_tokens):
        G.add_node(index_to_token[i])
    for i in range(len(edge_list[0])):
        G.add_edge(index_to_token[edge_list[0][i]], index_to_token[edge_list[1][i]])

    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True,
                     node_color='yellow', font_color='black', edge_color='blue',cmap="Set2")
    plt.show()

'''
def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()
'''
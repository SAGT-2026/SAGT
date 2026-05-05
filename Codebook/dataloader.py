from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
import torch
import numpy as np
from torch_geometric.utils import to_dense_adj


def load_the_dataset(dataset_name=None, batch_size=32, shuffle=True, num_workers=0, dataset=None):
    if dataset is None:
        dataset = TUDataset(root='/tmp/' + dataset_name, name=dataset_name)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


def parse_the_dataset_to_matrices(dataset_name=None, max_graphs=None, batch_size=32, dataset_here=None):
    dataset = dataset_here
    
    if dataset_here is None:
        dataset = TUDataset(root='/tmp/' + dataset_name, name=dataset_name)
        print(f"Loading {dataset_name} dataset with {len(dataset)} graphs")
    
    if max_graphs is not None:
        dataset = dataset[:max_graphs]
    
    print(f"Total graphs in dataset: {len(dataset)}")
    
    if dataset_here is None:
        num_node_features = dataset.num_node_features
        num_classes = dataset.num_classes
        max_nodes = max([data.num_nodes for data in dataset])
        print(f"Max nodes: {max_nodes}, Node features: {num_node_features}, Classes: {num_classes}")
    else:
        max_nodes = max([data.num_nodes for data in dataset])
        num_node_features = dataset.num_node_features if hasattr(dataset, 'num_node_features') else dataset[0].x.shape[1]
        num_classes = dataset.num_classes if hasattr(dataset, 'num_classes') else len(set([data.y.item() for data in dataset]))
        print(f"Max nodes: {max_nodes}, Node features: {num_node_features}, Classes: {num_classes}")
        
    all_batch_x = []
    all_batch_adj = []
    all_batch_num_nodes = []
    all_labels = []
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("Processing dataset in batches...")
    
    for batch_data in loader:
        batch_x, batch_adj, batch_num_nodes, labels = convert_pyg_batch_to_matrices(
            batch_data, max_nodes, num_node_features
        )
        
        all_batch_x.append(batch_x)
        all_batch_adj.append(batch_adj)
        all_batch_num_nodes.extend(batch_num_nodes)
        all_labels.append(labels)

    final_batch_x = torch.cat(all_batch_x, dim=0)
    final_batch_adj = torch.cat(all_batch_adj, dim=0)
    final_labels = torch.cat(all_labels, dim=0)
    
    return final_batch_x, final_batch_adj, all_batch_num_nodes, final_labels, max_nodes


def convert_pyg_batch_to_matrices(batch_data, max_nodes, num_node_features):
    batch_size = batch_data.num_graphs
    
    batch_x = torch.zeros(batch_size, max_nodes, num_node_features)
    batch_adj = torch.zeros(batch_size, max_nodes, max_nodes)
    batch_num_nodes = []
    
    ptr = batch_data.ptr
    
    for i in range(batch_size):
        start_idx = ptr[i]
        end_idx = ptr[i + 1]
        num_nodes = end_idx - start_idx
        batch_num_nodes.append(num_nodes.item())
        
        node_features = batch_data.x[start_idx:end_idx]
        batch_x[i, :num_nodes] = node_features
        
        edge_mask = (batch_data.edge_index[0] >= start_idx) & (batch_data.edge_index[0] < end_idx)
        graph_edges = batch_data.edge_index[:, edge_mask] - start_idx
        
        if graph_edges.size(1) > 0:
            adj_matrix = torch.zeros(max_nodes, max_nodes)
            adj_matrix[graph_edges[0], graph_edges[1]] = 1
            adj_matrix = adj_matrix + adj_matrix.t()
            adj_matrix = torch.clamp(adj_matrix, 0, 1)
            batch_adj[i] = adj_matrix
        
        for j in range(num_nodes):
            batch_adj[i, j, j] = 1
    
    labels = batch_data.y
    return batch_x, batch_adj, batch_num_nodes, labels


def load_and_parse_dataset(dataset_name, max_graphs=None, test_split=0.2):
    batch_x, batch_adj, batch_num_nodes, labels, max_nodes = parse_the_dataset_to_matrices(
        dataset_name, max_graphs
    )
    
    num_graphs = len(batch_num_nodes)
    num_test = int(num_graphs * test_split)
    num_train = num_graphs - num_test
    
    indices = torch.randperm(num_graphs)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]
    
    data_dict = {
        'train': {
            'x': batch_x[train_indices],
            'adj': batch_adj[train_indices],
            'num_nodes': [batch_num_nodes[i] for i in train_indices],
            'labels': labels[train_indices]
        },
        'test': {
            'x': batch_x[test_indices],
            'adj': batch_adj[test_indices], 
            'num_nodes': [batch_num_nodes[i] for i in test_indices],
            'labels': labels[test_indices]
        },
        'meta': {
            'max_nodes': max_nodes,
            'num_node_features': batch_x.shape[2],
            'num_classes': len(torch.unique(labels)),
            'dataset_name': dataset_name
        }
    }
    
    print(f"Dataset split - Train: {num_train}, Test: {num_test}")
    print(f"Feature shape: {batch_x.shape}, Adjacency shape: {batch_adj.shape}")
    
    return data_dict


def get_sample_batch(data_dict, split='train', batch_size=4, start_idx=0):
    data = data_dict[split]
    end_idx = min(start_idx + batch_size, len(data['num_nodes']))
    
    batch_x = data['x'][start_idx:end_idx]
    batch_adj = data['adj'][start_idx:end_idx]
    batch_num_nodes = data['num_nodes'][start_idx:end_idx]
    labels = data['labels'][start_idx:end_idx]
    
    return batch_x, batch_adj, batch_num_nodes, labels
    


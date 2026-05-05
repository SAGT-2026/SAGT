import torch
import matplotlib.pyplot as plt
import numpy as np
import math

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import Codebook.dataloader as dataloader
import Codebook.train as train

def remove_diagonal(adj_matrix):
    mask = torch.eye(adj_matrix.size(-1), device=adj_matrix.device).bool()
    adj_matrix_no_diag = adj_matrix.clone()
    adj_matrix_no_diag[mask] = 0
    return adj_matrix_no_diag

def tokenize_graphs(model, batch_x, batch_adj, batch_num_nodes, device):
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        batch_x = batch_x.to(device)
        batch_adj = batch_adj.to(device)
    
        cluster_features, cluster_adj, pooling_assignments = model.encoder.forward(
            batch_x, batch_adj, batch_num_nodes, return_cluster_info=True
        )
        
        quantized_features,  encodings, _ , _ = model.codebook(cluster_features)
        
        tokens = encodings  
        
        return {
            'tokens': tokens,
            'quantized_features': quantized_features,
            'cluster_adj': cluster_adj,
            'cluster_features': cluster_features
        }
        
def generate_sinusoidal_position_encoding(max_len, d_model, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pe = torch.zeros(max_len, d_model, device=device)
    
    position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
    
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * 
                        -(math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)  
    pe[:, 1::2] = torch.cos(position * div_term)  
    
    return pe


def load_pretrained_model_and_tokenize(
    model_path='..',
    batch_size=8,
    device=None,
    config=None,
    num_node_features=30,
    num_edge_features=40,
    verbose=True
):
    
    if config is None:
        config = {
            'hidden_dim': 256,
            'embedding_dim': 4096,
            'num_layers': 2,
            'assign_hidden_dim': 32,
            'assign_ratio': 0.5,
            'num_pooling': 2,  
            
            'codebook_size': 512,
            'codebook_type': 'euclidean',
            'num_codebooks': 1,
            'decay': 0.9,
            'threshold_ema_dead_code': 5,
            'sample_codebook_temp': 0.0,
            'gaussian_delta': 1.0,
            
            'unpooling_method': 'mlp_transform',

            'commitment_weight': 1.0,
            'reconstruction_weight': 1,
            'structure_weight': 0.3,
            'vq_weight': 1.0,
            'dropout': 0.1,
            'linkpred_weight': 1,  
            'linkpred': True,        
            'bn': True
        }
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"Using device: {device}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        ckpt_input_dim = checkpoint.get('input_dim', num_node_features)
        ckpt_max_nodes = checkpoint.get('max_nodes', num_node_features)
        current_model = train.create_model(ckpt_input_dim, ckpt_max_nodes, config)
        current_model.load_state_dict(checkpoint['model_state_dict'])
        current_model = current_model.to(device)
        current_model.eval()
        if verbose:
            print("Successfully loaded pretrained model!")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    try:
        if verbose:
            print("Loading dataset...")
        dataset_origin = torch.load('sider_hepatobiliary_dataset.pt',weights_only=False)
        batch_x, batch_adj, batch_num_nodes, _, max_nodes = dataloader.parse_the_dataset_to_matrices(dataset_here=dataset_origin)
        
        if verbose:
            print(f"Dataset info: {len(batch_x)} graphs, max {max_nodes} nodes")
            print(f"Graph sizes: {batch_num_nodes[:8]}")
            print(f"Batch shape: {batch_x.shape}, Adjacency shape: {batch_adj.shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")
    
    try:
        if verbose:
            print("tokenization...")
        tokenization_results = tokenize_graphs(current_model, batch_x, batch_adj, batch_num_nodes, device)
        
        if verbose:
            tokens = tokenization_results['tokens']
            print(f"Token shape: {tokens.shape}")
            print(f"Token range: {tokens.min().item()} - {tokens.max().item()}")
    except Exception as e:
        raise RuntimeError(f"Failed to tokenize graphs: {e}")
    
    cluster_adj = tokenization_results['cluster_adj']
    batch_size = cluster_adj.shape[0]
    processed_adj_matrices = []
    
    for i in range(batch_size):
        new_adj_matrix = (cluster_adj[i] > 1).float()
        new_adj_matrix = remove_diagonal(new_adj_matrix)
        processed_adj_matrices.append(new_adj_matrix)
    
    normalized_adj_batch = torch.stack(processed_adj_matrices, dim=0)
    
    
    return {
        **tokenization_results,
        'motif_features': tokenization_results['quantized_features'], 
        'adj_matrix':   normalized_adj_batch
    }
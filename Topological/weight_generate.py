
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import graph_tokenize 


def _default_init_func(_m):
    if isinstance(_m, nn.Linear):
        nn.init.trunc_normal_(_m.weight, std=.02)
        if _m.bias is not None:
            nn.init.constant_(_m.bias, 0)
    elif isinstance(_m, nn.LayerNorm):
        nn.init.constant_(_m.bias, 0)
        nn.init.constant_(_m.weight, 1.0)
    elif isinstance(_m, nn.Parameter):
        nn.init.trunc_normal_(_m, std=.02)

class GraphEncoder(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, output_dim, num_layers=3, gnn_type='GAT'):
        super().__init__()
        self.node_embedding = nn.Linear(node_feat_dim, hidden_dim)
        
        if gnn_type == 'GAT':
            self.gnn_layers = nn.ModuleList([
                GATConv(hidden_dim, hidden_dim, heads=4, concat=False) 
                for _ in range(num_layers)
            ])
        else:  
            self.gnn_layers = nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim) 
                for _ in range(num_layers)
            ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def adjacency_matrix_to_edge_index(self, adjacency_matrix):
        edge_index = adjacency_matrix.nonzero().t().contiguous()
        return edge_index

    def forward(self, node_features, adjacency_matrix, batch=None):
        edge_index = self.adjacency_matrix_to_edge_index(adjacency_matrix)
        
        x = self.node_embedding(node_features)
    
        for gnn_layer, norm in zip(self.gnn_layers, self.norm_layers):
            x_new = gnn_layer(x, edge_index)
            x = norm(x + self.dropout(x_new))  
            x = F.relu(x)
        
        if batch is not None:
            graph_embeddings = global_mean_pool(x, batch)
        else:
            graph_embeddings = x.mean(dim=0, keepdim=True)
        
        graph_embeddings = self.output_proj(graph_embeddings)
        
        return graph_embeddings


class GraphLoRAGenerater(nn.Module):
    def __init__(self, dim, depth,
                 node_feat_dim, graph_hidden_dim, pos_num,
                 llm_dim, llm_depth, lora_rank,
                 lora_type='qkvom', weights_sep=True,
                 mlp_ratio=4, num_heads=16, skip_layers=1, topolora_alpha=None,
                 gnn_type='GAT', gnn_layers=3):
        super().__init__()

        self.graph_encoder = GraphEncoder(
            node_feat_dim, graph_hidden_dim, dim, gnn_layers, gnn_type
        )

        self.lora_type = lora_type
        self.weights_sep = weights_sep
        self.skip_layers = skip_layers

        self.weights_head = nn.Linear(dim, len(lora_type)*llm_depth//skip_layers*llm_dim*lora_rank)

        self.apply(_default_init_func)
        self.As = nn.Parameter(
            torch.zeros(1, len(lora_type)*llm_depth//skip_layers, llm_dim*lora_rank),
            requires_grad=True
        )

        self.topolora_alpha = topolora_alpha if topolora_alpha is not None else lora_rank
        self.llm_depth = llm_depth
        self.llm_dim = llm_dim
        self.lora_rank = lora_rank
        
    def forward(self, node_features, adjacency_matrix, batch=None):
        graph_feature = self.graph_encoder(node_features, adjacency_matrix, batch)
        weights = self.weights_head(graph_feature)
        weights = weights.view(
            weights.shape[0], len(self.lora_type),
            self.llm_depth//self.skip_layers, self.llm_dim, self.lora_rank
        )

        As = self.As.reshape(
            1, len(self.lora_type), self.llm_depth//self.skip_layers,
            self.llm_dim, self.lora_rank
        )
        As = self.topolora_alpha / self.lora_rank * As

        lora_weights_list = []
        for depth in range(self.llm_depth):
            lora_weights = {}
            if (depth + 1) % self.skip_layers == 0:
                for i, type_char in enumerate(str(self.lora_type)):
                    A = As[:, i, depth//self.skip_layers]
                    B = weights[:, i, depth//self.skip_layers]
                    lora_weights[type_char] = (A, B)
                for j in ['q', 'k', 'v', 'o', 'm']:
                    if j not in lora_weights:
                        lora_weights[j] = (None, None)
            else:
                for j in ['q', 'k', 'v', 'o', 'm']:
                    lora_weights[j] = (None, None)
            lora_weights_list.append(lora_weights)

        return lora_weights_list


class HypernetworkLoRAGenerator(nn.Module):
    def __init__(self, dim, depth,
                 node_feat_dim, graph_hidden_dim, pos_num,
                 llm_dim, llm_depth, lora_rank,
                 lora_type='qkvom', weights_sep=True,
                 skip_layers=1, topolora_alpha=None,
                 gnn_type='GAT', gnn_layers=3,
                 factor_k=8, hyper_hidden_dim=512):
        super().__init__()

        self.graph_encoder = GraphEncoder(
            node_feat_dim, graph_hidden_dim, dim, gnn_layers, gnn_type
        )

        self.lora_type = lora_type
        self.skip_layers = skip_layers
        self.llm_depth = llm_depth
        self.llm_dim = llm_dim
        self.lora_rank = lora_rank
        self.factor_k = factor_k

        num_types = len(lora_type)
        num_active_layers = llm_depth // skip_layers

        self.layer_embedding = nn.Embedding(num_active_layers, dim)
        self.type_embedding = nn.Embedding(num_types, dim)

        output_dim = llm_dim * factor_k + factor_k * lora_rank
        self.hyper_mlp = nn.Sequential(
            nn.Linear(3 * dim, hyper_hidden_dim),
            nn.GELU(),
            nn.Linear(hyper_hidden_dim, output_dim)
        )

        self.As = nn.Parameter(
            torch.zeros(1, num_types * num_active_layers, llm_dim * lora_rank),
            requires_grad=True
        )

        self.topolora_alpha = topolora_alpha if topolora_alpha is not None else lora_rank

        self.apply(_default_init_func)

    def forward(self, node_features, adjacency_matrix, batch=None):
        graph_feature = self.graph_encoder(node_features, adjacency_matrix, batch)

        num_types = len(self.lora_type)
        num_active_layers = self.llm_depth // self.skip_layers
        batch_size = graph_feature.shape[0]

        layer_indices = torch.arange(num_active_layers, device=graph_feature.device)
        type_indices = torch.arange(num_types, device=graph_feature.device)
        layer_grid, type_grid = torch.meshgrid(layer_indices, type_indices, indexing='ij')
        layer_ids = layer_grid.reshape(-1)
        type_ids = type_grid.reshape(-1)

        layer_embs = self.layer_embedding(layer_ids)
        type_embs = self.type_embedding(type_ids)

        num_pairs = len(layer_ids)
        graph_expanded = graph_feature.unsqueeze(1).expand(-1, num_pairs, -1)
        layer_embs_expanded = layer_embs.unsqueeze(0).expand(batch_size, -1, -1)
        type_embs_expanded = type_embs.unsqueeze(0).expand(batch_size, -1, -1)

        mlp_input = torch.cat([graph_expanded, layer_embs_expanded, type_embs_expanded], dim=-1)
        mlp_input_flat = mlp_input.reshape(batch_size * num_pairs, -1)

        mlp_output = self.hyper_mlp(mlp_input_flat)

        U_dim = self.llm_dim * self.factor_k
        U = mlp_output[:, :U_dim].reshape(batch_size, num_pairs, self.llm_dim, self.factor_k)
        V = mlp_output[:, U_dim:].reshape(batch_size, num_pairs, self.factor_k, self.lora_rank)

        B_all = torch.matmul(U, V)
        B_all = B_all.reshape(batch_size, num_active_layers, num_types, self.llm_dim, self.lora_rank)

        As = self.As.reshape(1, num_types, num_active_layers, self.llm_dim, self.lora_rank)
        As = self.topolora_alpha / self.lora_rank * As

        lora_weights_list = []
        for depth in range(self.llm_depth):
            lora_weights = {}
            if (depth + 1) % self.skip_layers == 0:
                for i, type_char in enumerate(str(self.lora_type)):
                    A = As[:, i, depth // self.skip_layers]
                    B = B_all[:, depth // self.skip_layers, i]
                    lora_weights[type_char] = (A, B)
                for j in ['q', 'k', 'v', 'o', 'm']:
                    if j not in lora_weights:
                        lora_weights[j] = (None, None)
            else:
                for j in ['q', 'k', 'v', 'o', 'm']:
                    lora_weights[j] = (None, None)
            lora_weights_list.append(lora_weights)

        return lora_weights_list


class PrototypeLoRAGenerator(nn.Module):
    def __init__(self, dim, depth,
                 node_feat_dim, graph_hidden_dim, pos_num,
                 llm_dim, llm_depth, lora_rank,
                 lora_type='qkvom', weights_sep=True,
                 skip_layers=1, topolora_alpha=None,
                 gnn_type='GAT', gnn_layers=3,
                 prototype_K=4, proto_hidden_dim=256):
        super().__init__()

        self.graph_encoder = GraphEncoder(
            node_feat_dim, graph_hidden_dim, dim, gnn_layers, gnn_type
        )

        self.lora_type = lora_type
        self.skip_layers = skip_layers
        self.llm_depth = llm_depth
        self.llm_dim = llm_dim
        self.lora_rank = lora_rank
        self.prototype_K = prototype_K

        num_types = len(lora_type)
        num_active_layers = llm_depth // skip_layers

        self.prototypes = nn.Parameter(
            torch.zeros(num_types, prototype_K, llm_dim, lora_rank),
            requires_grad=True
        )

        mixing_output_dim = num_types * num_active_layers * prototype_K
        self.mixing_mlp = nn.Sequential(
            nn.Linear(dim, proto_hidden_dim),
            nn.GELU(),
            nn.Linear(proto_hidden_dim, mixing_output_dim)
        )

        self.As = nn.Parameter(
            torch.zeros(1, num_types * num_active_layers, llm_dim * lora_rank),
            requires_grad=True
        )

        self.topolora_alpha = topolora_alpha if topolora_alpha is not None else lora_rank

        self.apply(_default_init_func)

    def forward(self, node_features, adjacency_matrix, batch=None):
        graph_feature = self.graph_encoder(node_features, adjacency_matrix, batch)

        num_types = len(self.lora_type)
        num_active_layers = self.llm_depth // self.skip_layers
        batch_size = graph_feature.shape[0]

        mixing_raw = self.mixing_mlp(graph_feature)
        mixing_weights = mixing_raw.reshape(
            batch_size, num_types, num_active_layers, self.prototype_K
        )
        mixing_weights = F.softmax(mixing_weights, dim=-1)

        B_all = torch.einsum('btlk,tkdr->btldr', mixing_weights, self.prototypes)

        As = self.As.reshape(1, num_types, num_active_layers, self.llm_dim, self.lora_rank)
        As = self.topolora_alpha / self.lora_rank * As

        lora_weights_list = []
        for depth in range(self.llm_depth):
            lora_weights = {}
            if (depth + 1) % self.skip_layers == 0:
                for i, type_char in enumerate(str(self.lora_type)):
                    A = As[:, i, depth // self.skip_layers]
                    B = B_all[:, i, depth // self.skip_layers]
                    lora_weights[type_char] = (A, B)
                for j in ['q', 'k', 'v', 'o', 'm']:
                    if j not in lora_weights:
                        lora_weights[j] = (None, None)
            else:
                for j in ['q', 'k', 'v', 'o', 'm']:
                    lora_weights[j] = (None, None)
            lora_weights_list.append(lora_weights)

        return lora_weights_list


def get_graph_lora_generater(model_args):
    method = getattr(model_args, 'generation_method', 'direct')

    common_kwargs = dict(
        dim=model_args.topolora_dim,
        depth=model_args.topolora_depth,
        node_feat_dim=model_args.node_feat_dim,
        graph_hidden_dim=model_args.graph_hidden_dim,
        pos_num=model_args.topolora_pos_num,
        llm_dim=model_args.topolora_llm_dim,
        llm_depth=model_args.topolora_llm_depth,
        lora_rank=model_args.topolora_rank,
        lora_type=model_args.topolora_type,
        weights_sep=model_args.weights_sep,
        skip_layers=model_args.skip_layers,
        topolora_alpha=model_args.topolora_alpha,
        gnn_type=getattr(model_args, 'gnn_type', 'GAT'),
        gnn_layers=getattr(model_args, 'gnn_layers', 3),
    )

    if method == 'direct':
        return GraphLoRAGenerater(**common_kwargs)
    elif method == 'hypernetwork':
        return HypernetworkLoRAGenerator(
            **common_kwargs,
            factor_k=getattr(model_args, 'hypernetwork_factor_k', 8),
            hyper_hidden_dim=getattr(model_args, 'hypernetwork_hidden_dim', 512),
        )
    elif method == 'prototype':
        return PrototypeLoRAGenerator(
            **common_kwargs,
            prototype_K=getattr(model_args, 'prototype_K', 4),
            proto_hidden_dim=getattr(model_args, 'prototype_hidden_dim', 256),
        )
    else:
        raise ValueError(f"Unknown generation_method: {method}. "
                         f"Must be 'direct', 'hypernetwork', or 'prototype'.")


if __name__ == "__main__":
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    result = graph_tokenize.load_pretrained_model_and_tokenize()
    adjacency_matrix = result['adj_matrix']
    node_features = result['motif_features']
    for i in range(adjacency_matrix.shape[0]):
        print(f'Graph {i} adjacency matrix shape: {adjacency_matrix[i].shape}')
        print(f'Graph {i} node features shape: {node_features[i].shape}')
        break
    print("Graph tokenization completed successfully.")
    
    pe = graph_tokenize.generate_sinusoidal_position_encoding(adjacency_matrix[0].shape[0], node_features[0].shape[1])
    print(f"Position encoding shape: {pe.shape}")
    
    
    class MockArgs:
        topolora_dim = 512
        topolora_depth = 6
        node_feat_dim = pe.shape[1]  #
        graph_hidden_dim = 256
        topolora_pos_num = 32
        topolora_llm_dim = 4096
        topolora_llm_depth = 32
        topolora_rank = 32
        topolora_type = 'qkvo'
        weights_sep = True
        skip_layers = 1
        topolora_alpha = 32
        gnn_type = 'GAT'
        gnn_layers = 3
    
    model_args = MockArgs()
    model = get_graph_lora_generater(model_args)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameter nums: {total_params/1e6:.2f}M')
    
    print("\n=== Generating LoRA weights ===")
    adj_tensor = adjacency_matrix[0].to(device)
    node_feat_tensor = pe.to(device)
    
    
    with torch.no_grad():
        lora_weights_list = model(node_feat_tensor, adj_tensor)
    
    print(f'Generated LoRA weights for {len(lora_weights_list)} layers')
    
    for layer_idx, lora_weights in enumerate(lora_weights_list):
        print(f'\nLayer {layer_idx}:')
        for weight_type, (A, B) in lora_weights.items():
            if A is not None and B is not None:
                print(f'  {weight_type}: A shape: {A.shape}, B shape: {B.shape}')
            else:
                print(f'  {weight_type}: (None, None)')
    
    non_none_layers = 0
    for lora_weights in lora_weights_list:
        if any(A is not None for A, B in lora_weights.values()):
            non_none_layers += 1
    
    print(f'\nSummary:')
    print(f'- Total layers: {len(lora_weights_list)}')
    print(f'- Layers with LoRA weights: {non_none_layers}')
    print(f'- Skip layers setting: {model_args.skip_layers}')
    print(f'- LoRA types: {model_args.topolora_type}')